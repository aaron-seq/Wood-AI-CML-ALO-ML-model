"""Tests for the optional API-key gate.

The property that matters most is the default: with no key configured the
API must behave exactly as it did before authentication existed, so
upgrading cannot lock an existing deployment out of its own service.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

KEY = "a-sufficiently-long-test-key"
WRITE_BODY = {
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "High-consequence area adjacent to a fired heater",
    "sme_name": "A. Engineer",
}

READ_ENDPOINTS = [
    ("GET", "/model/info"),
    ("GET", "/sme-override"),
]


@pytest.fixture
def keyed_client(client, monkeypatch):
    """A client whose API has a key configured at the default scope."""
    from app import main

    monkeypatch.setattr(main.settings, "API_KEY", KEY)
    monkeypatch.setattr(main.settings, "API_KEY_SCOPE", "writes")
    return client


@pytest.fixture
def strict_client(client, monkeypatch):
    """A client whose API gates reads as well as writes."""
    from app import main

    monkeypatch.setattr(main.settings, "API_KEY", KEY)
    monkeypatch.setattr(main.settings, "API_KEY_SCOPE", "all")
    return client


class TestDisabledByDefault:
    def test_no_key_configured_leaves_everything_open(self, client, csv_upload, valid_cml_frame):
        assert client.get("/model/info").status_code == 200
        assert client.get("/sme-override").status_code == 200
        assert client.post("/sme-override", json=WRITE_BODY).status_code == 201
        assert client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).status_code == 200

    def test_a_stray_key_header_is_ignored_when_auth_is_off(self, client):
        response = client.get("/sme-override", headers={"X-API-Key": "nonsense"})
        assert response.status_code == 200

    def test_health_reports_auth_disabled(self, client):
        assert client.get("/health").json()["auth"] == "disabled"


class TestWriteScope:
    def test_writes_require_the_key(self, keyed_client):
        assert keyed_client.post("/sme-override", json=WRITE_BODY).status_code == 401
        assert keyed_client.delete("/sme-override/CML-042").status_code == 401

    def test_writes_succeed_with_the_key(self, keyed_client):
        response = keyed_client.post("/sme-override", json=WRITE_BODY, headers={"X-API-Key": KEY})
        assert response.status_code == 201

    def test_a_wrong_key_is_rejected(self, keyed_client):
        response = keyed_client.post(
            "/sme-override", json=WRITE_BODY, headers={"X-API-Key": "wrong-but-long-enough"}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize("method,path", READ_ENDPOINTS)
    def test_reads_stay_open_at_this_scope(self, keyed_client, method, path):
        assert keyed_client.request(method, path).status_code == 200

    def test_scoring_stays_open_at_this_scope(self, keyed_client, csv_upload, valid_cml_frame):
        response = keyed_client.post("/score-cml-data", files=csv_upload(valid_cml_frame))
        assert response.status_code == 200

    def test_health_reports_the_scope(self, keyed_client):
        assert keyed_client.get("/health").json()["auth"] == "writes"


class TestAllScope:
    @pytest.mark.parametrize("method,path", READ_ENDPOINTS)
    def test_reads_require_the_key(self, strict_client, method, path):
        assert strict_client.request(method, path).status_code == 401
        assert strict_client.request(method, path, headers={"X-API-Key": KEY}).status_code == 200

    def test_scoring_requires_the_key(self, strict_client, csv_upload, valid_cml_frame):
        assert (
            strict_client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).status_code
            == 401
        )
        assert (
            strict_client.post(
                "/score-cml-data",
                files=csv_upload(valid_cml_frame),
                headers={"X-API-Key": KEY},
            ).status_code
            == 200
        )

    def test_forecasting_requires_the_key(self, strict_client, csv_upload, valid_cml_frame):
        response = strict_client.post("/forecast-remaining-life", files=csv_upload(valid_cml_frame))
        assert response.status_code == 401


class TestProbesStayReachable:
    """An orchestrator cannot send credentials to a liveness probe."""

    @pytest.mark.parametrize("path", ["/", "/health"])
    def test_open_at_write_scope(self, keyed_client, path):
        assert keyed_client.get(path).status_code == 200

    @pytest.mark.parametrize("path", ["/", "/health"])
    def test_open_even_at_all_scope(self, strict_client, path):
        assert strict_client.get(path).status_code == 200


class TestErrorShape:
    def test_missing_and_wrong_keys_are_indistinguishable(self, keyed_client):
        """The response must not help an attacker probe for valid keys."""
        missing = keyed_client.post("/sme-override", json=WRITE_BODY)
        wrong = keyed_client.post(
            "/sme-override", json=WRITE_BODY, headers={"X-API-Key": "wrong-but-long-enough"}
        )
        assert missing.status_code == wrong.status_code == 401
        assert missing.json() == wrong.json()

    def test_the_key_is_never_echoed(self, keyed_client):
        response = keyed_client.post(
            "/sme-override", json=WRITE_BODY, headers={"X-API-Key": "wrong-but-long-enough"}
        )
        assert KEY not in response.text
        assert "wrong-but-long-enough" not in response.text

    def test_challenge_header_is_present(self, keyed_client):
        response = keyed_client.post("/sme-override", json=WRITE_BODY)
        assert response.headers["www-authenticate"] == "X-API-Key"


class TestKeyValidation:
    def test_a_short_key_is_refused_at_startup(self, monkeypatch):
        """A guessable key is worse than none: it implies safety."""
        monkeypatch.setenv("API_KEY", "short")
        with pytest.raises(ValidationError, match="at least 16 characters"):
            Settings(_env_file=None)

    def test_a_long_key_is_accepted(self, monkeypatch):
        monkeypatch.setenv("API_KEY", KEY)
        assert Settings(_env_file=None).API_KEY == KEY

    def test_unset_is_the_default(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        settings = Settings(_env_file=None)
        assert settings.API_KEY is None
        assert settings.API_KEY_SCOPE == "writes"

    def test_an_unknown_scope_is_refused(self, monkeypatch):
        monkeypatch.setenv("API_KEY_SCOPE", "sometimes")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestDashboardClient:
    def test_headers_are_empty_without_a_key(self, monkeypatch):
        from api_client import auth_headers

        monkeypatch.delenv("CML_API_KEY", raising=False)
        assert auth_headers() == {}

    def test_headers_carry_the_key_when_set(self, monkeypatch):
        from api_client import auth_headers

        monkeypatch.setenv("CML_API_KEY", KEY)
        assert auth_headers() == {"X-API-Key": KEY}
