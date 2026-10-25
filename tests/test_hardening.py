"""Regression tests for the hardening review of the API surface.

Each class corresponds to one finding: a gap between what the
documentation promised and what the code enforced, or input that reached
a log, a response or a file without a bound.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.routing import APIRoute

from app import main
from app.security import require_api_key, require_api_key_for_reads
from app.utils import neutralise_spreadsheet_formulas

KEY = "a-sufficiently-long-test-key"
VALID_OVERRIDE = {
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "High-consequence area adjacent to a fired heater",
    "sme_name": "A. Engineer",
}

# Never gated, at any scope, so orchestrator probes keep working.
UNGATED = {"/", "/health"}


@pytest.fixture
def strict(client, monkeypatch):
    monkeypatch.setattr(main.settings, "API_KEY", KEY)
    monkeypatch.setattr(main.settings, "API_KEY_SCOPE", "all")
    monkeypatch.setattr(main.settings, "METRICS_ENABLED", True)
    return client


class TestScopeAllGatesEveryDataEndpoint:
    """API_KEY_SCOPE=all was documented as gating everything but / and
    /health; /metrics, which reports traffic and override volumes, was open."""

    def test_metrics_requires_the_key(self, strict):
        assert strict.get("/metrics").status_code == 401
        assert strict.get("/metrics", headers={"X-API-Key": KEY}).status_code == 200

    def test_every_route_but_the_probes_carries_a_gate(self):
        """Structural, so a route added later without a gate fails here."""
        gates = {require_api_key, require_api_key_for_reads}
        ungated = [
            route.path
            for route in main.app.routes
            if isinstance(route, APIRoute)
            and route.path not in UNGATED
            and not gates & {dep.call for dep in route.dependant.dependencies}
        ]
        assert ungated == []

    def test_health_does_not_disclose_the_install_path(self, client):
        model_path = client.get("/health").json()["model_path"]
        assert model_path is None or ("/" not in model_path and "\\\\" not in model_path)


class TestOverrideFieldsAreBounded:
    """Override fields reached the log and the audit trail unbounded: a 5 MB
    id was accepted, and a newline in sme_name forged a second log line."""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("id_number", "C" * 65),
            ("id_number", "CML-1\nINFO forged"),
            ("id_number", "   "),
            ("sme_name", ""),
            ("sme_name", "x" * 129),
            ("sme_name", "A. Engineer\nINFO - Recorded SME override for CML-999 by admin"),
            ("sme_name", "A. Engineer\x1b[2J"),
            ("reason", " " * 12),
            ("reason", "x" * 2001),
            ("original_prediction", "maybe"),
        ],
    )
    def test_rejects(self, client, field, value):
        body = {**VALID_OVERRIDE, field: value}
        assert client.post("/sme-override", json=body).status_code == 422

    def test_a_multiline_reason_is_still_accepted(self, client):
        body = {**VALID_OVERRIDE, "reason": "Line one of the rationale.\nLine two."}
        assert client.post("/sme-override", json=body).status_code == 201

    def test_surrounding_whitespace_is_stripped(self, client):
        body = {**VALID_OVERRIDE, "id_number": "  CML-042  ", "sme_name": " A. Engineer "}
        stored = client.post("/sme-override", json=body).json()["override"]
        assert (stored["id_number"], stored["sme_name"]) == ("CML-042", "A. Engineer")

    def test_an_id_containing_a_slash_can_be_withdrawn(self, client):
        """Plant tags such as 10-P-1001/A are real. They could be recorded
        but never deleted, because the slash split the URL path."""
        body = {**VALID_OVERRIDE, "id_number": "10-P-1001/A"}
        assert client.post("/sme-override", json=body).status_code == 201
        assert client.delete("/sme-override/10-P-1001/A").status_code == 200


class TestServerErrorsDoNotLeakInternals:
    def test_a_persistence_failure_hides_the_os_error(self, client, monkeypatch):
        def fail(**_):
            raise OSError(17, "File exists", "/srv/secret/data/notadir.txt")

        monkeypatch.setattr(main.sme_manager, "add_override", fail)
        response = client.post("/sme-override", json=VALID_OVERRIDE)
        assert response.status_code == 500
        assert "/srv" not in response.text and "Errno" not in response.text
        assert response.headers["x-request-id"] in response.json()["detail"]

    def test_an_inference_failure_hides_the_exception(
        self, client, monkeypatch, csv_upload, valid_cml_frame
    ):
        def fail(*_):
            raise RuntimeError("feature mismatch in /srv/secret/model.joblib")

        monkeypatch.setattr(main, "_predict", fail)
        response = client.post("/score-cml-data", files=csv_upload(valid_cml_frame))
        assert response.status_code == 500
        assert "/srv" not in response.text


class TestRequestBounds:
    def test_limit_is_bounded(self, client, csv_upload, valid_cml_frame):
        response = client.post(
            "/score-cml-data", params={"limit": 10**12}, files=csv_upload(valid_cml_frame)
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("supplied", ["x" * 200, "has space", "a\tb"])
    def test_an_unusable_request_id_is_replaced(self, client, supplied):
        echoed = client.get("/health", headers={"x-request-id": supplied}).headers["x-request-id"]
        assert echoed != supplied and len(echoed) <= 64

    def test_a_well_formed_request_id_is_kept(self, client):
        echoed = client.get("/health", headers={"x-request-id": "trace-01HZX.9"}).headers
        assert echoed["x-request-id"] == "trace-01HZX.9"


class TestSpreadsheetFormulaNeutralising:
    """Exports carry values from uploaded files. A cell such as
    =HYPERLINK(...) executes when the CSV is opened in a spreadsheet."""

    @pytest.mark.parametrize("dangerous", ["=1+1", "+1", "-cmd", "@SUM(A1)", "\t=1", "\r=1"])
    def test_text_cells_that_start_a_formula_are_prefixed(self, dangerous):
        out = neutralise_spreadsheet_formulas(pd.DataFrame({"id_number": [dangerous]}))
        assert out.loc[0, "id_number"] == "'" + dangerous

    def test_numbers_and_ordinary_text_are_untouched(self):
        frame = pd.DataFrame({"id_number": ["CML-1"], "days": [-417], "rate": [-0.5]})
        out = neutralise_spreadsheet_formulas(frame)
        assert out.to_dict("records") == frame.to_dict("records")

    def test_the_input_frame_is_not_modified(self):
        frame = pd.DataFrame({"id_number": ["=1"]})
        neutralise_spreadsheet_formulas(frame)
        assert frame.loc[0, "id_number"] == "=1"


class TestNonFiniteJsonIsAClientError:
    """Python's JSON parser accepts Infinity and NaN. Pydantic rejected them,
    but FastAPI's default 422 handler echoed the input back and could not
    serialise it, so any float field turned a bad request into a 500."""

    def test_on_an_existing_route(self, client):
        import json

        body = json.dumps({**VALID_OVERRIDE, "original_probability": float("inf")})
        response = client.post(
            "/sme-override", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][-1] == "original_probability"


class TestTextLogsCannotBeForged:
    """Values from requests -- an upload's filename, ids inside the file --
    reach log messages. In the default text format a newline in one forged
    a convincing second log line. JSON output was already safe."""

    @pytest.mark.parametrize("separator", ["\n", "\r", "\x1b[2J", " ", " ", "\x85"])
    def test_control_and_line_separators_are_escaped(self, separator):
        import logging

        from app.observability import TextFormatter

        record = logging.LogRecord(
            "app.main",
            logging.INFO,
            __file__,
            1,
            "Parsed upload %s",
            (f"x{separator}FORGED",),
            None,
        )
        line = TextFormatter("%(message)s").format(record)
        assert separator not in line
        assert "FORGED" in line  # escaped, not dropped

    def test_tracebacks_keep_their_lines(self):
        import logging
        import sys

        from app.observability import TextFormatter

        try:
            raise RuntimeError("boom")
        except RuntimeError:
            exc_info = sys.exc_info()
        record = logging.LogRecord("app", logging.ERROR, __file__, 1, "failed", (), exc_info)
        assert "\nRuntimeError: boom" in TextFormatter("%(message)s").format(record)

    def test_the_default_format_uses_it(self, monkeypatch):
        import logging

        from app.observability import TextFormatter, configure_logging

        monkeypatch.setattr(main.settings, "LOG_FORMAT", "text")
        configure_logging()
        assert all(isinstance(h.formatter, TextFormatter) for h in logging.getLogger().handlers)


class TestOddCredentialsAreRejectedNot500:
    def test_a_non_ascii_key_is_a_401(self, client, monkeypatch):
        """compare_digest raises TypeError on non-ASCII str."""
        monkeypatch.setattr(main.settings, "API_KEY", KEY)
        response = client.post(
            "/sme-override", json=VALID_OVERRIDE, headers={"X-API-Key": b"\xe9" * 24}
        )
        assert response.status_code == 401


class TestRemainingDisclosuresAndLimits:
    def test_model_info_reports_the_file_name_not_the_install_path(self, client):
        assert "/" not in client.get("/model/info").json()["model_path"]

    def test_a_trailing_newline_does_not_sneak_into_the_request_id(self):
        """re.match with $ accepts a trailing newline. Unreachable over HTTP,
        where servers strip it, but reachable from any raw ASGI caller."""
        assert main._request_id_from("abc\n") != "abc\n"
        assert main._request_id_from("abc") == "abc"

    def test_the_dashboard_upload_cap_is_no_larger_than_the_apis(self):
        import tomllib
        from pathlib import Path

        config = tomllib.loads(
            (Path(__file__).resolve().parents[1] / ".streamlit/config.toml").read_text()
        )
        assert config["server"]["maxUploadSize"] <= main.settings.MAX_UPLOAD_BYTES // (1024 * 1024)
