"""Endpoint contract tests.

These assert the *specific* status codes and body shapes clients depend
on. The previous suite accepted ``status_code in [200, 500]``, which
passed whether or not scoring worked.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestMeta:
    def test_root_advertises_the_docs(self, client):
        body = client.get("/").json()
        assert body["documentation"] == "/docs"
        assert body["version"]

    def test_health_reports_a_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_path"].endswith(".joblib")

    def test_health_reports_degraded_without_a_model(self, client, monkeypatch):
        from app import main

        monkeypatch.setattr(main, "model", None)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_model_info_lists_the_fitted_features(self, client):
        body = client.get("/model/info").json()
        assert body["model_type"] == "Pipeline"
        assert "average_corrosion_rate" in body["features_used"]
        assert "corrosion_thickness_ratio" in body["features_used"]

    def test_openapi_schema_is_generated(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/health").headers["x-request-id"]

    def test_a_supplied_request_id_is_echoed(self, client):
        response = client.get("/health", headers={"x-request-id": "trace-me"})
        assert response.headers["x-request-id"] == "trace-me"


class TestUpload:
    def test_returns_schema_and_preview(self, client, csv_upload, valid_cml_frame):
        body = client.post("/upload-cml-data", files=csv_upload(valid_cml_frame)).json()
        assert body["rows"] == 3
        assert "average_corrosion_rate" in body["columns"]
        assert len(body["preview"]) == 3
        assert body["validation"]["valid"] is True

    def test_reports_validation_errors_for_an_incomplete_schema(self, client, csv_upload):
        """Upload is a dry run: it reports problems rather than rejecting."""
        partial = pd.DataFrame({"id_number": ["CML-001"], "thickness_mm": [9.5]})
        response = client.post("/upload-cml-data", files=csv_upload(partial))
        assert response.status_code == 200
        assert response.json()["validation"]["valid"] is False
        assert response.json()["validation"]["errors"]

    def test_missing_cells_are_serialised_as_null(self, client, csv_upload):
        frame = pd.DataFrame(
            {
                "id_number": ["CML-001"],
                "average_corrosion_rate": [0.1],
                "thickness_mm": [None],
                "commodity": ["Crude Oil"],
                "feature_type": ["Pipe"],
                "cml_shape": ["Both"],
            }
        )
        response = client.post("/upload-cml-data", files=csv_upload(frame))
        assert response.status_code == 200
        assert response.json()["preview"][0]["thickness_mm"] is None


class TestScoring:
    def test_scores_every_row(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        assert body["rows_scored"] == 3
        assert body["total_results"] == 3

    def test_result_fields_are_within_contract(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        for result in body["results"]:
            assert result["predicted_elimination_flag"] in (0, 1)
            assert 0.0 <= result["elimination_probability"] <= 1.0
            assert result["recommendation"] in ("KEEP", "ELIMINATE")
            assert result["confidence"] in ("HIGH", "MODERATE")

    def test_recommendation_agrees_with_the_flag(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        for result in body["results"]:
            expected = "ELIMINATE" if result["predicted_elimination_flag"] == 1 else "KEEP"
            assert result["recommendation"] == expected

    def test_returns_503_when_no_model_is_loaded(
        self, client, csv_upload, valid_cml_frame, monkeypatch
    ):
        from app import main

        monkeypatch.setattr(main, "model", None)
        response = client.post("/score-cml-data", files=csv_upload(valid_cml_frame))
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

    def test_scores_the_bundled_sample_dataset(self, client):
        with open("data/cml_sample_500.csv", "rb") as handle:
            response = client.post("/score-cml-data", files={"file": ("s.csv", handle, "text/csv")})
        assert response.status_code == 200
        assert response.json()["total_results"] == 500


class TestForecasting:
    def test_returns_one_forecast_per_row(self, client, csv_upload, valid_cml_frame):
        body = client.post("/forecast-remaining-life", files=csv_upload(valid_cml_frame)).json()
        assert len(body) == 3

    def test_forecast_fields_are_within_contract(self, client, csv_upload, valid_cml_frame):
        body = client.post("/forecast-remaining-life", files=csv_upload(valid_cml_frame)).json()
        for forecast in body:
            assert forecast["remaining_life_years"] >= 0
            assert forecast["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert forecast["recommended_inspection_frequency_months"] >= 1
            assert forecast["next_inspection_date"]

    def test_forecasting_needs_no_model(self, client, csv_upload, valid_cml_frame, monkeypatch):
        """Remaining life is deterministic arithmetic, not a prediction."""
        from app import main

        monkeypatch.setattr(main, "model", None)
        response = client.post("/forecast-remaining-life", files=csv_upload(valid_cml_frame))
        assert response.status_code == 200


class TestReporting:
    def test_report_summarises_the_dataset(self, client, csv_upload, valid_cml_frame):
        body = client.post("/generate-report", files=csv_upload(valid_cml_frame)).json()
        summary = body["summary"]
        assert summary["total_cmls"] == 3
        assert (
            summary["recommended_eliminations"] + summary["recommended_keep"]
            == summary["total_cmls"]
        )
        assert 0 <= summary["elimination_rate"] <= 100
        assert body["generated_from"]

    def test_report_requires_a_model(self, client, csv_upload, valid_cml_frame, monkeypatch):
        from app import main

        monkeypatch.setattr(main, "model", None)
        response = client.post("/generate-report", files=csv_upload(valid_cml_frame))
        assert response.status_code == 503


class TestSMEOverrides:
    OVERRIDE = {
        "id_number": "CML-042",
        "sme_decision": "KEEP",
        "reason": "High consequence area adjacent to a fired heater",
        "sme_name": "A. Engineer",
    }

    def test_create_then_list(self, client):
        assert client.post("/sme-override", json=self.OVERRIDE).status_code == 201

        body = client.get("/sme-override").json()
        assert body["statistics"]["total_overrides"] == 1
        assert body["overrides"][0]["id_number"] == "CML-042"

    def test_reposting_the_same_id_replaces_rather_than_duplicates(self, client):
        client.post("/sme-override", json=self.OVERRIDE)
        client.post("/sme-override", json={**self.OVERRIDE, "sme_decision": "ELIMINATE"})

        body = client.get("/sme-override").json()
        assert body["statistics"]["total_overrides"] == 1
        assert body["overrides"][0]["sme_decision"] == "ELIMINATE"

    def test_delete_removes_the_record(self, client):
        client.post("/sme-override", json=self.OVERRIDE)
        assert client.delete("/sme-override/CML-042").status_code == 200
        assert client.get("/sme-override").json()["statistics"]["total_overrides"] == 0

    def test_deleting_an_unknown_id_is_404(self, client):
        assert client.delete("/sme-override/NOPE").status_code == 404

    @pytest.mark.parametrize(
        "field,value",
        [
            ("sme_decision", "MAYBE"),  # not in the allowed pattern
            ("reason", "short"),  # below the 10-character minimum
        ],
    )
    def test_invalid_payloads_are_rejected(self, client, field, value):
        response = client.post("/sme-override", json={**self.OVERRIDE, field: value})
        assert response.status_code == 422

    def test_overrides_are_persisted_across_manager_instances(self, client, tmp_path):
        from app import main
        from app.sme_override import SMEOverrideManager

        client.post("/sme-override", json=self.OVERRIDE)
        reloaded = SMEOverrideManager(main.sme_manager.override_file)
        assert reloaded.get_override("CML-042")["sme_decision"] == "KEEP"


class TestCORS:
    def test_configured_origin_is_allowed(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:8501"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"

    def test_unlisted_origin_gets_no_allow_header(self, client):
        response = client.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in response.headers
