"""Tests for structured logging, metrics, pagination and per-CML minimums."""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from app.features import (
    MINIMUM_THICKNESS_COLUMN,
    engineer_features,
    resolve_minimum_thickness,
)
from app.forecasting import CMLForecaster
from app.observability import JsonFormatter, Metrics


def _many_cmls(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_number": [f"CML-{index:03d}" for index in range(rows)],
            "average_corrosion_rate": [0.1] * rows,
            "thickness_mm": [9.0] * rows,
            "commodity": ["Crude Oil"] * rows,
            "feature_type": ["Pipe"] * rows,
            "cml_shape": ["Both"] * rows,
        }
    )


class TestJsonFormatter:
    def _record(self, **extra) -> logging.LogRecord:
        record = logging.LogRecord("app.test", logging.INFO, "f.py", 1, "scored %d", (7,), None)
        record.__dict__.update(extra)
        return record

    def test_emits_one_json_object_per_line(self):
        payload = json.loads(JsonFormatter().format(self._record()))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert payload["message"] == "scored 7"
        assert payload["timestamp"]

    def test_extra_fields_become_top_level_keys(self):
        payload = json.loads(
            JsonFormatter().format(self._record(request_id="abc123", duration_ms=4.2))
        )
        assert payload["request_id"] == "abc123"
        assert payload["duration_ms"] == 4.2

    def test_output_is_a_single_line(self):
        assert "\n" not in JsonFormatter().format(self._record(note="multi\nline"))

    def test_exceptions_are_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = self._record()
            record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exception"]


class TestMetrics:
    def test_counts_requests_by_method_path_and_status(self):
        metrics = Metrics()
        metrics.observe_request("GET", "/health", 200, 0.01)
        metrics.observe_request("GET", "/health", 200, 0.02)
        metrics.observe_request("GET", "/health", 500, 0.03)

        rendered = metrics.render()
        assert 'cml_requests_total{method="GET",path="/health",status="200"} 2' in rendered
        assert 'cml_requests_total{method="GET",path="/health",status="500"} 1' in rendered

    def test_accumulates_fractional_durations(self):
        metrics = Metrics()
        metrics.observe_request("GET", "/health", 200, 0.25)
        metrics.observe_request("GET", "/health", 200, 0.5)
        assert "0.750000" in metrics.render()

    def test_tracks_scoring_volume_and_overrides(self):
        metrics = Metrics()
        metrics.observe_scoring(cmls=500, overrides=3)
        metrics.observe_scoring(cmls=100, overrides=1)

        rendered = metrics.render()
        assert "cml_scored_total 600" in rendered
        assert "cml_sme_overrides_applied_total 4" in rendered

    def test_every_series_declares_help_and_type(self):
        metrics = Metrics()
        metrics.observe_request("GET", "/health", 200, 0.01)
        rendered = metrics.render()
        for name in (
            "cml_requests_total",
            "cml_request_duration_seconds_total",
            "cml_scored_total",
            "cml_sme_overrides_applied_total",
            "cml_process_uptime_seconds",
        ):
            assert f"# HELP {name} " in rendered
            assert f"# TYPE {name} " in rendered

    def test_reset_clears_counters(self):
        metrics = Metrics()
        metrics.observe_scoring(cmls=10, overrides=1)
        metrics.reset()
        assert "cml_scored_total 0" in metrics.render()


class TestMetricsEndpoint:
    def test_disabled_by_default(self, client):
        assert client.get("/metrics").status_code == 404

    def test_served_when_enabled(self, client, monkeypatch):
        from app import main

        monkeypatch.setattr(main.settings, "METRICS_ENABLED", True)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "cml_requests_total" in response.text

    def test_scoring_is_counted(self, client, csv_upload, valid_cml_frame, monkeypatch):
        from app import main

        monkeypatch.setattr(main.settings, "METRICS_ENABLED", True)
        main.metrics.reset()

        client.post("/score-cml-data", files=csv_upload(valid_cml_frame))
        assert "cml_scored_total 3" in client.get("/metrics").text

    def test_path_labels_use_the_route_template(self, client, monkeypatch):
        """Otherwise every CML id would mint its own metric series."""
        from app import main

        monkeypatch.setattr(main.settings, "METRICS_ENABLED", True)
        main.metrics.reset()

        client.delete("/sme-override/CML-001")
        client.delete("/sme-override/CML-002")

        rendered = client.get("/metrics").text
        assert 'path="/sme-override/{id_number}"' in rendered
        assert "CML-001" not in rendered


class TestPagination:
    def test_defaults_match_the_previous_cap(self, client, csv_upload):
        body = client.post("/score-cml-data", files=csv_upload(_many_cmls(150))).json()
        assert len(body["results"]) == 100
        assert body["offset"] == 0
        assert body["results_truncated"] is True
        assert body["total_results"] == 150

    def test_offset_and_limit_page_through_results(self, client, csv_upload):
        frame = _many_cmls(150)
        first = client.post("/score-cml-data?offset=0&limit=50", files=csv_upload(frame)).json()
        second = client.post("/score-cml-data?offset=50&limit=50", files=csv_upload(frame)).json()

        assert [row["id_number"] for row in first["results"]][:3] == [
            "CML-000",
            "CML-001",
            "CML-002",
        ]
        assert second["results"][0]["id_number"] == "CML-050"
        assert first["total_results"] == second["total_results"] == 150

    def test_the_whole_batch_is_reachable(self, client, csv_upload):
        """The old hard cap made rows 100+ unretrievable without re-uploading."""
        frame = _many_cmls(150)
        collected: list[str] = []
        for offset in (0, 50, 100):
            body = client.post(
                f"/score-cml-data?offset={offset}&limit=50", files=csv_upload(frame)
            ).json()
            collected += [row["id_number"] for row in body["results"]]

        assert collected == frame["id_number"].tolist()

    def test_a_full_page_is_not_reported_as_truncated(self, client, csv_upload):
        body = client.post("/score-cml-data?limit=150", files=csv_upload(_many_cmls(150))).json()
        assert len(body["results"]) == 150
        assert body["results_truncated"] is False

    def test_an_offset_past_the_end_returns_an_empty_page(self, client, csv_upload):
        body = client.post("/score-cml-data?offset=999", files=csv_upload(_many_cmls(10))).json()
        assert body["results"] == []
        assert body["total_results"] == 10

    @pytest.mark.parametrize("query", ["offset=-1", "limit=0"])
    def test_invalid_paging_is_rejected(self, client, csv_upload, query):
        response = client.post(f"/score-cml-data?{query}", files=csv_upload(_many_cmls(5)))
        assert response.status_code == 422


class TestPerCMLMinimumThickness:
    def test_absent_column_uses_the_default(self):
        frame = pd.DataFrame({"average_corrosion_rate": [0.1], "thickness_mm": [10.0]})
        assert resolve_minimum_thickness(frame).tolist() == [3.0]

    def test_supplied_values_win_row_by_row(self):
        frame = pd.DataFrame(
            {
                "average_corrosion_rate": [0.1, 0.1],
                "thickness_mm": [10.0, 10.0],
                MINIMUM_THICKNESS_COLUMN: [3.0, 8.0],
            }
        )
        life = engineer_features(frame)["remaining_life_years"]
        assert life.tolist() == pytest.approx([70.0, 20.0])

    @pytest.mark.parametrize("bad", [None, 0.0, -2.0, "not-a-number"])
    def test_unusable_values_fall_back_to_the_default(self, bad):
        frame = pd.DataFrame(
            {
                "average_corrosion_rate": [0.1],
                "thickness_mm": [10.0],
                MINIMUM_THICKNESS_COLUMN: [bad],
            }
        )
        assert resolve_minimum_thickness(frame).tolist() == [3.0]

    def test_the_forecaster_honours_the_column(self):
        frame = pd.DataFrame(
            {
                "id_number": ["A", "B"],
                "average_corrosion_rate": [0.1, 0.1],
                "thickness_mm": [10.0, 10.0],
                MINIMUM_THICKNESS_COLUMN: [3.0, 8.0],
            }
        )
        forecast = CMLForecaster().forecast_batch(frame)
        assert forecast["remaining_life_years"].tolist() == pytest.approx([50.0, 20.0])

    def test_the_forecast_endpoint_honours_the_column(self, client, csv_upload):
        frame = pd.DataFrame(
            {
                "id_number": ["A", "B"],
                "average_corrosion_rate": [0.1, 0.1],
                "thickness_mm": [10.0, 10.0],
                "commodity": ["Crude Oil"] * 2,
                "feature_type": ["Pipe"] * 2,
                "cml_shape": ["Both"] * 2,
                MINIMUM_THICKNESS_COLUMN: [3.0, 8.0],
            }
        )
        body = client.post("/forecast-remaining-life", files=csv_upload(frame)).json()
        assert [row["remaining_life_years"] for row in body] == pytest.approx([50.0, 20.0])

    def test_a_file_without_the_column_is_unchanged(self, client, csv_upload, valid_cml_frame):
        body = client.post("/forecast-remaining-life", files=csv_upload(valid_cml_frame)).json()
        assert len(body) == 3
        assert all(row["remaining_life_years"] >= 0 for row in body)
