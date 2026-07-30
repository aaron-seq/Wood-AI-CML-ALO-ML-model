"""Regression tests for defects found during the modernisation audit.

Each test here corresponds to a bug that was observable in the shipped
code. They are grouped in one module so the cost of a regression is
obvious: if one of these fails, a previously-fixed production fault is
back.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.features import engineer_features, remaining_life_years
from app.forecasting import CMLForecaster
from app.sme_override import SMEOverrideManager


class TestClientErrorsAreNot500:
    """A caller mistake must not be reported as a server fault.

    Each endpoint raised HTTPException(400) inside a try block whose own
    ``except Exception`` caught it and re-raised it as a 500, so every
    rejected upload looked like an outage.
    """

    @pytest.mark.parametrize(
        "endpoint",
        ["/upload-cml-data", "/score-cml-data", "/forecast-remaining-life"],
    )
    def test_unsupported_extension_returns_400(self, client, csv_upload, valid_cml_frame, endpoint):
        response = client.post(endpoint, files=csv_upload(valid_cml_frame, filename="data.txt"))
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    @pytest.mark.parametrize(
        "endpoint",
        ["/upload-cml-data", "/score-cml-data", "/forecast-remaining-life"],
    )
    def test_empty_file_returns_400(self, client, endpoint):
        response = client.post(endpoint, files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_missing_required_columns_returns_400(self, client, csv_upload):
        partial = pd.DataFrame({"id_number": ["CML-001"], "thickness_mm": [9.5]})
        response = client.post("/score-cml-data", files=csv_upload(partial))
        assert response.status_code == 400
        assert "average_corrosion_rate" in response.json()["detail"]

    def test_oversized_upload_is_rejected(self, client):
        payload = io.BytesIO(b"x" * (26 * 1024 * 1024))
        response = client.post("/score-cml-data", files={"file": ("big.csv", payload, "text/csv")})
        assert response.status_code == 400
        assert "too large" in response.json()["detail"]


class TestZeroCorrosionRate:
    """A non-corroding CML must not fail the batch it travels in.

    ``(thickness - min) / rate`` produced ``inf`` for a zero rate, and
    scikit-learn rejects the whole matrix with "Input X contains
    infinity" -- so one benign row failed every other row's scoring.
    """

    def test_remaining_life_is_finite_at_zero_rate(self):
        life = remaining_life_years(pd.Series([10.0, 10.0]), pd.Series([0.0, 0.1]))
        assert life.notna().all()
        assert (life != float("inf")).all()
        assert life.iloc[0] == 50.0
        assert life.iloc[1] == pytest.approx(70.0)

    def test_engineered_features_are_finite_for_degenerate_inputs(self):
        df = pd.DataFrame(
            {
                "id_number": ["A", "B", "C"],
                "average_corrosion_rate": [0.0, -1.0, 0.1],
                "thickness_mm": [10.0, 0.0, 2.0],
            }
        )
        out = engineer_features(df)
        numeric = out[["corrosion_thickness_ratio", "remaining_life_years"]]
        assert numeric.notna().all().all()
        assert (numeric.abs() != float("inf")).all().all()

    def test_scoring_succeeds_with_a_zero_rate_row(self, client, csv_upload, valid_cml_frame):
        frame = valid_cml_frame.copy()
        frame.loc[0, "average_corrosion_rate"] = 0.0
        response = client.post("/score-cml-data", files=csv_upload(frame))
        assert response.status_code == 200
        assert response.json()["total_results"] == 3


class TestForecastBatchAlignment:
    """Forecasts must be aligned positionally, not joined on id_number.

    ``df.merge(forecasts, on="id_number")`` cross-joined duplicate ids, so
    a file with a repeated CML returned more forecasts than it had rows.
    """

    def test_duplicate_ids_do_not_multiply_rows(self):
        df = pd.DataFrame(
            {
                "id_number": ["A", "A", "B"],
                "thickness_mm": [10.0, 9.0, 8.0],
                "average_corrosion_rate": [0.1, 0.2, 0.3],
            }
        )
        result = CMLForecaster().forecast_batch(df)
        assert len(result) == len(df)
        assert result["id_number"].tolist() == ["A", "A", "B"]

    def test_forecast_column_wins_over_a_stale_input_column(self):
        """A pre-existing remaining_life_years must not shadow the forecast."""
        df = pd.DataFrame(
            {
                "id_number": ["A"],
                "thickness_mm": [10.0],
                "average_corrosion_rate": [0.5],
                "remaining_life_years": [999.0],  # stale value from the source file
            }
        )
        result = CMLForecaster().forecast_batch(df)
        assert result["remaining_life_years"].iloc[0] == pytest.approx(14.0)
        assert result["remaining_life_years_input"].iloc[0] == 999.0

    def test_summary_reports_the_forecast_not_the_input(self):
        df = pd.DataFrame(
            {
                "id_number": ["A", "B"],
                "thickness_mm": [10.0, 10.0],
                "average_corrosion_rate": [0.5, 0.5],
                "remaining_life_years": [999.0, 999.0],
            }
        )
        summary = CMLForecaster().generate_forecast_summary(df)
        assert summary["avg_remaining_life_years"] == pytest.approx(14.0)


class TestSMEOverrideStatistics:
    """Statistics must not crash once any override exists.

    ``override_date`` is stored as an ISO string and ``nlargest`` raises
    TypeError on a non-numeric column, so the statistics call failed for
    every non-empty override file -- including the one in the repository.
    """

    def test_statistics_with_stored_overrides(self, tmp_path):
        manager = SMEOverrideManager(tmp_path / "overrides.json")
        manager.add_override("CML-1", "KEEP", "High consequence area", "A. Engineer")
        manager.add_override("CML-2", "ELIMINATE", "Redundant with CML-3", "B. Engineer")

        stats = manager.get_override_statistics()

        assert stats["total_overrides"] == 2
        assert stats["keep_overrides"] == 1
        assert stats["eliminate_overrides"] == 1
        assert [item["id_number"] for item in stats["recent_overrides"]] == [
            "CML-2",
            "CML-1",
        ]

    def test_statistics_survive_an_unparseable_date(self, tmp_path):
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id_number": "CML-1",
                        "sme_decision": "KEEP",
                        "reason": "r",
                        "sme_name": "n",
                        "override_date": "not-a-date",
                    }
                ]
            )
        )
        stats = SMEOverrideManager(path).get_override_statistics()
        assert stats["total_overrides"] == 1
        assert len(stats["recent_overrides"]) == 1

    def test_statistics_on_the_repository_override_file(self):
        """The file committed to the repo must not blow up the SME endpoint."""
        from app.config import settings

        stats = SMEOverrideManager(settings.SME_OVERRIDE_FILE).get_override_statistics()
        assert "total_overrides" in stats


class TestResponseTruncationIsDeclared:
    """Truncated result lists must be flagged, not silently short."""

    def test_truncation_flag_is_set_for_large_files(self, client, csv_upload):
        rows = 150
        frame = pd.DataFrame(
            {
                "id_number": [f"CML-{i:03d}" for i in range(rows)],
                "average_corrosion_rate": [0.1] * rows,
                "thickness_mm": [9.0] * rows,
                "commodity": ["Crude Oil"] * rows,
                "feature_type": ["Pipe"] * rows,
                "cml_shape": ["Both"] * rows,
            }
        )
        body = client.post("/score-cml-data", files=csv_upload(frame)).json()

        assert body["total_results"] == rows
        assert len(body["results"]) == 100
        assert body["results_truncated"] is True

    def test_truncation_flag_is_false_for_small_files(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        assert body["results_truncated"] is False
        assert len(body["results"]) == body["total_results"] == 3


class TestPredictionRowAlignment:
    """Results must be matched to rows positionally.

    ``iterrows`` yields index *labels* while predictions are a positional
    numpy array; the two only coincide for a default RangeIndex.
    """

    def test_ids_line_up_with_predictions(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        returned = [row["id_number"] for row in body["results"]]
        assert returned == valid_cml_frame["id_number"].tolist()
