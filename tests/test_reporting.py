"""Tests for report generation and the file/JSON helpers in app.utils."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.utils import (
    generate_elimination_report,
    load_sme_overrides,
    save_predictions_to_csv,
    save_sme_override,
)


@pytest.fixture
def scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_number": ["CML-1", "CML-2", "CML-3", "CML-4"],
            "predicted_elimination": [1, 0, 1, 0],
            "elimination_probability": [0.92, 0.10, 0.55, 0.48],
            "recommendation": ["ELIMINATE", "KEEP", "ELIMINATE", "KEEP"],
            "confidence_level": ["HIGH", "HIGH", "MODERATE", "MODERATE"],
            "commodity": ["Crude Oil", "Steam", "Crude Oil", "Steam"],
            "feature_type": ["Pipe", "Elbow", "Pipe", "Tee"],
            "average_corrosion_rate": [0.1, 0.2, 0.3, 0.4],
            "thickness_mm": [9.0, 8.0, 7.0, 6.0],
        }
    )


class TestEliminationReport:
    def test_summary_totals_add_up(self, scored_frame):
        summary = generate_elimination_report(scored_frame)["summary"]
        assert summary["total_cmls"] == 4
        assert summary["recommended_eliminations"] == 2
        assert summary["recommended_keep"] == 2
        assert summary["elimination_rate"] == 50.0

    def test_groups_eliminations_by_commodity_and_feature(self, scored_frame):
        report = generate_elimination_report(scored_frame)
        assert report["elimination_by_commodity"] == {"Crude Oil": 2}
        assert report["elimination_by_feature"] == {"Pipe": 2}

    def test_marginal_cases_are_those_near_the_boundary(self, scored_frame):
        report = generate_elimination_report(scored_frame)
        marginal = {row["id_number"] for row in report["marginal_cases"]}
        assert marginal == {"CML-3", "CML-4"}

    def test_top_candidates_are_ordered_by_probability(self, scored_frame):
        top = generate_elimination_report(scored_frame)["top_elimination_candidates"]
        assert [row["id_number"] for row in top] == ["CML-1", "CML-3"]

    def test_empty_frame_reports_an_error(self):
        assert "error" in generate_elimination_report(pd.DataFrame())

    def test_missing_prediction_column_raises(self):
        with pytest.raises(ValueError, match="predicted_elimination"):
            generate_elimination_report(pd.DataFrame({"id_number": ["CML-1"]}))

    def test_optional_columns_may_be_absent(self):
        minimal = pd.DataFrame({"predicted_elimination": [1, 0]})
        report = generate_elimination_report(minimal)
        assert report["summary"]["total_cmls"] == 2
        assert "elimination_by_commodity" not in report


class TestSavePredictions:
    def test_writes_a_csv_and_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "out.csv"
        result = save_predictions_to_csv(pd.DataFrame({"a": [1]}), target)
        assert result == target
        assert pd.read_csv(target)["a"].tolist() == [1]

    def test_refuses_to_write_an_empty_frame(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_predictions_to_csv(pd.DataFrame(), tmp_path / "out.csv")


class TestOverrideJsonHelpers:
    def test_missing_file_loads_as_empty(self, tmp_path):
        assert load_sme_overrides(tmp_path / "absent.json") == []

    def test_round_trip(self, tmp_path):
        path = tmp_path / "overrides.json"
        save_sme_override({"id_number": "CML-1", "decision": "KEEP"}, path)
        save_sme_override({"id_number": "CML-2", "decision": "ELIMINATE"}, path)

        loaded = load_sme_overrides(path)
        assert [item["id_number"] for item in loaded] == ["CML-1", "CML-2"]

    def test_datetime_is_serialised_as_iso(self, tmp_path):
        from datetime import datetime

        path = tmp_path / "overrides.json"
        save_sme_override(
            {"id_number": "CML-1", "override_date": datetime(2024, 5, 1, 12, 0)}, path
        )
        assert load_sme_overrides(path)[0]["override_date"].startswith("2024-05-01")

    def test_invalid_json_propagates(self, tmp_path):
        path = tmp_path / "overrides.json"
        path.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            load_sme_overrides(path)
