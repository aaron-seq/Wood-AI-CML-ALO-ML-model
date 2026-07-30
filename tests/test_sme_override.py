"""Tests for the SME override store."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.sme_override import SMEOverrideManager, create_override_manager


@pytest.fixture
def manager(tmp_path) -> SMEOverrideManager:
    return SMEOverrideManager(tmp_path / "overrides.json")


class TestLifecycle:
    def test_creates_its_store_on_first_use(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "overrides.json"
        SMEOverrideManager(path)
        assert path.exists()
        assert json.loads(path.read_text()) == []

    def test_add_and_fetch(self, manager):
        manager.add_override("CML-1", "KEEP", "Adjacent to a fired heater", "A. Eng")
        stored = manager.get_override("CML-1")
        assert stored["sme_decision"] == "KEEP"
        assert stored["override_date"]

    def test_fetch_unknown_id_returns_none(self, manager):
        assert manager.get_override("nope") is None

    def test_remove_returns_false_for_unknown_id(self, manager):
        assert manager.remove_override("nope") is False

    @pytest.mark.parametrize("decision", ["MAYBE", "keep", ""])
    def test_invalid_decision_is_rejected(self, manager, decision):
        with pytest.raises(ValueError, match="KEEP"):
            manager.add_override("CML-1", decision, "reason text", "A. Eng")

    def test_corrupt_store_reads_as_empty(self, tmp_path):
        path = tmp_path / "overrides.json"
        path.write_text("{not json")
        assert SMEOverrideManager(path).get_all_overrides() == []

    def test_factory_defaults_to_the_conventional_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert create_override_manager().override_file.name == "sme_overrides.json"


class TestApplyToPredictions:
    def test_marks_and_overrides_matching_rows(self, manager):
        manager.add_override("CML-1", "KEEP", "Adjacent to a fired heater", "A. Eng")
        predictions = pd.DataFrame(
            {"id_number": ["CML-1", "CML-2"], "recommendation": ["ELIMINATE", "KEEP"]}
        )

        result = manager.apply_overrides_to_predictions(predictions)

        assert bool(result.loc[0, "has_sme_override"]) is True
        assert result.loc[0, "final_decision"] == "KEEP"
        assert result.loc[0, "sme_name"] == "A. Eng"
        assert bool(result.loc[1, "has_sme_override"]) is False

    def test_frame_without_ids_is_returned_untouched(self, manager):
        manager.add_override("CML-1", "KEEP", "Adjacent to a fired heater", "A. Eng")
        predictions = pd.DataFrame({"other": [1, 2]})
        pd.testing.assert_frame_equal(
            manager.apply_overrides_to_predictions(predictions), predictions
        )

    def test_no_overrides_leaves_the_frame_untouched(self, manager):
        predictions = pd.DataFrame({"id_number": ["CML-1"]})
        pd.testing.assert_frame_equal(
            manager.apply_overrides_to_predictions(predictions), predictions
        )


class TestStatistics:
    def test_empty_store(self, manager):
        assert manager.get_override_statistics() == {
            "total_overrides": 0,
            "keep_overrides": 0,
            "eliminate_overrides": 0,
        }

    def test_agreement_rate_against_the_model(self, manager):
        manager.add_override(
            "CML-1", "KEEP", "Agrees with the model", "A. Eng", original_prediction="KEEP"
        )
        manager.add_override(
            "CML-2",
            "KEEP",
            "Disagrees with the model",
            "A. Eng",
            original_prediction="ELIMINATE",
        )

        stats = manager.get_override_statistics()

        assert stats["disagreements_with_ml"] == 1
        assert stats["agreement_rate"] == 50.0
        assert stats["sme_distribution"] == {"A. Eng": 2}
