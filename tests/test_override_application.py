"""Tests that recorded expert decisions actually reach the output.

Regression: ``apply_overrides_to_predictions`` existed and was tested, but
no endpoint ever called it. An engineer could record "KEEP this CML" and
every subsequent scoring run still returned ELIMINATE -- the
human-in-the-loop was decorative.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from app.sme_override import SMEOverrideManager

OVERRIDE = {
    "id_number": "CML-001",
    "sme_decision": "KEEP",
    "reason": "High-consequence area adjacent to a fired heater",
    "sme_name": "A. Engineer",
}


class TestOverridesReachScoring:
    def test_scoring_without_overrides_follows_the_model(self, client, csv_upload, valid_cml_frame):
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        assert body["sme_overrides_applied"] == 0
        for result in body["results"]:
            assert result["sme_override"] is None
            assert result["recommendation"] == result["model_recommendation"]

    def test_an_override_changes_the_recommendation(self, client, csv_upload, valid_cml_frame):
        baseline = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        target = baseline["results"][0]
        opposite = "KEEP" if target["model_recommendation"] == "ELIMINATE" else "ELIMINATE"

        client.post(
            "/sme-override",
            json={**OVERRIDE, "id_number": target["id_number"], "sme_decision": opposite},
        )
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()

        overridden = next(r for r in body["results"] if r["id_number"] == target["id_number"])
        assert overridden["recommendation"] == opposite
        assert overridden["model_recommendation"] == target["model_recommendation"]
        assert overridden["sme_override"]["sme_name"] == "A. Engineer"
        assert body["sme_overrides_applied"] == 1

    def test_the_raw_model_output_is_still_reported(self, client, csv_upload, valid_cml_frame):
        """Overriding must not erase what the model said -- that is the audit trail."""
        baseline = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        target = baseline["results"][0]
        opposite = "KEEP" if target["model_recommendation"] == "ELIMINATE" else "ELIMINATE"

        client.post(
            "/sme-override",
            json={**OVERRIDE, "id_number": target["id_number"], "sme_decision": opposite},
        )
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        overridden = next(r for r in body["results"] if r["id_number"] == target["id_number"])

        assert overridden["predicted_elimination_flag"] == target["predicted_elimination_flag"]
        assert overridden["elimination_probability"] == pytest.approx(
            target["elimination_probability"]
        )

    def test_untouched_cmls_are_unaffected(self, client, csv_upload, valid_cml_frame):
        client.post("/sme-override", json={**OVERRIDE, "id_number": "CML-001"})
        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()

        for result in body["results"]:
            if result["id_number"] != "CML-001":
                assert result["sme_override"] is None
                assert result["recommendation"] == result["model_recommendation"]

    def test_withdrawing_an_override_restores_the_model_decision(
        self, client, csv_upload, valid_cml_frame
    ):
        baseline = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        target = baseline["results"][0]
        opposite = "KEEP" if target["model_recommendation"] == "ELIMINATE" else "ELIMINATE"

        client.post(
            "/sme-override",
            json={**OVERRIDE, "id_number": target["id_number"], "sme_decision": opposite},
        )
        client.delete(f"/sme-override/{target['id_number']}")

        body = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        restored = next(r for r in body["results"] if r["id_number"] == target["id_number"])
        assert restored["recommendation"] == target["model_recommendation"]
        assert restored["sme_override"] is None


class TestOverridesReachReporting:
    def test_report_totals_count_the_final_decisions(self, client, csv_upload, valid_cml_frame):
        before = client.post("/generate-report", files=csv_upload(valid_cml_frame)).json()
        eliminations_before = before["summary"]["recommended_eliminations"]

        scored = client.post("/score-cml-data", files=csv_upload(valid_cml_frame)).json()
        kept = next((r for r in scored["results"] if r["model_recommendation"] == "KEEP"), None)
        if kept is None:
            pytest.skip("model eliminated every row in the fixture")

        client.post(
            "/sme-override",
            json={**OVERRIDE, "id_number": kept["id_number"], "sme_decision": "ELIMINATE"},
        )

        after = client.post("/generate-report", files=csv_upload(valid_cml_frame)).json()
        assert after["summary"]["recommended_eliminations"] == eliminations_before + 1
        assert (
            after["summary"]["recommended_eliminations"] + after["summary"]["recommended_keep"]
            == after["summary"]["total_cmls"]
        )


def _write_override(args: tuple[str, str]) -> None:
    """Top-level so it can be pickled for the process pool."""
    path, cml_id = args
    from pathlib import Path

    SMEOverrideManager(Path(path)).add_override(
        cml_id, "KEEP", "Concurrent write test record", "A. Engineer"
    )


class TestDurableWrites:
    """The override file is a compliance artifact, not scratch space."""

    def test_the_store_is_replaced_atomically(self, tmp_path, monkeypatch):
        manager = SMEOverrideManager(tmp_path / "overrides.json")
        manager.add_override("CML-1", "KEEP", "Original decision recorded", "A. Engineer")

        # Fail mid-serialisation. json.dump writes incrementally, so a
        # direct-to-target write would leave a truncated file behind.
        def explode(*args, **kwargs):
            raise RuntimeError("disk gave up")

        monkeypatch.setattr(json, "dump", explode)
        with pytest.raises(RuntimeError):
            manager.add_override("CML-2", "KEEP", "This write should fail", "B. Engineer")

        monkeypatch.undo()
        surviving = SMEOverrideManager(tmp_path / "overrides.json").get_all_overrides()
        assert [item["id_number"] for item in surviving] == ["CML-1"]

    def test_a_failed_write_leaves_no_temp_files(self, tmp_path, monkeypatch):
        manager = SMEOverrideManager(tmp_path / "overrides.json")
        monkeypatch.setattr(
            json, "dump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
        )
        with pytest.raises(RuntimeError):
            manager.add_override("CML-1", "KEEP", "Should not persist", "A. Engineer")
        monkeypatch.undo()

        strays = [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]
        assert not strays, f"temp files left behind: {strays}"

    def test_concurrent_writers_do_not_lose_records(self, tmp_path):
        """Without locking, parallel read-modify-write cycles drop decisions."""
        path = tmp_path / "overrides.json"
        SMEOverrideManager(path)

        ids = [f"CML-{index:03d}" for index in range(12)]
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(_write_override, [(str(path), cml_id) for cml_id in ids]))

        stored = {item["id_number"] for item in SMEOverrideManager(path).get_all_overrides()}
        assert stored == set(ids)


class TestOverrideMap:
    def test_keys_are_strings(self, tmp_path):
        """CML ids arrive from CSV as strings; the map must match on them."""
        manager = SMEOverrideManager(tmp_path / "overrides.json")
        manager.add_override("123", "KEEP", "Numeric-looking identifier", "A. Engineer")
        assert "123" in manager.get_override_map()

    def test_empty_store_yields_an_empty_map(self, tmp_path):
        assert SMEOverrideManager(tmp_path / "overrides.json").get_override_map() == {}

    def test_apply_to_predictions_still_works_on_frames(self, tmp_path):
        manager = SMEOverrideManager(tmp_path / "overrides.json")
        manager.add_override("CML-1", "KEEP", "High-consequence area", "A. Engineer")
        frame = pd.DataFrame(
            {"id_number": ["CML-1", "CML-2"], "recommendation": ["ELIMINATE", "KEEP"]}
        )
        result = manager.apply_overrides_to_predictions(frame)
        assert result.loc[0, "final_decision"] == "KEEP"
