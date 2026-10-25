"""Tests for single-CML scoring over JSON (issue #10).

The property that matters most is parity: a CML scored on its own must
get exactly the answer it gets inside a batch upload. Two scoring paths
that can disagree are how this codebase previously ended up with risk
levels that differed on 70% of CMLs.
"""

from __future__ import annotations

import math

import pytest

from app import main

RECORD = {
    "id_number": "CML-001",
    "average_corrosion_rate": 0.12,
    "thickness_mm": 9.5,
    "commodity": "Crude Oil",
    "feature_type": "Pipe",
    "cml_shape": "Internal",
}


def _batch_results(client, csv_upload, frame):
    body = client.post("/score-cml-data", files=csv_upload(frame)).json()
    return {result["id_number"]: result for result in body["results"]}


class TestPredict:
    def test_scores_a_single_record(self, client):
        response = client.post("/predict", json=RECORD)
        assert response.status_code == 200
        body = response.json()
        assert body["id_number"] == "CML-001"
        assert body["recommendation"] in {"KEEP", "ELIMINATE"}
        assert 0.0 <= body["elimination_probability"] <= 1.0
        assert body["sme_override"] is None
        assert isinstance(body["warnings"], list)

    def test_matches_batch_scoring_for_every_row(self, client, csv_upload, valid_cml_frame):
        batch = _batch_results(client, csv_upload, valid_cml_frame)
        for record in valid_cml_frame.to_dict("records"):
            single = client.post("/predict", json=record).json()
            single.pop("warnings")
            assert single == batch[str(record["id_number"])]

    def test_optional_inputs_are_honoured_exactly_as_in_a_batch(
        self, client, csv_upload, valid_cml_frame
    ):
        frame = valid_cml_frame.head(1).assign(
            minimum_thickness_mm=8.0, last_inspection_date="2024-01-15"
        )
        batch = _batch_results(client, csv_upload, frame)
        single = client.post("/predict", json=frame.to_dict("records")[0]).json()
        single.pop("warnings")
        assert single == batch[frame.loc[0, "id_number"]]

    @pytest.mark.parametrize(
        "messy",
        [
            pytest.param({"commodity": " Crude Oil "}, id="padded-category"),
            pytest.param({"id_number": "  CML-001 "}, id="padded-id"),
            pytest.param({"id_number": "00042"}, id="leading-zero-id"),
            pytest.param({"last_inspection_date": "2020-01-15T10:00:00"}, id="datetime-string"),
            # pandas guesses the order of an ambiguous date per value and warns
            # when it reads one day-first. Parity holds; the ambiguity itself
            # is pre-existing and tracked in #27.
            pytest.param(
                {"last_inspection_date": "15/01/2020"},
                id="day-first-date",
                marks=pytest.mark.filterwarnings("ignore:Parsing dates in:UserWarning"),
            ),
            pytest.param({"thickness_mm": 10}, id="integral-number"),
        ],
    )
    def test_matches_batch_scoring_on_messy_input(self, client, csv_upload, messy):
        """Found by review: the clean-fixture parity tests could not catch
        that /predict stripped whitespace and parsed ids and dates
        differently from an upload of the same values."""
        record = {**RECORD, **messy}
        import pandas as pd

        batch = client.post("/score-cml-data", files=csv_upload(pd.DataFrame([record]))).json()[
            "results"
        ][0]
        single = client.post("/predict", json=record).json()
        single.pop("warnings")
        assert single == batch

    def test_an_sme_override_supersedes_the_model(self, client):
        model_said = client.post("/predict", json=RECORD).json()["model_recommendation"]
        decision = "KEEP" if model_said == "ELIMINATE" else "ELIMINATE"
        client.post(
            "/sme-override",
            json={
                "id_number": "CML-001",
                "sme_decision": decision,
                "reason": "Inspected in the field last turnaround",
                "sme_name": "A. Engineer",
            },
        )
        body = client.post("/predict", json=RECORD).json()
        assert body["recommendation"] == decision
        assert body["model_recommendation"] == model_said
        assert body["sme_override"]["sme_decision"] == decision

    def test_out_of_range_values_warn_like_a_batch_does(self, client):
        body = client.post("/predict", json={**RECORD, "thickness_mm": 900.0}).json()
        assert any("thickness" in warning for warning in body["warnings"])

    @pytest.mark.parametrize(
        "change",
        [
            {"thickness_mm": None},
            {"average_corrosion_rate": "fast"},
            {"commodity": ""},
            {"id_number": "C" * 65},
            {"cml_shape": "Internal\nINFO forged"},
            {"unexpected_field": 1},
            {"last_inspection_date": 1579046400},
        ],
    )
    def test_rejects_malformed_records(self, client, change):
        assert client.post("/predict", json={**RECORD, **change}).status_code == 422

    def test_rejects_a_missing_required_field(self, client):
        record = {key: value for key, value in RECORD.items() if key != "commodity"}
        assert client.post("/predict", json=record).status_code == 422

    @pytest.mark.parametrize("value", [math.inf, math.nan])
    def test_rejects_non_finite_numbers(self, client, value):
        """json.dumps emits Infinity/NaN, which Python's parser accepts."""
        import json

        payload = json.dumps({**RECORD, "average_corrosion_rate": value})
        response = client.post(
            "/predict", content=payload, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    def test_is_unavailable_without_a_model(self, client, monkeypatch):
        monkeypatch.setattr(main, "model", None)
        assert client.post("/predict", json=RECORD).status_code == 503
