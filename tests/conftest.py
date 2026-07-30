"""Shared fixtures.

The API keeps a module-level ``SMEOverrideManager`` pointed at the
repository's ``data/sme_overrides.json``. Tests must never write there, so
:func:`api_client_fixture` redirects it at a temporary file for the
duration of each test.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.sme_override import SMEOverrideManager

VALID_ROW = {
    "id_number": "CML-001",
    "average_corrosion_rate": 0.12,
    "thickness_mm": 9.5,
    "commodity": "Crude Oil",
    "feature_type": "Pipe",
    "cml_shape": "Both",
}


@pytest.fixture
def valid_cml_frame() -> pd.DataFrame:
    """A small, schema-complete CML dataset."""
    return pd.DataFrame(
        {
            "id_number": ["CML-001", "CML-002", "CML-003"],
            "average_corrosion_rate": [0.12, 0.08, 0.25],
            "thickness_mm": [9.5, 10.2, 6.0],
            "commodity": ["Crude Oil", "Natural Gas", "Steam"],
            "feature_type": ["Pipe", "Elbow", "Reducer"],
            "cml_shape": ["Both", "Internal", "External"],
        }
    )


@pytest.fixture
def csv_upload():
    """Build a multipart payload from a DataFrame."""

    def _build(df: pd.DataFrame, filename: str = "cml.csv") -> dict:
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return {"file": (filename, buffer, "text/csv")}

    return _build


@pytest.fixture(name="client")
def api_client_fixture(tmp_path, monkeypatch) -> TestClient:
    """A TestClient with SME overrides isolated to a temporary file.

    Uses the context-manager form so the lifespan handler runs and the
    model is actually loaded, matching production startup.
    """
    monkeypatch.setattr(main, "sme_manager", SMEOverrideManager(tmp_path / "sme_overrides.json"))
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client
