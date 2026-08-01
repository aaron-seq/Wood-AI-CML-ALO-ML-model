"""Tests for the Streamlit dashboard.

650 lines of UI had no test coverage at all, which is how it came to
display a hardcoded "System Status: Operational" regardless of whether
the API was reachable, and a "Model Version" string that was not a version
of anything.

``streamlit.testing.v1.AppTest`` runs the script in-process and exposes
the rendered widgets, so the pages can be exercised without a browser.
The API is stubbed: these test the dashboard's own behaviour, and the
endpoints have their own tests.
"""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = "streamlit_app.py"
STARTUP_TIMEOUT = 30


@pytest.fixture
def offline_api(monkeypatch):
    """Every outbound call fails, as if the API were down."""
    import api_client

    monkeypatch.setattr(api_client, "check_api_health", lambda *a, **k: False)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_number": ["CML-001", "CML-002", "CML-003"],
            "average_corrosion_rate": [0.12, 0.08, 0.30],
            "thickness_mm": [9.5, 12.0, 4.5],
            "commodity": ["Crude Oil", "Natural Gas", "Steam"],
            "feature_type": ["Pipe", "Elbow", "Tee"],
            "cml_shape": ["Both", "Internal", "External"],
        }
    )


def _run(page: str = "Overview", **session_state) -> AppTest:
    """Render one page of the dashboard and return the result."""
    app = AppTest.from_file(APP, default_timeout=STARTUP_TIMEOUT)
    for key, value in session_state.items():
        app.session_state[key] = value
    app.run()
    if page != "Overview":
        app.radio[0].set_value(page).run()
    return app


class TestItStarts:
    def test_the_app_runs_without_exceptions(self):
        app = _run()
        assert not app.exception

    def test_navigation_offers_every_page(self):
        app = _run()
        assert set(app.radio[0].options) == {
            "Overview",
            "Upload & Analyze",
            "Forecasting",
            "SME Overrides",
            "Reports",
            "How It Works",
            "About Application",
        }

    @pytest.mark.parametrize(
        "page",
        [
            "Overview",
            "Upload & Analyze",
            "Forecasting",
            "SME Overrides",
            "Reports",
            "How It Works",
            "About Application",
        ],
    )
    def test_every_page_renders(self, page, offline_api):
        """Including with no data loaded and the API unreachable."""
        app = _run(page)
        assert not app.exception


class TestOverviewReportsRealState:
    def test_api_status_is_probed_not_asserted(self, offline_api):
        """It used to say "Operational" unconditionally."""
        app = _run("Overview")
        statuses = [metric.value for metric in app.metric]
        assert "Offline" in statuses
        assert "Operational" not in statuses

    def test_it_says_where_it_looked_when_offline(self, offline_api):
        app = _run("Overview")
        captions = " ".join(element.value for element in app.caption)
        assert "No response from" in captions

    def test_model_is_reported_from_the_api_not_hardcoded(self, offline_api):
        """ "RF-Ensemble v2.1" was a string, not a version of anything."""
        app = _run("Overview")
        values = [metric.value for metric in app.metric]
        assert "RF-Ensemble v2.1" not in values
        assert "Unknown" in values

    def test_dataset_size_is_shown_once_loaded(self, offline_api, sample_frame):
        app = _run("Overview", data=sample_frame)
        assert any("3 Records" in metric.value for metric in app.metric)

    def test_prompts_for_data_when_none_is_loaded(self, offline_api):
        app = _run("Overview")
        assert any("Upload & Analyze" in warning.value for warning in app.warning)


class TestPagesThatNeedData:
    @pytest.mark.parametrize("page", ["Forecasting", "Reports"])
    def test_they_ask_for_data_rather_than_erroring(self, page, offline_api):
        app = _run(page)
        assert not app.exception
        assert app.warning, f"{page} should tell the user data is required"

    def test_forecasting_runs_on_loaded_data(self, offline_api, sample_frame):
        app = _run("Forecasting", data=sample_frame)
        assert not app.exception

        app.button[0].click().run()
        assert not app.exception
        metrics = {metric.label for metric in app.metric}
        assert "Critical CMLs" in metrics

    def test_reports_renders_the_analytics(self, offline_api, sample_frame):
        app = _run("Reports", data=sample_frame)
        assert not app.exception
        assert any(metric.label == "Total Assets Analyzed" for metric in app.metric)


class TestScoringResultsPresentation:
    def _analysis(self, *, truncated: bool, override: dict | None = None):
        return {
            "total_results": 150 if truncated else 2,
            "results_truncated": truncated,
            "sme_overrides_applied": 1 if override else 0,
            "results": [
                {
                    "id_number": "CML-001",
                    "predicted_elimination_flag": 1,
                    "elimination_probability": 0.91,
                    "model_recommendation": "ELIMINATE",
                    "recommendation": "ELIMINATE",
                    "confidence": "HIGH",
                    "sme_override": override,
                },
                {
                    "id_number": "CML-002",
                    "predicted_elimination_flag": 0,
                    "elimination_probability": 0.12,
                    "model_recommendation": "KEEP",
                    "recommendation": "KEEP",
                    "confidence": "HIGH",
                    "sme_override": None,
                },
            ],
        }

    def test_a_truncated_view_says_so(self, offline_api, sample_frame):
        """Counts over the first 100 rows were presented as dataset totals."""
        app = _run(
            "Upload & Analyze",
            data=sample_frame,
            analysis_results=self._analysis(truncated=True),
        )
        assert not app.exception
        assert any("150" in info.value for info in app.info)

    def test_a_complete_view_does_not(self, offline_api, sample_frame):
        app = _run(
            "Upload & Analyze",
            data=sample_frame,
            analysis_results=self._analysis(truncated=False),
        )
        assert not app.exception
        assert not any("Showing the first" in info.value for info in app.info)

    def test_elimination_and_keep_counts_are_shown(self, offline_api, sample_frame):
        app = _run(
            "Upload & Analyze",
            data=sample_frame,
            analysis_results=self._analysis(truncated=False),
        )
        labels = {metric.label: metric.value for metric in app.metric}
        assert labels["Candidates for Elimination"] == "1"
        assert labels["Critical Monitoring Points"] == "1"
