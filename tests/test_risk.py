"""Tests for the shared risk classifier.

The most important property here is agreement: every code path that
reports a risk level must produce the same answer for the same CML.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import risk
from app.forecasting import CMLForecaster
from app.utils import calculate_inspection_schedule


class TestClassify:
    @pytest.mark.parametrize(
        "life,rate,thickness,expected",
        [
            (0.5, 0.1, 10.0, "CRITICAL"),  # almost no life left
            (20.0, 0.1, 4.0, "CRITICAL"),  # thin wall, whatever the life
            (2.0, 0.1, 10.0, "HIGH"),
            (20.0, 0.30, 10.0, "HIGH"),  # corroding fast
            (5.0, 0.1, 10.0, "MEDIUM"),
            (20.0, 0.18, 10.0, "MEDIUM"),
            (20.0, 0.05, 10.0, "LOW"),
        ],
    )
    def test_levels(self, life, rate, thickness, expected):
        assert risk.classify(life, rate, thickness) == expected

    def test_a_thin_wall_is_critical_even_when_barely_corroding(self):
        """The case the old remaining-life-only rule got dangerously wrong.

        A 3.5 mm wall over a 3.0 mm minimum reads as decades of remaining
        life at a low corrosion rate, but has almost no material left.
        """
        life = (3.5 - 3.0) / 0.01
        assert life > 40
        assert risk.classify(life, 0.01, 3.5) == "CRITICAL"

    def test_severity_is_monotonic_in_remaining_life(self):
        order = [risk.RISK_ORDER.index(risk.classify(life, 0.1, 10.0)) for life in range(1, 30)]
        assert order == sorted(order, reverse=True)


class TestClassifySeries:
    def test_matches_the_scalar_classifier(self):
        frame = pd.DataFrame(
            {
                "life": [0.5, 2.0, 5.0, 20.0, 20.0],
                "rate": [0.1, 0.1, 0.1, 0.30, 0.05],
                "thickness": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        vectorised = risk.classify_series(frame["life"], frame["rate"], frame["thickness"])
        scalar = [risk.classify(*row) for row in frame.itertuples(index=False)]
        assert vectorised.tolist() == scalar

    def test_returns_a_severity_ordered_categorical(self):
        result = risk.classify_series(pd.Series([20.0]), pd.Series([0.05]), pd.Series([10.0]))
        assert list(result.cat.categories) == list(risk.RISK_ORDER)
        assert result.cat.ordered

    def test_preserves_the_index(self):
        result = risk.classify_series(
            pd.Series([20.0, 1.0], index=["x", "y"]),
            pd.Series([0.05, 0.05], index=["x", "y"]),
            pd.Series([10.0, 10.0], index=["x", "y"]),
        )
        assert result.index.tolist() == ["x", "y"]


class TestInspectionInterval:
    def test_is_clamped_to_the_bounds(self):
        assert risk.inspection_interval_months(0.1, 0.1) == risk.MIN_INSPECTION_INTERVAL_MONTHS
        assert risk.inspection_interval_months(500.0, 0.1) == risk.MAX_INSPECTION_INTERVAL_MONTHS

    def test_fast_corrosion_shortens_the_interval(self):
        slow = risk.inspection_interval_months(20.0, 0.10)
        fast = risk.inspection_interval_months(20.0, 0.30)
        assert fast <= slow

    def test_custom_bounds_are_honoured(self):
        assert risk.inspection_interval_months(500.0, 0.1, max_months=24) == 24
        assert risk.inspection_interval_months(0.1, 0.1, min_months=6) == 6

    def test_rejects_an_unsafe_safety_factor(self):
        with pytest.raises(ValueError, match="safety factor"):
            risk.inspection_interval_months(10.0, 0.1, safety_factor=0.5)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="exceeds"):
            risk.inspection_interval_months(10.0, 0.1, min_months=48, max_months=12)


class TestNoDivergenceBetweenCodePaths:
    """Regression: the API and the dashboard disagreed on 70% of CMLs.

    ``calculate_inspection_schedule`` backs /forecast-remaining-life and
    ``CMLForecaster`` backs the dashboard's Forecasting page. They were
    independent implementations with different thresholds, so the same
    asset was reported at two different risk levels depending on which
    surface the user looked at.
    """

    def test_agreement_across_a_grid_of_realistic_cmls(self):
        forecaster = CMLForecaster()
        thicknesses = [x / 2 for x in range(7, 41)]  # 3.5 - 20.0 mm
        rates = [x / 100 for x in range(1, 51)]  # 0.01 - 0.50 mm/yr

        mismatches = []
        for thickness in thicknesses:
            for rate in rates:
                schedule = calculate_inspection_schedule(corrosion_rate=rate, thickness=thickness)
                life = forecaster.calculate_remaining_life(thickness, rate)
                if schedule["risk_level"] != forecaster.calculate_risk_level(life, rate, thickness):
                    mismatches.append((thickness, rate, "risk"))
                if schedule[
                    "inspection_interval_months"
                ] != forecaster.calculate_inspection_interval(life, rate):
                    mismatches.append((thickness, rate, "interval"))

        assert not mismatches, f"{len(mismatches)} divergences, e.g. {mismatches[:3]}"

    def test_analytics_agrees_with_the_scalar_classifier(self):
        from app.advanced_analytics import calculate_advanced_statistics

        frame = pd.DataFrame(
            {
                "id_number": ["A", "B", "C"],
                "average_corrosion_rate": [0.30, 0.10, 0.02],
                "thickness_mm": [4.0, 9.0, 14.0],
                "commodity": ["Steam", "Crude Oil", "Fuel Gas"],
                "feature_type": ["Pipe", "Elbow", "Tee"],
                "remaining_life_years": [0.5, 20.0, 40.0],
            }
        )
        distribution = calculate_advanced_statistics(frame)["risk_distribution"]

        expected: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for row in frame.itertuples():
            level = risk.classify(
                row.remaining_life_years, row.average_corrosion_rate, row.thickness_mm
            )
            expected[level.lower()] += 1

        assert distribution == expected
