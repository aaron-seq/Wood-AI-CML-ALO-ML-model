"""Property-based tests for the engineering calculations.

Example-based tests check the cases someone thought of. These state the
invariants that must hold for *every* input and let Hypothesis search for
a counterexample -- which is the right tool for numeric code whose inputs
come from field measurements and can be zero, negative, enormous or
missing.

The properties here are the ones a corrosion engineer would recognise:
thinning a wall never extends its life, a slower corrosion rate never
shortens it, and risk never decreases as an asset gets worse.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app import risk
from app.features import (
    MAX_REMAINING_LIFE_YEARS,
    corrosion_thickness_ratio,
    engineer_features,
    remaining_life_years,
)

# Ranges chosen to cover plausible field values plus the degenerate edges
# that broke the original code: zero and negative corrosion rates, walls
# thinner than the minimum, and absurdly large readings.
thicknesses = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
rates = st.floats(min_value=-1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
minimums = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
lives = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


def _life(thickness: float, rate: float, minimum: float = 3.0) -> float:
    return float(remaining_life_years(pd.Series([thickness]), pd.Series([rate]), minimum).iloc[0])


class TestRemainingLifeInvariants:
    @given(thickness=thicknesses, rate=rates, minimum=minimums)
    def test_is_always_finite_and_non_negative(self, thickness, rate, minimum):
        """The invariant the original inf-producing division violated."""
        life = _life(thickness, rate, minimum)
        assert life == life  # not NaN
        assert life != float("inf")
        assert life >= 0.0

    @given(thickness=thicknesses, rate=rates, minimum=minimums)
    def test_a_wall_at_or_below_the_minimum_has_no_life_left(self, thickness, rate, minimum):
        """True for *any* rate, including none at all.

        Hypothesis found this: the ceiling for a non-corroding wall was
        applied before the exhausted-wall check, so a wall already at its
        minimum reported 50 years where CMLForecaster reported zero.
        """
        assume(thickness <= minimum)
        assert _life(thickness, rate, minimum) == 0.0

    @given(thickness=thicknesses, minimum=minimums)
    def test_a_non_corroding_wall_reports_the_ceiling(self, thickness, minimum):
        assume(thickness > minimum)
        assert _life(thickness, 0.0, minimum) == MAX_REMAINING_LIFE_YEARS

    @given(thickness=thicknesses, rate=rates, minimum=minimums)
    def test_thinning_a_wall_never_extends_its_life(self, thickness, rate, minimum):
        """Monotonic in thickness: less metal cannot mean more time."""
        assume(rate > 0)
        thinner = _life(thickness, rate, minimum)
        thicker = _life(thickness + 1.0, rate, minimum)
        assert thicker >= thinner

    @given(thickness=thicknesses, rate=st.floats(min_value=0.01, max_value=100.0), minimum=minimums)
    def test_corroding_faster_never_extends_life(self, thickness, rate, minimum):
        """Monotonic in rate, in the other direction."""
        slower = _life(thickness, rate, minimum)
        faster = _life(thickness, rate * 2, minimum)
        assert faster <= slower

    @given(thickness=thicknesses, rate=st.floats(min_value=0.01, max_value=100.0), minimum=minimums)
    def test_a_stricter_minimum_never_extends_life(self, thickness, rate, minimum):
        """Raising the floor leaves less usable metal, so less time."""
        lenient = _life(thickness, rate, minimum)
        strict = _life(thickness, rate, minimum + 1.0)
        assert strict <= lenient

    @given(thickness=thicknesses, rate=st.floats(0.0, 1e-7), minimum=minimums)
    def test_an_immeasurably_small_rate_does_not_overflow(self, thickness, rate, minimum):
        """Hypothesis found this too.

        ``where=rate > 0`` admits denormal floats, and dividing by one
        overflows straight back to the infinity the guard exists to
        prevent -- so a corrosion rate of 1e-311 still failed the batch.
        """
        life = _life(thickness, rate, minimum)
        assert life != float("inf")
        assert life <= MAX_REMAINING_LIFE_YEARS

    @given(thickness=st.floats(min_value=3.1, max_value=1000.0), rate=st.floats(0.01, 100.0))
    def test_matches_the_closed_form_where_it_is_defined(self, thickness, rate):
        expected = (thickness - 3.0) / rate
        assert _life(thickness, rate) == expected


class TestCorrosionRatioInvariants:
    @given(rate=rates, thickness=thicknesses)
    def test_is_always_finite(self, rate, thickness):
        ratio = float(corrosion_thickness_ratio(pd.Series([rate]), pd.Series([thickness])).iloc[0])
        assert ratio == ratio
        assert abs(ratio) != float("inf")

    @given(rate=st.floats(min_value=0.0, max_value=100.0), thickness=st.floats(0.0, 1e-7))
    def test_an_immeasurably_thin_wall_does_not_overflow(self, rate, thickness):
        """`> 0` admits denormals; dividing by one overflows to infinity."""
        ratio = float(corrosion_thickness_ratio(pd.Series([rate]), pd.Series([thickness])).iloc[0])
        assert ratio == 0.0

    @given(rate=st.floats(min_value=0.0, max_value=100.0), thickness=thicknesses)
    def test_a_non_positive_wall_yields_zero(self, rate, thickness):
        assume(thickness <= 0)
        assert (
            float(corrosion_thickness_ratio(pd.Series([rate]), pd.Series([thickness])).iloc[0])
            == 0.0
        )


class TestRiskInvariants:
    @given(life=lives, rate=st.floats(0.0, 100.0), thickness=thicknesses)
    def test_always_returns_a_known_level(self, life, rate, thickness):
        assert risk.classify(life, rate, thickness) in risk.RISK_ORDER

    @given(life=lives, rate=st.floats(0.0, 100.0), thickness=thicknesses)
    def test_a_thin_wall_is_always_critical(self, life, rate, thickness):
        """Whatever the remaining life says -- the fix from ADR-0005."""
        assume(thickness < risk.CRITICAL_THICKNESS_MM)
        assert risk.classify(life, rate, thickness) == "CRITICAL"

    @given(life=lives, rate=st.floats(0.0, 100.0), thickness=thicknesses)
    def test_losing_life_never_lowers_risk(self, life, rate, thickness):
        """Monotonic in severity: an asset getting worse cannot get safer."""
        worse = risk.RISK_ORDER.index(risk.classify(life / 2, rate, thickness))
        better = risk.RISK_ORDER.index(risk.classify(life, rate, thickness))
        assert worse >= better

    @given(life=lives, rate=st.floats(0.01, 50.0), thickness=thicknesses)
    def test_corroding_faster_never_lowers_risk(self, life, rate, thickness):
        worse = risk.RISK_ORDER.index(risk.classify(life, rate * 2, thickness))
        better = risk.RISK_ORDER.index(risk.classify(life, rate, thickness))
        assert worse >= better

    @given(life=lives, rate=st.floats(0.0, 100.0), thickness=thicknesses)
    def test_the_vectorised_classifier_agrees_with_the_scalar_one(self, life, rate, thickness):
        vectorised = risk.classify_series(
            pd.Series([life]), pd.Series([rate]), pd.Series([thickness])
        ).iloc[0]
        assert vectorised == risk.classify(life, rate, thickness)


class TestInspectionIntervalInvariants:
    @given(life=lives, rate=st.floats(0.0, 100.0))
    def test_is_always_within_the_configured_bounds(self, life, rate):
        months = risk.inspection_interval_months(life, rate)
        assert months >= risk.MIN_INSPECTION_INTERVAL_MONTHS
        assert months <= risk.MAX_INSPECTION_INTERVAL_MONTHS

    @given(life=lives, rate=st.floats(0.0, 100.0), factor=st.floats(1.0, 10.0))
    def test_a_larger_safety_factor_never_lengthens_the_interval(self, life, rate, factor):
        assume(factor > 1.0)
        assert risk.inspection_interval_months(
            life, rate, safety_factor=factor
        ) <= risk.inspection_interval_months(life, rate, safety_factor=1.0)

    @given(life=lives, rate=st.floats(0.0, 100.0))
    def test_the_result_is_a_whole_number_of_months(self, life, rate):
        assert isinstance(risk.inspection_interval_months(life, rate), int)


class TestFeatureEngineeringInvariants:
    @settings(max_examples=50)
    @given(
        rows=st.lists(
            st.tuples(thicknesses, rates),
            min_size=1,
            max_size=25,
        )
    )
    def test_engineered_columns_are_always_finite(self, rows):
        """No input should be able to produce a value sklearn will reject."""
        frame = pd.DataFrame(
            {
                "id_number": [f"CML-{index}" for index in range(len(rows))],
                "thickness_mm": [thickness for thickness, _ in rows],
                "average_corrosion_rate": [rate for _, rate in rows],
            }
        )
        engineered = engineer_features(frame)
        numeric = engineered[["corrosion_thickness_ratio", "remaining_life_years", "risk_score"]]
        assert numeric.notna().all().all()
        assert (numeric.abs() != float("inf")).all().all()

    @settings(max_examples=50)
    @given(rows=st.lists(st.tuples(thicknesses, rates), min_size=1, max_size=25))
    def test_row_count_and_order_are_preserved(self, rows):
        frame = pd.DataFrame(
            {
                "id_number": [f"CML-{index}" for index in range(len(rows))],
                "thickness_mm": [thickness for thickness, _ in rows],
                "average_corrosion_rate": [rate for _, rate in rows],
            }
        )
        engineered = engineer_features(frame)
        assert len(engineered) == len(frame)
        assert engineered["id_number"].tolist() == frame["id_number"].tolist()

    @settings(max_examples=50)
    @given(rows=st.lists(st.tuples(thicknesses, rates), min_size=1, max_size=25))
    def test_risk_score_stays_within_its_declared_range(self, rows):
        frame = pd.DataFrame(
            {
                "thickness_mm": [thickness for thickness, _ in rows],
                "average_corrosion_rate": [rate for _, rate in rows],
            }
        )
        score = engineer_features(frame)["risk_score"]
        assert (score >= 0).all()
        assert (score <= 100).all()
