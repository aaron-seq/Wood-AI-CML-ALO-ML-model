"""Tests for the shared feature-engineering module."""

from __future__ import annotations

import pandas as pd
import pytest

from app.features import (
    BASE_NUMERIC_COLUMNS,
    DEFAULT_DAYS_SINCE_INSPECTION,
    ENGINEERED_COLUMNS,
    MAX_REMAINING_LIFE_YEARS,
    corrosion_thickness_ratio,
    days_since_inspection,
    engineer_features,
    fallback_risk_score,
    remaining_life_years,
)


class TestRemainingLife:
    def test_uses_the_api_570_formula(self):
        life = remaining_life_years(pd.Series([10.0]), pd.Series([0.5]))
        assert life.iloc[0] == pytest.approx((10.0 - 3.0) / 0.5)

    def test_respects_a_custom_minimum_thickness(self):
        life = remaining_life_years(pd.Series([10.0]), pd.Series([0.5]), 5.0)
        assert life.iloc[0] == pytest.approx(10.0)

    @pytest.mark.parametrize("rate", [0.0, -0.5])
    def test_non_positive_rate_yields_the_ceiling(self, rate):
        assert remaining_life_years(pd.Series([10.0]), pd.Series([rate])).iloc[0] == (
            MAX_REMAINING_LIFE_YEARS
        )

    def test_wall_below_minimum_yields_zero(self):
        assert remaining_life_years(pd.Series([2.0]), pd.Series([0.5])).iloc[0] == 0.0

    def test_values_above_the_ceiling_are_not_capped(self):
        """Training used the uncapped column; capping here would skew serving."""
        life = remaining_life_years(pd.Series([20.0]), pd.Series([0.01]))
        assert life.iloc[0] == pytest.approx(1700.0)

    def test_non_numeric_input_becomes_zero_not_nan(self):
        life = remaining_life_years(pd.Series(["abc"]), pd.Series([0.5]))
        assert life.iloc[0] == 0.0


class TestCorrosionThicknessRatio:
    def test_ordinary_case(self):
        ratio = corrosion_thickness_ratio(pd.Series([0.5]), pd.Series([10.0]))
        assert ratio.iloc[0] == pytest.approx(0.05)

    @pytest.mark.parametrize("thickness", [0.0, -1.0])
    def test_non_positive_thickness_yields_zero(self, thickness):
        ratio = corrosion_thickness_ratio(pd.Series([0.5]), pd.Series([thickness]))
        assert ratio.iloc[0] == 0.0


class TestDaysSinceInspection:
    def test_computes_age_against_a_fixed_reference(self):
        age = days_since_inspection(pd.Series(["2024-01-01"]), now=pd.Timestamp("2024-01-31"))
        assert age.iloc[0] == 30

    @pytest.mark.parametrize("value", [None, "not-a-date"])
    def test_unusable_dates_fall_back_to_one_year(self, value):
        age = days_since_inspection(pd.Series([value]), now=pd.Timestamp("2024-01-31"))
        assert age.iloc[0] == DEFAULT_DAYS_SINCE_INSPECTION


class TestFallbackRiskScore:
    def test_is_bounded_to_zero_and_one_hundred(self):
        score = fallback_risk_score(pd.Series([0.0, 10.0]), pd.Series([100.0, 0.0]))
        assert score.min() >= 0
        assert score.max() <= 100


class TestEngineerFeatures:
    def test_adds_every_engineered_column(self, valid_cml_frame):
        out = engineer_features(valid_cml_frame)
        for column in ENGINEERED_COLUMNS:
            assert column in out.columns

    def test_does_not_mutate_the_input(self, valid_cml_frame):
        before = valid_cml_frame.copy()
        engineer_features(valid_cml_frame)
        pd.testing.assert_frame_equal(valid_cml_frame, before)

    def test_existing_risk_score_is_preserved(self, valid_cml_frame):
        frame = valid_cml_frame.assign(risk_score=[11, 22, 33])
        assert engineer_features(frame)["risk_score"].tolist() == [11, 22, 33]

    def test_missing_last_inspection_date_defaults(self, valid_cml_frame):
        out = engineer_features(valid_cml_frame)
        assert (out["days_since_inspection"] == DEFAULT_DAYS_SINCE_INSPECTION).all()

    @pytest.mark.parametrize("column", BASE_NUMERIC_COLUMNS)
    def test_missing_base_column_raises_keyerror(self, valid_cml_frame, column):
        with pytest.raises(KeyError, match=column):
            engineer_features(valid_cml_frame.drop(columns=[column]))

    def test_works_on_a_non_default_index(self):
        """Guards against positional/label confusion in the helpers."""
        df = pd.DataFrame(
            {"average_corrosion_rate": [0.1, 0.2], "thickness_mm": [10.0, 8.0]},
            index=["x", "y"],
        )
        out = engineer_features(df)
        assert out.index.tolist() == ["x", "y"]
        assert out["remaining_life_years"].notna().all()
