"""CML Remaining Life Forecasting Module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from app import risk
from app.features import MAX_REMAINING_LIFE_YEARS, MINIMUM_THICKNESS_COLUMN


class CMLForecaster:
    """Forecast remaining life and inspection schedules for CMLs."""

    def __init__(
        self,
        minimum_thickness: float = 3.0,
        safety_factor: float = 1.5,
        min_inspection_interval_months: int = 12,
        max_inspection_interval_months: int = 72,
    ):
        self.minimum_thickness = minimum_thickness
        self.safety_factor = safety_factor
        self.min_inspection_interval = min_inspection_interval_months
        self.max_inspection_interval = max_inspection_interval_months

    def calculate_remaining_life(
        self,
        current_thickness: float,
        corrosion_rate: float,
        min_thickness: float | None = None,
    ) -> float:
        """Calculate remaining life in years."""
        if min_thickness is None:
            min_thickness = self.minimum_thickness

        available_thickness = current_thickness - min_thickness

        if available_thickness <= 0:
            return 0.0

        if corrosion_rate <= 0:
            return MAX_REMAINING_LIFE_YEARS

        remaining_life = available_thickness / corrosion_rate
        return min(remaining_life, MAX_REMAINING_LIFE_YEARS)

    def calculate_inspection_interval(
        self, remaining_life_years: float, corrosion_rate: float
    ) -> int:
        """Recommended inspection interval in months.

        Delegates to app.risk so the dashboard and the
        /forecast-remaining-life endpoint cannot disagree.
        """
        return risk.inspection_interval_months(
            remaining_life_years,
            corrosion_rate,
            self.safety_factor,
            min_months=self.min_inspection_interval,
            max_months=self.max_inspection_interval,
        )

    def calculate_next_inspection_date(
        self, last_inspection_date: datetime, interval_months: int
    ) -> datetime:
        """Calculate next inspection date."""
        return last_inspection_date + timedelta(days=interval_months * 30.44)

    def estimate_future_thickness(
        self, current_thickness: float, corrosion_rate: float, years_ahead: float
    ) -> float:
        """Estimate thickness at future date."""
        future_thickness = current_thickness - (corrosion_rate * years_ahead)
        return max(0, future_thickness)

    def calculate_risk_level(
        self,
        remaining_life_years: float,
        corrosion_rate: float,
        current_thickness: float,
    ) -> str:
        """Determine the risk level for a CML. See app.risk for the rule."""
        return risk.classify(remaining_life_years, corrosion_rate, current_thickness)

    def forecast_single_cml(
        self,
        id_number: str,
        current_thickness: float,
        corrosion_rate: float,
        last_inspection_date: datetime | None = None,
        minimum_thickness: float | None = None,
    ) -> dict:
        """Generate complete forecast for a single CML."""
        if last_inspection_date is None:
            last_inspection_date = datetime.now()

        remaining_life = self.calculate_remaining_life(
            current_thickness, corrosion_rate, minimum_thickness
        )

        inspection_interval = self.calculate_inspection_interval(remaining_life, corrosion_rate)

        next_inspection = self.calculate_next_inspection_date(
            last_inspection_date, inspection_interval
        )

        years_until_inspection = inspection_interval / 12
        thickness_at_inspection = self.estimate_future_thickness(
            current_thickness, corrosion_rate, years_until_inspection
        )

        risk_level = self.calculate_risk_level(remaining_life, corrosion_rate, current_thickness)

        return {
            "id_number": id_number,
            "remaining_life_years": round(remaining_life, 1),
            "next_inspection_date": next_inspection.date(),
            "recommended_inspection_frequency_months": inspection_interval,
            "estimated_thickness_at_next_inspection": round(thickness_at_inspection, 2),
            "risk_level": risk_level,
            "current_thickness_mm": current_thickness,
            "corrosion_rate_mm_per_year": corrosion_rate,
        }

    def forecast_batch(
        self, df: pd.DataFrame, minimum_thickness: float | None = None
    ) -> pd.DataFrame:
        """Generate forecasts for multiple CMLs."""
        forecasts: list[dict] = []

        for _, row in df.iterrows():
            last_inspection = None
            if "last_inspection_date" in row:
                try:
                    parsed_date = pd.to_datetime(row["last_inspection_date"], errors="coerce")
                    # Check if parsing succeeded (not NaT)
                    if pd.notna(parsed_date):
                        last_inspection = parsed_date
                except Exception:
                    pass  # Use None if date parsing fails

            # A per-row minimum allowable thickness, where the source
            # data carries one, beats both the argument and the instance
            # default -- real programmes set this per circuit.
            row_minimum = minimum_thickness
            supplied = row.get(MINIMUM_THICKNESS_COLUMN)
            if supplied is not None and pd.notna(supplied) and float(supplied) > 0:
                row_minimum = float(supplied)

            forecast = self.forecast_single_cml(
                id_number=row.get("id_number", f"CML-{len(forecasts) + 1}"),
                current_thickness=row["thickness_mm"],
                corrosion_rate=row["average_corrosion_rate"],
                last_inspection_date=last_inspection,
                minimum_thickness=row_minimum,
            )
            forecasts.append(forecast)

        # One forecast was produced per input row, in input order, so the
        # two frames are aligned positionally. Joining on ``id_number``
        # instead would multiply rows whenever an id repeats -- a file
        # with a duplicated CML id returned more forecasts than it had
        # CMLs, silently inflating every downstream count.
        forecast_df = pd.DataFrame(forecasts, index=df.index)

        # Where a source column shares a name with a forecast column, the
        # forecast keeps the canonical name and the input value is
        # preserved under an "_input" suffix. Previously the merge kept
        # the *source* value under the canonical name, so a file that
        # already carried a remaining_life_years column made
        # generate_forecast_summary report the input rather than the
        # forecast it claims to summarise.
        overlapping = {
            column: f"{column}_input" for column in forecast_df.columns if column in df.columns
        }
        return pd.concat([df.rename(columns=overlapping), forecast_df], axis=1)

    def generate_forecast_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for forecasts."""
        forecast_df = self.forecast_batch(df)

        summary = {
            "total_cmls": len(forecast_df),
            "risk_distribution": forecast_df["risk_level"].value_counts().to_dict(),
            "avg_remaining_life_years": float(forecast_df["remaining_life_years"].mean()),
            "min_remaining_life_years": float(forecast_df["remaining_life_years"].min()),
            "critical_cmls": len(forecast_df[forecast_df["risk_level"] == "CRITICAL"]),
            "high_risk_cmls": len(forecast_df[forecast_df["risk_level"] == "HIGH"]),
            "inspections_needed_next_12_months": len(
                forecast_df[
                    pd.to_datetime(forecast_df["next_inspection_date"], errors="coerce")
                    < datetime.now() + timedelta(days=365)
                ]
            ),
            "top_priority_inspections": forecast_df.nsmallest(10, "remaining_life_years")[
                [
                    "id_number",
                    "remaining_life_years",
                    "next_inspection_date",
                    "risk_level",
                ]
            ].to_dict("records")
            if "id_number" in forecast_df.columns
            else [],
        }

        return summary


def forecast_cml_life(df: pd.DataFrame, minimum_thickness: float = 3.0) -> pd.DataFrame:
    """Convenience function to forecast remaining life for CML data."""
    forecaster = CMLForecaster(minimum_thickness=minimum_thickness)
    return forecaster.forecast_batch(df)
