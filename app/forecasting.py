"""CML Remaining Life Forecasting Module."""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


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
        min_thickness: float = None,
    ) -> float:
        """Calculate remaining life in years."""
        if min_thickness is None:
            min_thickness = self.minimum_thickness

        available_thickness = current_thickness - min_thickness

        if available_thickness <= 0:
            return 0.0

        if corrosion_rate <= 0:
            return 50.0

        remaining_life = available_thickness / corrosion_rate
        return min(remaining_life, 50.0)

    def calculate_inspection_interval(
        self, remaining_life_years: float, corrosion_rate: float
    ) -> int:
        """Calculate recommended inspection interval in months."""
        safe_interval_years = remaining_life_years / self.safety_factor

        interval_months = int(safe_interval_years * 12)

        # Apply corrosion rate adjustment
        if corrosion_rate > 0.20:
            interval_months = int(interval_months * 0.7)
        elif corrosion_rate < 0.05:
            interval_months = int(interval_months * 1.3)

        # Clamp to min/max
        interval_months = max(self.min_inspection_interval, interval_months)
        interval_months = min(self.max_inspection_interval, interval_months)

        return interval_months

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
        """Determine risk level for CML."""
        if remaining_life_years < 1 or current_thickness < 5:
            return "CRITICAL"
        elif remaining_life_years < 3 or corrosion_rate > 0.25:
            return "HIGH"
        elif remaining_life_years < 7 or corrosion_rate > 0.15:
            return "MEDIUM"
        else:
            return "LOW"

    def forecast_single_cml(
        self,
        id_number: str,
        current_thickness: float,
        corrosion_rate: float,
        last_inspection_date: datetime = None,
        minimum_thickness: float = None,
    ) -> Dict:
        """Generate complete forecast for a single CML."""
        if last_inspection_date is None:
            last_inspection_date = datetime.now()

        remaining_life = self.calculate_remaining_life(
            current_thickness, corrosion_rate, minimum_thickness
        )

        inspection_interval = self.calculate_inspection_interval(
            remaining_life, corrosion_rate
        )

        next_inspection = self.calculate_next_inspection_date(
            last_inspection_date, inspection_interval
        )

        years_until_inspection = inspection_interval / 12
        thickness_at_inspection = self.estimate_future_thickness(
            current_thickness, corrosion_rate, years_until_inspection
        )

        risk_level = self.calculate_risk_level(
            remaining_life, corrosion_rate, current_thickness
        )

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
        self, df: pd.DataFrame, minimum_thickness: float = None
    ) -> pd.DataFrame:
        """Generate forecasts for multiple CMLs."""
        forecasts = []

        for _, row in df.iterrows():
            last_inspection = None
            if "last_inspection_date" in row and pd.notna(row["last_inspection_date"]):
                try:
                    last_inspection = pd.to_datetime(
                        row["last_inspection_date"], errors="coerce"
                    )
                    if pd.isna(last_inspection):
                        last_inspection = None
                except Exception:
                    last_inspection = None

            forecast = self.forecast_single_cml(
                id_number=row.get("id_number", f"CML-{len(forecasts) + 1}"),
                current_thickness=row["thickness_mm"],
                corrosion_rate=row["average_corrosion_rate"],
                last_inspection_date=last_inspection,
                minimum_thickness=minimum_thickness,
            )
            forecasts.append(forecast)

        forecast_df = pd.DataFrame(forecasts)

        if "id_number" in df.columns:
            result_df = df.merge(
                forecast_df, on="id_number", suffixes=("", "_forecast")
            )
        else:
            result_df = pd.concat([df, forecast_df], axis=1)

        return result_df

    def generate_forecast_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for forecasts."""
        forecast_df = self.forecast_batch(df)

        summary = {
            "total_cmls": len(forecast_df),
            "risk_distribution": forecast_df["risk_level"].value_counts().to_dict(),
            "avg_remaining_life_years": float(
                forecast_df["remaining_life_years"].mean()
            ),
            "min_remaining_life_years": float(
                forecast_df["remaining_life_years"].min()
            ),
            "critical_cmls": len(forecast_df[forecast_df["risk_level"] == "CRITICAL"]),
            "high_risk_cmls": len(forecast_df[forecast_df["risk_level"] == "HIGH"]),
            "inspections_needed_next_12_months": len(
                forecast_df[
                    pd.to_datetime(forecast_df["next_inspection_date"], errors="coerce")
                    < datetime.now() + timedelta(days=365)
                ]
            ),
            "top_priority_inspections": forecast_df.nsmallest(
                10, "remaining_life_years"
            )[
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
