"""Canonical feature engineering for CML records.

Every consumer of engineered features -- the scoring endpoint, the
training pipeline and the analytics charts -- goes through this module so
that a change to the remaining-life formula cannot silently apply to one
of them and not the others.

The two derived quantities both divide by a user-supplied measurement, so
both are computed with an explicit guard. An unguarded division produced
``inf`` for any CML with a zero corrosion rate, which scikit-learn rejects
with "Input X contains infinity", failing the *entire* batch because of a
single benign row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A CML that is not corroding has an unbounded remaining life. Reporting
# it as this finite ceiling keeps the value usable by the model and by
# arithmetic downstream, and matches the ceiling already applied by
# app.utils.calculate_inspection_schedule and app.forecasting.
MAX_REMAINING_LIFE_YEARS = 50.0

DEFAULT_MINIMUM_THICKNESS_MM = 3.0
DEFAULT_DAYS_SINCE_INSPECTION = 365

#: Columns a caller must supply before features can be engineered.
BASE_NUMERIC_COLUMNS = ("average_corrosion_rate", "thickness_mm")

#: Columns this module adds to a CML frame.
ENGINEERED_COLUMNS = (
    "corrosion_thickness_ratio",
    "remaining_life_years",
    "days_since_inspection",
    "risk_score",
)


def remaining_life_years(
    thickness_mm: pd.Series,
    corrosion_rate: pd.Series,
    minimum_thickness_mm: float = DEFAULT_MINIMUM_THICKNESS_MM,
) -> pd.Series:
    """Years until ``thickness_mm`` corrodes down to ``minimum_thickness_mm``.

    Implements the API 570 remaining-life formula
    ``(t_actual - t_minimum) / corrosion_rate`` with two guards:

    * a non-positive corrosion rate yields :data:`MAX_REMAINING_LIFE_YEARS`
      rather than infinity;
    * a wall already at or below the minimum yields ``0.0``.

    Values are otherwise left uncapped, because the model was trained on
    the uncapped column and capping here would introduce train/serve skew.
    """
    thickness = pd.to_numeric(thickness_mm, errors="coerce")
    rate = pd.to_numeric(corrosion_rate, errors="coerce")

    available = thickness - minimum_thickness_mm
    life = pd.Series(
        np.divide(
            available.to_numpy(dtype="float64"),
            rate.to_numpy(dtype="float64"),
            out=np.full(len(rate), MAX_REMAINING_LIFE_YEARS, dtype="float64"),
            where=rate.to_numpy(dtype="float64") > 0,
        ),
        index=thickness.index,
    )
    return life.clip(lower=0.0).fillna(0.0)


def corrosion_thickness_ratio(corrosion_rate: pd.Series, thickness_mm: pd.Series) -> pd.Series:
    """Corrosion rate per mm of remaining wall, ``0.0`` where thickness is unusable."""
    rate = pd.to_numeric(corrosion_rate, errors="coerce").to_numpy(dtype="float64")
    thickness = pd.to_numeric(thickness_mm, errors="coerce")
    ratio = np.divide(
        rate,
        thickness.to_numpy(dtype="float64"),
        out=np.zeros(len(rate), dtype="float64"),
        where=thickness.to_numpy(dtype="float64") > 0,
    )
    return pd.Series(ratio, index=thickness.index).fillna(0.0)


def days_since_inspection(
    last_inspection_date: pd.Series, now: pd.Timestamp | None = None
) -> pd.Series:
    """Age of the last inspection in days, defaulting to one year when unknown."""
    reference = now if now is not None else pd.Timestamp.now()
    parsed = pd.to_datetime(last_inspection_date, errors="coerce")
    age = (reference - parsed).dt.days
    return age.fillna(DEFAULT_DAYS_SINCE_INSPECTION).astype("float64")


def fallback_risk_score(corrosion_rate: pd.Series, thickness_mm: pd.Series) -> pd.Series:
    """Composite 0-100 risk score, used only when the source data omits one."""
    rate = pd.to_numeric(corrosion_rate, errors="coerce").fillna(0.0)
    thickness = pd.to_numeric(thickness_mm, errors="coerce").fillna(0.0)
    return (rate * 20 + (10 - thickness) * 5).clip(0, 100)


def engineer_features(
    df: pd.DataFrame,
    minimum_thickness_mm: float = DEFAULT_MINIMUM_THICKNESS_MM,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered model features added.

    Args:
        df: CML records containing at least the columns in
            :data:`BASE_NUMERIC_COLUMNS`.
        minimum_thickness_mm: Minimum allowable wall thickness.
        now: Reference time for inspection-age calculations; defaults to
            the current time. Injectable so tests are deterministic.

    Returns:
        A new DataFrame; the input is never mutated.

    Raises:
        KeyError: If a required base column is missing.
    """
    missing = [column for column in BASE_NUMERIC_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    out = df.copy()
    out["corrosion_thickness_ratio"] = corrosion_thickness_ratio(
        out["average_corrosion_rate"], out["thickness_mm"]
    )

    # A supplied remaining_life_years is authoritative. The training
    # pipeline consumes this column straight from the source file, so
    # recomputing it at inference -- as the scoring endpoint used to --
    # fed the model a differently-scaled value than it was fitted on. The
    # dataset generator caps the column at 300 years, for instance, while
    # the raw formula is unbounded.
    if "remaining_life_years" in out.columns:
        out["remaining_life_years"] = pd.to_numeric(
            out["remaining_life_years"], errors="coerce"
        ).fillna(
            remaining_life_years(
                out["thickness_mm"], out["average_corrosion_rate"], minimum_thickness_mm
            )
        )
    else:
        out["remaining_life_years"] = remaining_life_years(
            out["thickness_mm"], out["average_corrosion_rate"], minimum_thickness_mm
        )

    if "last_inspection_date" in out.columns:
        out["days_since_inspection"] = days_since_inspection(out["last_inspection_date"], now=now)
    else:
        out["days_since_inspection"] = float(DEFAULT_DAYS_SINCE_INSPECTION)

    if "risk_score" not in out.columns:
        out["risk_score"] = fallback_risk_score(out["average_corrosion_rate"], out["thickness_mm"])

    return out
