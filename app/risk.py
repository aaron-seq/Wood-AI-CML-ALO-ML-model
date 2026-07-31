"""Canonical risk classification and inspection scheduling.

Risk level and inspection interval were previously derived by three
independent implementations -- ``app.utils.calculate_inspection_schedule``
(behind the ``/forecast-remaining-life`` endpoint), ``CMLForecaster``
(behind the dashboard's Forecasting page), and ``pd.cut`` bins inside
``app.advanced_analytics``. Over a grid of 1,700 realistic CMLs the first
two disagreed on **70%** of them, so the API and the dashboard reported
different risk levels for the same asset.

This module is the single definition. It adopts the more conservative of
the two rules: across that grid the forecaster's classification was never
lower than the utils one, and it correctly escalates the case the
remaining-life-only rule gets dangerously wrong -- a wall already close to
its minimum thickness reads as LOW under a pure remaining-life test
whenever the corrosion rate is small, even though it has almost no
material left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Risk thresholds. A CML qualifies for a level if *any* of its conditions
# is met, evaluated most-severe first.
CRITICAL_LIFE_YEARS = 1.0
CRITICAL_THICKNESS_MM = 5.0

HIGH_LIFE_YEARS = 3.0
HIGH_CORROSION_RATE = 0.25

MEDIUM_LIFE_YEARS = 7.0
MEDIUM_CORROSION_RATE = 0.15

#: Ordered least to most severe, for sorting and comparison.
RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Inspection interval bounds, after the safety factor is applied.
MIN_INSPECTION_INTERVAL_MONTHS = 12
MAX_INSPECTION_INTERVAL_MONTHS = 72

# Fast-corroding assets get inspected sooner than the raw division
# suggests, slow-corroding ones later.
FAST_CORROSION_RATE = 0.20
FAST_CORROSION_INTERVAL_FACTOR = 0.7
SLOW_CORROSION_RATE = 0.05
SLOW_CORROSION_INTERVAL_FACTOR = 1.3


def classify(remaining_life_years: float, corrosion_rate: float, thickness_mm: float) -> str:
    """Classify one CML as LOW, MEDIUM, HIGH or CRITICAL.

    Args:
        remaining_life_years: Years until the minimum thickness is reached.
        corrosion_rate: Corrosion rate in mm/year.
        thickness_mm: Current wall thickness in mm.

    Returns:
        One of :data:`RISK_ORDER`.
    """
    if remaining_life_years < CRITICAL_LIFE_YEARS or thickness_mm < CRITICAL_THICKNESS_MM:
        return "CRITICAL"
    if remaining_life_years < HIGH_LIFE_YEARS or corrosion_rate > HIGH_CORROSION_RATE:
        return "HIGH"
    if remaining_life_years < MEDIUM_LIFE_YEARS or corrosion_rate > MEDIUM_CORROSION_RATE:
        return "MEDIUM"
    return "LOW"


def classify_series(
    remaining_life_years: pd.Series,
    corrosion_rate: pd.Series,
    thickness_mm: pd.Series,
) -> pd.Series:
    """Vectorised :func:`classify`, for charting a whole dataset.

    Returns an ordered categorical so plots and ``value_counts`` sort by
    severity rather than alphabetically.
    """
    life = pd.to_numeric(remaining_life_years, errors="coerce").fillna(0.0)
    rate = pd.to_numeric(corrosion_rate, errors="coerce").fillna(0.0)
    thickness = pd.to_numeric(thickness_mm, errors="coerce").fillna(0.0)

    # Evaluated least-severe first so that later conditions overwrite
    # earlier ones, leaving the most severe match in place.
    levels = np.full(len(life), "LOW", dtype=object)
    levels[(life < MEDIUM_LIFE_YEARS) | (rate > MEDIUM_CORROSION_RATE)] = "MEDIUM"
    levels[(life < HIGH_LIFE_YEARS) | (rate > HIGH_CORROSION_RATE)] = "HIGH"
    levels[(life < CRITICAL_LIFE_YEARS) | (thickness < CRITICAL_THICKNESS_MM)] = "CRITICAL"

    return pd.Series(
        pd.Categorical(levels, categories=list(RISK_ORDER), ordered=True),
        index=life.index,
    )


def inspection_interval_months(
    remaining_life_years: float,
    corrosion_rate: float,
    safety_factor: float = 1.5,
    min_months: int = MIN_INSPECTION_INTERVAL_MONTHS,
    max_months: int = MAX_INSPECTION_INTERVAL_MONTHS,
) -> int:
    """Recommended months between inspections.

    Divides remaining life by the safety factor, adjusts for how fast the
    asset is corroding, then clamps to ``[min_months, max_months]``.

    Args:
        remaining_life_years: Years until the minimum thickness is reached.
        corrosion_rate: Corrosion rate in mm/year.
        safety_factor: Divisor applied to remaining life; must be >= 1.0.
        min_months: Lower clamp. Configurable because ``CMLForecaster``
            exposes it as a constructor argument.
        max_months: Upper clamp.

    Returns:
        Interval in whole months.

    Raises:
        ValueError: If ``safety_factor`` is below 1.0, or the bounds are
            inverted.
    """
    if safety_factor < 1.0:
        raise ValueError(f"Invalid safety factor: {safety_factor}. Must be >= 1.0.")
    if min_months > max_months:
        raise ValueError(f"min_months ({min_months}) exceeds max_months ({max_months}).")

    months = int((remaining_life_years / safety_factor) * 12)

    if corrosion_rate > FAST_CORROSION_RATE:
        months = int(months * FAST_CORROSION_INTERVAL_FACTOR)
    elif corrosion_rate < SLOW_CORROSION_RATE:
        months = int(months * SLOW_CORROSION_INTERVAL_FACTOR)

    return max(min_months, min(months, max_months))
