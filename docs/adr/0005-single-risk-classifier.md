# ADR-0005: One risk classifier, and it is the conservative one

**Status:** Accepted · **Date:** 2026-07-31

## Context

Three implementations decided how risky a CML was:

| Where | Rule |
| --- | --- |
| `app.utils.calculate_inspection_schedule` (`/forecast-remaining-life`) | remaining life only: CRITICAL < 2y, HIGH < 5y, MEDIUM < 10y |
| `CMLForecaster` (dashboard Forecasting page) | remaining life **or** thickness **or** corrosion rate |
| `app.advanced_analytics` (charts and statistics) | `pd.cut` bins matching the first rule |

Measured over a grid of 1,700 CMLs (thickness 3.5–20 mm × rate
0.01–0.50 mm/yr), the first two disagreed on **70%**. The API and the
dashboard reported different risk levels and different inspection
intervals for the same asset, and nothing in the test suite noticed.

The divergence was not symmetric. Across the whole grid the forecaster's
classification was **never less severe** than the utils one — 1,195
escalations, zero de-escalations.

## Decision

`app/risk.py` holds the only definition of both risk level and inspection
interval. All three call sites delegate to it.

The forecaster's rule wins. Two reasons, in order:

1. **It is right about the case that matters.** A 3.5 mm wall over a 3.0 mm
   minimum, corroding at 0.01 mm/yr, has 50 years of nominal remaining
   life and almost no material left. The remaining-life-only rule called
   that `LOW`. Thickness belongs in the test.
2. **It is the safe direction to be wrong in.** Where the two disagreed,
   adopting it can only raise a level, never lower one. For a tool that
   informs pressure-equipment inspection, a spurious escalation costs an
   inspection; a missed one does not.

## Alternatives considered

**Keep the utils rule, since it backs the public endpoint.** Rejected: it
is the less safe of the two, and "it was already published" is not a
reason to keep classifying thin walls as low risk.

**Make the thresholds configurable per deployment.** Real programmes do set
minimum thickness per circuit from design pressure and material, so this
is directionally right. Rejected for now because the immediate problem is
*disagreement*, not calibration, and a settings surface would let two
deployments diverge again. Worth revisiting once per-circuit minimum
thickness is supported.

**Merge the two into something stricter than either.** Rejected: no
evidence supports new thresholds, and inventing them would be a change
nobody could justify from data.

## Consequences

The API, the dashboard and the analytics charts cannot disagree; a test
replays the full 1,700-CML grid and asserts zero divergence in both level
and interval.

`/forecast-remaining-life` now returns different risk levels than before —
always equal or more severe. On the bundled 500-row dataset the
distribution is 446 LOW / 54 MEDIUM. Any client with a stored expectation
of the old levels will see a change, and any downstream count of "how many
CRITICAL CMLs" will move. That is the point.

The thresholds are still judgement calls inherited from the original code,
not values validated against field outcomes. They are now at least
*visible* judgement calls, in one file, with names.
