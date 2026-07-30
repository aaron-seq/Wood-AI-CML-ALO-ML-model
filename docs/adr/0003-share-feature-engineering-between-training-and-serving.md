# ADR-0003: Share feature engineering between training and serving

**Status:** Accepted · **Date:** 2026-07-30

## Context

The remaining-life and corrosion-ratio formulas were implemented four times:

| Location | Behaviour |
| --- | --- |
| `app/main.py` | Recomputed `remaining_life_years`, clipped at 0, no upper bound, unguarded division |
| `ml/train_enhanced.py` | Used `remaining_life_years` **from the source file**; never recomputed it |
| `feature_engineering.py` (repo root) | A third copy, imported by nothing |
| `app/advanced_analytics.py` | Inlined five times, each `(t − 3.0) / rate.clip(lower=0.01)` |

Two defects followed directly from the duplication.

**Train/serve skew.** Training consumed `remaining_life_years` from the CSV.
The dataset generator caps that column at 300 years; the older 200-row dataset
caps it at 50. Serving discarded the supplied value and substituted an
uncapped recomputation — on `data/cml_sample_500.csv` reaching 415 where the
file said 300. The model was scoring a feature on a different scale from the
one it was fitted on.

**Division by zero.** The serving formula divided by `average_corrosion_rate`
with no guard, producing `inf` for a non-corroding CML. scikit-learn rejects
the matrix with "Input X contains infinity", so one benign row returned a 500
for the entire file. The analytics copies avoided this with
`.clip(lower=0.01)`, silently changing the value instead — a third behaviour.

## Decision

`app/features.py` is the single definition, imported by the API, the trainer
and the analytics module.

Two rules encoded there:

1. **A supplied value wins.** `remaining_life_years` and `risk_score` are used
   as provided and derived only when absent — matching what training consumes.
   The docstring says why, so the next person does not "fix" it back.
2. **Guarded division, no silent substitution.** A non-positive corrosion rate
   yields `MAX_REMAINING_LIFE_YEARS` (50.0), not `inf` and not a fabricated
   0.01 rate. Values above the ceiling are *not* capped, because training used
   the uncapped column.

## Alternatives considered

**Recompute on both sides instead.** Also removes the skew, and is arguably
cleaner — serving would not depend on a caller-supplied derived column. But it
changes the model's inputs, so it requires retraining and revalidation. Honour
the supplied value now; revisit when the model is next retrained, at which
point recomputing everywhere is the better end state.

**Cap remaining life at 50 years everywhere** for consistency with
`CMLForecaster`. Rejected: it would change predictions for every CML with a low
corrosion rate. The forecaster's cap is a presentation choice; the model's
feature must match its training distribution.

**A scikit-learn custom transformer inside the pipeline.** The textbook answer
— feature engineering travels with the artifact and skew becomes impossible.
Rejected for this pass because it changes the artifact format, so it cannot be
done without retraining. It is the natural companion to the retrain above.

## Consequences

Training and serving cannot diverge, both divisions are safe, and the formula
has one place and one set of tests. Verified: the skew fix changes **zero**
predictions on the bundled 500-row dataset, so it is safe to deploy against the
existing artifact.

The residual sharp edge: a caller who supplies `remaining_life_years` computed
with their own formula gets it fed to the model as-is. Documented in
`docs/TROUBLESHOOTING.md` under "Predictions look wrong", with the remedy —
omit the column and have it derived.

`ml/` now imports from `app/`, which couples the trainer to the application
package. Acceptable: they ship in one repository and must agree by
construction.
