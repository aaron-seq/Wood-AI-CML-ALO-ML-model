# ADR-0002: Pin dependencies to the model artifact's scikit-learn

**Status:** Accepted · **Date:** 2026-07-30

## Context

`requirements.txt` used open lower bounds only (`scikit-learn>=1.3.0`,
`pandas>=2.1.0`, `fastapi>=0.104.0`). A fresh install in July 2026 resolved to
scikit-learn 1.9.0 and pandas 3.0.5.

`models/cml_elimination_model.joblib` was pickled by scikit-learn **1.7.2**.
Loading it under 1.9.0 emitted five `InconsistentVersionWarning`s. scikit-learn
is explicit that cross-version unpickling is unsupported and may produce
different results — silently, since the model still loads and still predicts.

So the shipped state was: whether the API's predictions matched the ones its
metrics were measured on depended on the day you ran `pip install`. Nothing in
the test suite noticed; all 25 tests passed under both.

## Decision

Pin every runtime dependency with an upper bound, with `scikit-learn>=1.7.2,<1.8`
as the anchor — the version that produced the artifact.

Make the failure mode loud in three places:

1. `pyproject.toml` promotes `InconsistentVersionWarning` to an error under
   pytest, so the suite fails rather than passing with a suspect model.
2. CI unpickles the artifact under `warnings.simplefilter("error")` as its own
   step, before the tests run.
3. `requirements.txt` says in a comment that raising the scikit-learn bound
   requires retraining, and `docs/DEPLOYMENT.md` gives the procedure.

## Alternatives considered

**Retrain against the latest scikit-learn.** Rejected for now: it changes
predictions, and the point of this pass was to fix defects without moving the
model underneath the user. It is the right move once the model is validated
against real inspection data — at which point the retrain is deliberate rather
than incidental.

**Export to ONNX or PMML** to decouple serving from scikit-learn's pickle
format. This genuinely solves the problem and is the right long-term answer,
but it adds a conversion step and a runtime dependency for a project whose
model is not yet validated. Premature.

**Lockfile (`pip-compile`, uv, Poetry).** Stronger reproducibility, and
compatible with this decision. Deferred because it changes the contributor
workflow; the upper bounds capture the safety property that actually matters
here.

## Consequences

An install now reproduces the environment the model was trained and measured
in, and a dependency bump that invalidates the artifact fails CI instead of
shipping.

The cost is real: the project sits on scikit-learn 1.7.x and pandas 2.x rather
than the newest releases, and security updates to a pinned package require
bumping the bound deliberately. That is the intended trade — a known-good
model beats a newer library.

Dependency freshness should be reviewed on a schedule, not left to drift
indefinitely behind the pins.
