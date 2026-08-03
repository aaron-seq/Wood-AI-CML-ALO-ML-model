# Contributing

## Setup

```bash
make setup   # .venv + all dependencies
make check   # lint, types, audit, tests — exactly what CI runs
```

`make help` lists every target. Use them rather than raw `pytest`/`ruff`
invocations so local runs and CI cannot diverge.

## Workflow

1. Branch from `main`.
2. Write the failing test first. Every bug fix needs a regression test; every
   feature needs tests for its happy path and its failure modes.
3. Make it pass.
4. `make check` must be green before you push.
5. Open a PR describing what changed and why.

Coverage must not go down; the build fails below 90%. It currently sits at
91% of `app/`, `ml/` and `api_client.py`.

Test order is randomised on every run (`pytest-randomly`); the seed is
printed, and `-p no:randomly` disables it. A test that only passes in one
order is a test with a hidden dependency.

The Streamlit dashboard is tested with `streamlit.testing.v1.AppTest`
(`tests/test_dashboard.py`), which runs the script in-process. Coverage
cannot attribute those lines because AppTest executes rather than imports
the script, so `streamlit_app.py` is absent from the coverage source --
tested, but not counted.

Numeric code should carry a property test as well as examples. The
invariants in `tests/test_properties.py` found three overflow bugs that
example-based tests had missed — state what must be true for *every*
input, not just the inputs you thought of.

## Standards

Ruff enforces formatting and lint rules (`pyproject.toml`); `make format`
applies them. Beyond that:

- **Type-hint public functions.** Add `from __future__ import annotations` to
  new modules.
- **Docstrings state what a function returns and what it raises** — Google
  style. Skip them on obvious private helpers.
- **Comments explain *why*.** A comment restating the code is noise. A comment
  recording a constraint that is not visible locally — why a value is not
  capped, why an order matters — is the point. If you fix a subtle bug, say
  what the old behaviour was, so nobody restores it.
- **Never swallow an exception.** Handle it and log something actionable, or
  let it propagate. `except Exception: pass` will be sent back.
- **Never raise `HTTPException` inside a `try` that catches broad exceptions.**
  This exact bug turned every client error into a 500
  ([ADR-0004](docs/adr/0004-centralise-upload-validation.md)). Raise a domain
  error and let the boundary map it.
- **Configuration comes from `app.config.settings`**, never a literal at a call
  site.
- **Business logic goes in `app/`**, never in `streamlit_app.py`. The dashboard
  is a view.
- **Feature engineering goes in `app/features.py`**, which the API, the trainer
  and the analytics module all share. Do not add a local copy of a formula
  ([ADR-0003](docs/adr/0003-share-feature-engineering-between-training-and-serving.md)).
- **Risk levels and inspection intervals come from `app/risk.py`.** Three
  copies of that logic drifted into disagreeing on 70% of CMLs
  ([ADR-0005](docs/adr/0005-single-risk-classifier.md)).
- **Type annotations are checked.** `make typecheck` must pass; new modules
  should be fully annotated.

## Before adding code

Ask whether the change can be made by deleting something, or by reusing what
exists, or without a new dependency. This codebase carried four copies of its
API and twelve dead scripts; the cost was invisible until documented endpoints
turned out not to exist.

New dependencies need a reason in the PR description.

## Changes needing extra care

Say so explicitly in the PR, and include a rollback plan:

| Area | Why |
| --- | --- |
| Model artifact or feature set | Changes predictions. Include before/after metrics and retrain against `data/cml_sample_500.csv` |
| `scikit-learn` pin | Requires retraining — the artifact is a pickle. Follow [DEPLOYMENT.md](docs/DEPLOYMENT.md#upgrading-scikit-learn) and commit requirements, artifact and metadata together |
| Upload limits | These bound worker memory |
| CORS, `app/security.py` or anything auth-adjacent | Security surface. Auth must stay disabled by default |
| Calibration or the probability contract | See [ADR-0007](docs/adr/0007-calibration-measured-and-declined.md) before re-enabling |
| SME override storage format | Existing audit trails must stay readable, and writes must stay atomic |
| Risk thresholds in `app/risk.py` | Changes what gets inspected and when |
| CI or Dockerfile | Affects everything downstream |

## Architecture decisions

Record a decision as an ADR in [`docs/adr/`](docs/adr/) when it would otherwise
be re-litigated or silently reversed — not for routine choices. State what was
rejected and what the decision costs, not just what was picked.

## Documentation

Docs are part of the change, not a follow-up. If you add an endpoint, update
[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) in the same PR.

Documentation that describes behaviour the code does not have is worse than no
documentation. This repository advertised `/sme-override` and
`/generate-report` for months before either existed, and published model
metrics derived from a four-positive test split. If a number is not
reproducible from the repository, do not publish it — and if it is weak, say
so.
