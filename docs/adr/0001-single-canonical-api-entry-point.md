# ADR-0001: One canonical API entry point

**Status:** Accepted · **Date:** 2026-07-30

## Context

`app/` contained four FastAPI applications: `main.py` (the one actually
served), `main_old.py`, `main_backup.py` and `main_enhanced.py`. They had
diverged in ways that mattered:

- `main_enhanced.py` was the only one that imported `app.config.Settings`, and
  the only one with SME override and report endpoints. The served `main.py`
  used hardcoded paths and lacked both.
- The README and `docs/API_DOCUMENTATION.md` documented `/sme-override` and
  `/generate-report`, which existed only in the unserved copy. Anyone
  following the docs got a 404.
- `main_enhanced.py` used the deprecated `@app.on_event("startup")` hook;
  `main.py` loaded the model at import time.

Keeping a "backup" copy in the repository is what git is for, and the cost was
concrete: the feature work in the enhanced copy never reached users, while the
documentation described it as if it had.

## Decision

`app/main.py` is the only application module. The endpoints that existed only
in `main_enhanced.py` were reimplemented there, against the current
`Settings`, and the other three files were deleted.

The model loads in an `asynccontextmanager` lifespan handler rather than at
import time or via `on_event`, so importing the module for a test does not
touch the filesystem, and startup failures surface as startup failures.

## Alternatives considered

**Keep `main_enhanced.py` and serve it instead.** Rejected: it also carried
drift the served module did not have (a `HealthResponse` returning a
`timestamp` field the schema does not declare), so it was not simply the better
copy. Merging forward was the smaller correct change.

**Split routes into an `api/routers/` package.** Rejected as premature. Nine
endpoints in ~430 well-sectioned lines is navigable; a package would add
indirection without removing anything. Worth revisiting past roughly twice
this size.

## Consequences

Anything documented is now served, and there is one place to change a route.
Deleting `main_enhanced.py` loses the four unused derived features its trainer
computed — deliberately, since wiring them in would change the model's inputs.

`app/main.py` will grow, and there is no longer a copy of the old behaviour
sitting in the tree to diff against. `git log` covers that.
