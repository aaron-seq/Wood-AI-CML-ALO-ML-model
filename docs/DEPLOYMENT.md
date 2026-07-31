# Deployment

## Docker Compose

```bash
docker compose up --build -d
docker compose ps          # both services should read healthy
docker compose logs -f api
docker compose down        # add -v to also drop SME overrides
```

Brings up `wood-cml-api` on 8000 and `wood-cml-dashboard` on 8501. The
dashboard waits for the API's healthcheck, so it never starts against an API
that has not loaded its model.

Properties worth knowing:

- The image runs as uid 10001, not root. CI asserts this.
- The build toolchain stays in the builder stage; the runtime image ships only
  the virtualenv.
- `HEALTHCHECK` polls `/health` every 30s after a 20s grace period.
- The source tree is baked in, not bind-mounted — what runs is what was built.
  Use `make dev` for live reload while developing.
- SME overrides live on the `cml-data` named volume and survive a rebuild.

Build only the API image:

```bash
docker build --build-arg TARGET=api -t wood-cml-api .
docker run -p 8000:8000 wood-cml-api
```

`TARGET=dashboard` installs the UI stack instead.

## Dependency and security checks

```bash
make audit      # pip-audit over everything installed; fails on any known CVE
make typecheck  # mypy over app/ and ml/
make check      # lint, types, audit and tests -- what CI runs
```

CI runs the audit as its own job, and Dependabot opens a grouped weekly PR.
scikit-learn is excluded from those updates on purpose: bumping it requires
retraining the committed pickle (see below).

## Configuration

Every setting has a working default; see [`.env.example`](../.env.example) for
the full list. What actually needs attention in production:

| Variable | Why |
| --- | --- |
| `CORS_ORIGINS` | Must name your real dashboard origin over HTTPS. `*` is rejected at startup |
| `ENVIRONMENT` | Tags logs |
| `LOG_LEVEL` | `INFO` is right for production; `DEBUG` is noisy |
| `MAX_UPLOAD_BYTES` | Uploads are parsed in memory — this is the per-request memory ceiling |
| `MAX_UPLOAD_ROWS` | Second bound, on row count |
| `MODEL_DIR` | Point at a mounted volume to swap models without rebuilding |
| `CML_API_URL` | Where the dashboard looks for the API |

## Production checklist

Before exposing this to users:

- [ ] **Put an authenticating gateway in front.** There is no authentication.
      Anything that can reach the port can score data and write SME overrides.
- [ ] **Terminate TLS** at the gateway or load balancer.
- [ ] **Set `CORS_ORIGINS`** to your dashboard's real origin.
- [ ] **Rate-limit** at the gateway. Scoring is synchronous and CPU-bound; an
      unthrottled client can saturate the worker.
- [ ] **Size memory** against `MAX_UPLOAD_BYTES` × workers, with headroom for
      the parsed DataFrame (several times the file size).
- [ ] **Persist `data/`** on a real volume. Losing it loses the SME audit trail.
- [ ] **Run one replica, or move overrides to a database first.** Writes are
      atomic and guarded by an advisory lock, which makes concurrent writers
      on a *single host* safe. The lock does not coordinate across hosts or
      over a network filesystem.
- [ ] **Wire `/health` to your orchestrator's liveness and readiness probes.**
      Treat `status: degraded` as not-ready.
- [ ] **Ship logs somewhere queryable.** Correlate client reports by the
      `x-request-id` header, which appears on every response.
- [ ] **Validate the model against real inspection outcomes.** Both bundled
      datasets are synthetic — see [MODEL_CARD.md](MODEL_CARD.md).

Multiple workers are fine for scoring (the model is read-only and loaded per
process) but multiply the memory footprint:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Upgrading scikit-learn

The served model is a pickle. Loading it under a different scikit-learn minor
version raises `InconsistentVersionWarning`, and predictions are **not**
guaranteed to match. `requirements.txt` therefore pins `scikit-learn>=1.7.2,<1.8`,
and `pyproject.toml` promotes that warning to an error under pytest, so an
accidental bump fails the suite rather than shipping.

To upgrade deliberately:

1. Raise the pin in `requirements.txt`.
2. `make setup` to rebuild the environment.
3. Retrain: `make train DATASET=data/cml_sample_500.csv`. This writes a new
   timestamped artifact plus `cml_elimination_model.joblib` and a metadata JSON.
4. `make test` — the version check and the full suite must pass.
5. Compare the new metadata's metrics against the previous file in `models/`.
   Investigate a material drop before committing.
6. Commit the requirements change, the artifact and the metadata **together**.
   They are one unit; splitting them produces a repository that cannot load its
   own model.

CI verifies the committed artifact unpickles warning-free on every run.

## Rollback

The model and the code roll back independently.

**Model only** — previous artifacts are kept in `models/` with timestamps:

```bash
cp models/cml_elimination_model_20251207_115308.joblib \
   models/cml_elimination_model.joblib
docker compose restart api
curl -s localhost:8000/model/info
```

No rebuild needed if `models/` is a mounted volume.

**Code** — revert and redeploy:

```bash
git revert <sha>          # or: git checkout <previous-tag>
docker compose up --build -d
```

Nothing in this change set migrates data. SME overrides stay readable by every
version: the JSON shape is unchanged, so a rollback does not need a data step.

The one behavioural difference to expect on rollback: reverting past
`03f7706` reinstates the defects it fixed — client errors reported as `500`,
zero-corrosion-rate rows failing whole batches, and `get_override_statistics()`
raising `TypeError` once any override exists.

## Deploying elsewhere

The image is a plain uvicorn service with a healthcheck and no ambient state
beyond `models/` and `data/`, so it drops into Kubernetes, ECS or App Service
without changes. Two requirements carry over: mount `data/` on a persistent
volume, and keep the override store to a single writer until it moves to a
database.

## Future work

Ordered by what most limits production readiness:

1. **Validate against real inspection outcomes.** Everything else is secondary
   to knowing whether the model works on field data.
2. **Authentication and authorisation**, so overrides are attributable to an
   identity rather than a free-text `sme_name`.
3. **Move overrides to a database** with row-level locking; removes the
   single-host constraint that the advisory lock leaves in place.
4. **Calibrate probabilities** (`CalibratedClassifierCV`) so
   `elimination_probability` can be read as a probability and the confidence
   label means something checkable.
5. **Stream or queue large uploads** instead of parsing them in-request.
6. **Metrics and tracing** — Prometheus counters and OpenTelemetry spans.
   Request ids are already threaded through, which is the prerequisite.
7. **Per-circuit minimum thickness.** `app/risk.py` uses one global 3.0 mm
   floor; real programmes derive it per circuit from design pressure and
   material, which would also let the risk thresholds be calibrated rather
   than inherited.
