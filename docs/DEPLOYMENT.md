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

## Measured throughput

From `make bench` on one core (medians of three runs, synthetic data
shaped like the real dataset):

| CMLs | parse | validate | features | inference | end-to-end | peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 2 ms | 3 ms | 4 ms | 25 ms | 35 ms | 0.2 MiB |
| 10,000 | 13 ms | 8 ms | 6 ms | 99 ms | 130 ms | 4 MiB |
| 100,000 | 136 ms | 109 ms | 32 ms | 799 ms | 1.1 s | 41 MiB |

A full 50,000-row scoring request measured **553 ms end to end** through
the API, including multipart handling and JSON serialisation.

Two things follow:

- **Inference dominates** — roughly three quarters of the time at every
  size. Parsing and feature engineering are not worth optimising; model
  size is the lever, and CPU is what to scale on.
- **The row limit binds before the byte limit.** 100,000 rows is about
  5.5 MB of CSV, well under the 25 MB default. Peak memory at that size is
  ~41 MiB per in-flight request, so `MAX_UPLOAD_BYTES` × workers
  overstates the real footprint by a wide margin. Size on measured peak
  plus headroom, not on the byte limit.

Re-run after any change to the model or the feature pipeline; the script
takes about a minute.

## Production checklist

Before exposing this to users:

- [ ] **Set `API_KEY`** (16+ characters, e.g.
      `python -c "import secrets; print(secrets.token_urlsafe(32))"`). It is
      unset by default, which leaves every endpoint open. Consider
      `API_KEY_SCOPE=all` if nothing internal needs unauthenticated scoring.
- [ ] **Still put an authenticating gateway in front.** The API key is a
      shared secret, not an identity: it cannot tell you *which* engineer
      recorded an override, and `sme_name` remains self-declared.
- [ ] **Terminate TLS** at the gateway or load balancer.
- [ ] **Set `CORS_ORIGINS`** to your dashboard's real origin.
- [ ] **Rate-limit** at the gateway. Scoring is synchronous and CPU-bound; an
      unthrottled client can saturate the worker.
- [ ] **Size memory** from the measured peak above (~41 MiB per in-flight
      100,000-row request) × workers, plus headroom. `MAX_UPLOAD_BYTES` is a
      ceiling on input, not a prediction of memory use.
- [ ] **Persist `data/`** on a real volume. Losing it loses the SME audit trail.
- [ ] **Run one replica, or move overrides to a database first.** Writes are
      atomic and guarded by an advisory lock, which makes concurrent writers
      on a *single host* safe. The lock does not coordinate across hosts or
      over a network filesystem.
- [ ] **Wire `/health` to your orchestrator's liveness and readiness probes.**
      Treat `status: degraded` as not-ready.
- [ ] **Set `LOG_FORMAT=json`** so logs carry request id, path, status and
      duration as fields rather than prose. Correlate client reports by the
      `x-request-id` header, which appears on every response.
- [ ] **Enable `METRICS_ENABLED`** and scrape `/metrics` from inside the
      network. Watch `cml_sme_overrides_applied_total` against
      `cml_scored_total`: a rising ratio means experts are increasingly
      overruling the model.
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
4. **Revisit calibration once the model is worth calibrating.**
   `CalibratedClassifierCV` is already wired in behind
   `make train CALIBRATE=sigmoid`, but on the current data it costs more F1
   than it recovers in calibration error (see MODEL_CARD.md).
5. **Stream or queue large uploads** instead of parsing them in-request.
6. **Distributed tracing** — OpenTelemetry spans. Prometheus counters and
   request ids are in place; spans across the dashboard/API boundary are not.
7. **Per-circuit minimum thickness.** `app/risk.py` uses one global 3.0 mm
   floor; real programmes derive it per circuit from design pressure and
   material, which would also let the risk thresholds be calibrated rather
   than inherited.
