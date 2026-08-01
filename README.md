# Wood AI CML ALO ML Model

Machine-learning **Condition Monitoring Location (CML) optimisation** for
pipework inspection programmes. Given a set of CMLs with wall thickness and
corrosion-rate measurements, the system recommends which monitoring locations
can be retired, projects remaining life and inspection dates, and records the
expert decisions that override it.

It is a **decision support tool**. Recommendations are advisory; the SME
override system exists because an engineer, not the model, owns the call.

[![CI](https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Quick start

```bash
git clone https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model.git
cd Wood-AI-CML-ALO-ML-model

make setup   # create .venv and install everything
make test    # 233 tests, should pass in ~35s
make dev     # API on :8000, dashboard on :8501
```

Or with Docker:

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| API docs (Swagger) | http://localhost:8000/docs |
| Health probe | http://localhost:8000/health |
| Dashboard | http://localhost:8501 |

`make help` lists every command. Local development, CI and the container
images all run these same targets, so they cannot drift apart.

---

## Input data

Upload CSV or Excel. Six columns are required:

| Column | Type | Meaning |
| --- | --- | --- |
| `id_number` | string | Unique CML identifier |
| `average_corrosion_rate` | float | Corrosion rate, mm/year |
| `thickness_mm` | float | Current wall thickness, mm |
| `commodity` | string | Process commodity (Crude Oil, Natural Gas, Steam, …) |
| `feature_type` | string | Component (Pipe, Elbow, Tee, Flange, Weld, …) |
| `cml_shape` | string | Monitoring location (Internal, External, Both) |

These are optional but improve results, because the model was trained with
them:

| Column | Effect if omitted |
| --- | --- |
| `risk_score` | Derived from corrosion rate and thickness |
| `minimum_thickness_mm` | Falls back to the global 3.0 mm floor |
| `remaining_life_years` | Derived via the API 570 formula |
| `last_inspection_date` | Inspection age defaults to 365 days |

`data/cml_sample_500.csv` is a working example. Upload limits are 25 MB and
100,000 rows by default; both are configurable (see [`.env.example`](.env.example)).

---

## API

Full reference with request and response bodies:
[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and model-loaded status |
| `GET` | `/model/info` | Loaded estimator and its feature columns |
| `GET` | `/metrics` | Prometheus metrics (404 unless enabled) |
| `POST` | `/upload-cml-data` | Parse and validate a file without scoring it |
| `POST` | `/score-cml-data` | Score every CML, applying any expert overrides |
| `POST` | `/forecast-remaining-life` | Remaining life and next inspection date |
| `POST` | `/generate-report` | Aggregated elimination analysis |
| `GET` | `/sme-override` | List overrides and agreement statistics |
| `POST` | `/sme-override` | Record an expert override |
| `DELETE` | `/sme-override/{id}` | Withdraw an override |

```bash
curl -X POST http://localhost:8000/score-cml-data \
  -F "file=@data/cml_sample_500.csv"
```

`/score-cml-data` echoes at most 100 results in the body. `total_results`
always reports the true count and `results_truncated` says whether the list
is partial.

Recorded expert decisions supersede the model: `recommendation` is the
decision to act on, `model_recommendation` is what the model said on its
own, and `sme_override` names who overruled it and why.

Pass `?offset=&limit=` to page through a large batch — the default is the
first 100, as before.

### Authentication

Off by default. Set `API_KEY` (16+ characters) and callers must send
`X-API-Key`. `API_KEY_SCOPE=writes` (the default) gates only the endpoints
that mutate the SME audit trail; `all` gates scoring and reporting too. `/`
and `/health` are never gated, so orchestrator probes keep working.

This is a shared secret, not an identity: `sme_name` on an override is still
self-declared. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Dashboard

`make dashboard` (or the `dashboard` container) serves seven pages: Overview,
Upload & Analyze, Forecasting, SME Overrides, Reports, How It Works, and
About. Reports carries the API 570 analytics — risk matrix, remaining-life
distribution, inspection timeline, priority scatter and portfolio gauges.

The dashboard calls the API over HTTP and reads its address from
`CML_API_URL` (default `http://localhost:8000`).

---

## Model

A scikit-learn `Pipeline`: `StandardScaler` + `OneHotEncoder` feeding a
`RandomForestClassifier` (300 trees, `max_depth=10`,
`class_weight="balanced_subsample"`), selected by 5-fold grid search on F1.

Feature importances, read from the served artifact:

| Feature | Importance |
| --- | --- |
| `corrosion_thickness_ratio` | 37.2% |
| `average_corrosion_rate` | 29.2% |
| `risk_score` | 9.7% |
| `remaining_life_years` | 7.1% |
| `thickness_mm` | 6.5% |
| `days_since_inspection` | 3.0% |
| commodity / feature type / shape (one-hot) | 7.3% combined |

### Performance, and why the earlier numbers were misleading

The README previously advertised 90%+ accuracy and ROC-AUC 0.92+. Those came
from the served artifact's metadata, which was produced on a **40-row test
split containing 4 positive examples**. An AUC computed from four positives
is not a usable estimate of anything, and it should not have been published
as one.

Retraining the same pipeline on the larger 500-row dataset gives:

| Metric | 200-row set (as shipped) | 500-row set (measured) |
| --- | --- | --- |
| Test rows / positives | 40 / 4 | 100 / 39 |
| Accuracy | 0.95 | 0.68 |
| F1 (eliminate) | 0.75 | 0.53 |
| ROC-AUC | 0.99 | 0.69 |
| CV F1 (mean ± sd) | 0.87 ± 0.12 | 0.61 ± 0.05 |

**Treat ROC-AUC ≈ 0.69 as the honest current figure.** Both datasets are
synthetic, so neither predicts field performance; validation against real
inspection outcomes is the prerequisite for production use. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use and limitations.

Retrain with:

```bash
make train DATASET=data/cml_sample_500.csv
```

Raising the `scikit-learn` pin requires retraining — the artifact is a pickle
and cross-version predictions are not guaranteed. CI fails if the committed
model does not unpickle cleanly. See
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#upgrading-scikit-learn).

---

## Layout

```
app/                      FastAPI application
  main.py                 All routes; the single entry point
  config.py               Settings (env / .env driven)
  features.py             Shared feature engineering (API + training)
  risk.py                 Risk classification and inspection intervals
  security.py             Optional API-key gate
  observability.py        JSON logging and Prometheus metrics
  ingestion.py            Upload validation and parsing
  forecasting.py          Remaining-life and inspection scheduling
  sme_override.py         Expert override store
  advanced_analytics.py   Plotly charts and statistics
  utils.py                Validation and report generation
  schemas.py              Pydantic request/response models
ml/train_enhanced.py      Training pipeline with grid search
streamlit_app.py          Dashboard
api_client.py             Dashboard's HTTP client
tests/                    233 tests
docs/                     Architecture, API, deployment, model card, ADRs
```

---

## Development

```bash
make check      # exactly what CI runs: lint, types, audit, tests
make format     # apply formatting and safe fixes
make typecheck  # mypy over app/ and ml/
make audit      # fail on any known CVE in a shipped dependency
make bench      # scoring throughput and memory by stage
```

Coverage is enforced at a 90% floor. Property-based tests (Hypothesis)
cover the engineering calculations; they found three overflow bugs the
example-based tests missed. `pre-commit install` is optional and runs the
fast checks locally.

`make check` runs ruff (lint + format), mypy, pip-audit and the tests.
Configuration for all of them is in `pyproject.toml`.
See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Components, request lifecycle, data flow, diagrams |
| [API reference](docs/API_DOCUMENTATION.md) | Every endpoint, with examples |
| [Deployment](docs/DEPLOYMENT.md) | Docker, production checklist, upgrades, rollback |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Symptoms and fixes |
| [Model card](docs/MODEL_CARD.md) | Intended use, metrics, limitations |
| [ADRs](docs/adr/) | Why the significant decisions were made |
| [Contributing](CONTRIBUTING.md) | Workflow and standards |

---

## Known limitations

- **Both datasets are synthetic.** Reported metrics do not predict field
  performance.
- **Authentication is a shared secret at best.** The optional API key gates
  access but does not identify the caller, so overrides remain attributable
  only to a self-declared name. Deploy behind an authenticating gateway.
- **Overrides are stored in a JSON file**, written atomically and guarded by
  an advisory lock. That makes a single host safe; it does not coordinate
  across hosts, so run one replica until the store moves to a database.
- **Uploads are parsed entirely in memory.** The size limit is the memory
  bound; size container limits accordingly.
- **`elimination_probability` is a Random Forest vote share**, not a calibrated
  probability. Calibration is implemented (`make train CALIBRATE=sigmoid`) but
  off by default because it was measured and did not help — see the model card.
- **Single process, no queue.** A large file is scored synchronously inside the
  request. Measured at 553 ms for 50,000 CMLs, so this is a concurrency limit
  rather than a latency one — see [measured throughput](docs/DEPLOYMENT.md#measured-throughput).

Planned work is tracked in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#future-work).

---

## Credits

Original project by **Wood PLC** — Jeffrey Anokye (Project Owner), Jason
Strouse (Development Lead), Mariana Lima (Project Lead). Subsequent
development by Aaron Sequeira (Smarter.Codes.AI).

Licensed under the [MIT License](LICENSE).
