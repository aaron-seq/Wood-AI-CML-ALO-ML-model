# Troubleshooting

Every response carries an `x-request-id`. Quote it when reporting a problem —
it appears in the server logs for that request.

---

## `503 ML model is not loaded`

The API started but found no usable artifact. Check what it looked for:

```bash
curl -s localhost:8000/health
```

`model_path: null` means the file is absent. Train one:

```bash
make train DATASET=data/cml_sample_500.csv
```

If the path exists but the model still did not load, the startup log has the
traceback (`Failed to load model from ...`). Usually a truncated file — Git LFS
not pulled, or an incomplete copy — or a pickle from an incompatible
scikit-learn. Verify:

```bash
.venv/bin/python -c "import joblib; print(joblib.load('models/cml_elimination_model.joblib'))"
```

Forecasting keeps working without a model; only scoring and reporting need one.

---

## `400 Unsupported file format`

Only `.csv`, `.xlsx` and `.xls` are accepted, matched on the filename
extension. A CSV named `export` with no extension is rejected — rename it.

## `400 Missing required columns: ...`

The message names exactly which are absent. Six are mandatory:
`id_number`, `average_corrosion_rate`, `thickness_mm`, `commodity`,
`feature_type`, `cml_shape`. Names are case-sensitive; Excel exports often
carry trailing spaces in headers.

Use `/upload-cml-data` to inspect a file without scoring it — it returns the
columns it found plus a full validation report.

## `400 File is too large` / `exceeds the ... row limit`

Defaults are 25 MB and 100,000 rows. Split the file, or raise
`MAX_UPLOAD_BYTES` / `MAX_UPLOAD_ROWS` — but note uploads are parsed in memory,
so raising the limit raises the worker's memory ceiling. Size the container to
match.

## `400 The file could not be parsed`

Malformed CSV. Common causes: an Excel file renamed to `.csv`, inconsistent
column counts between rows, or a non-UTF-8 encoding. Re-export as UTF-8 CSV or
as real `.xlsx`.

---

## Dashboard says the API is offline

The dashboard reports the URL it tried. Check the API directly:

```bash
curl -s localhost:8000/health
```

If the API is up but the dashboard cannot see it, the address is wrong. It
reads `CML_API_URL`:

```bash
CML_API_URL=http://localhost:8000 make dashboard
```

Under Compose this must be `http://api:8000` — the service name, not
`localhost`, which inside the dashboard container refers to that container.

## Dashboard loads but charts are missing

The Reports page needs `plotly`. `make setup` installs it;
`pip install -r requirements.txt` alone does not — that set is deliberately
API-only. Use `requirements-streamlit.txt` for the dashboard.

## Results look truncated

By default `/score-cml-data` returns the first 100 rows; `total_results` is
the true count and `results_truncated` flags the partial view. Every row is
always scored — the cap applies to the response body only.

Page through the rest rather than re-uploading:

```bash
curl -X POST "localhost:8000/score-cml-data?offset=100&limit=100" \
  -F "file=@data/cml_sample_500.csv"
```

Or raise `MAX_RESULTS_IN_RESPONSE` to change the default page size.

---

## `401 A valid X-API-Key header is required`

Authentication is enabled on the API. Send the key:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/sme-override
```

Check which endpoints are gated — `/health` reports the posture and is never
gated itself:

```bash
curl -s localhost:8000/health   # {"auth": "disabled" | "writes" | "all"}
```

For the dashboard, set `CML_API_KEY` to match the API's `API_KEY`.

Missing and wrong keys return identical responses on purpose, so the body
will not tell you which it was — the server log will, keyed by request id.

## The API refuses to start: "API_KEY must be at least 16 characters"

Deliberate. A key short enough to guess implies protection that is not
there. Generate one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Or unset `API_KEY` entirely to run without authentication.

## `/metrics` returns 404

Metrics are off by default so an unconfigured deployment does not publish
its traffic volume. Set `METRICS_ENABLED=true` and scrape from inside the
network rather than through a public ingress.

---

## Browser console: CORS error

The calling origin is not in `CORS_ORIGINS`. Add it (comma-separated):

```bash
CORS_ORIGINS=http://localhost:8501,https://cml.example.com
```

`*` is rejected at startup on purpose — list origins explicitly.

---

## `InconsistentVersionWarning` on startup

The installed scikit-learn differs from the one that pickled the model.
Predictions are not guaranteed to match. Reinstall against the pins:

```bash
make setup
```

If you intended to upgrade, retrain — see
[DEPLOYMENT.md](DEPLOYMENT.md#upgrading-scikit-learn). Under pytest this
warning is an error, so a bad bump fails the suite.

---

## Tests fail after a dependency change

```bash
make setup && make test
```

A failure in `tests/test_regressions.py` means a previously fixed production
defect is back; the test's docstring says which one. Do not relax the
assertion — fix the cause.

A failure in `test_statistics_on_the_repository_override_file` means
`data/sme_overrides.json` is malformed.

## Port already in use

```bash
make api API_PORT=8001
make dashboard DASHBOARD_PORT=8502
```

For Compose, change the host side of the port mapping in
`docker-compose.yml`. Do not change the container side — the healthcheck
targets 8000.

---

## Container reports unhealthy

```bash
docker compose logs api
docker inspect -f '{{json .State.Health}}' wood-cml-api | python -m json.tool
```

The healthcheck curls `/health` with a 20s grace period. If the API answers but
reports `status: degraded`, the container is running and the model is missing —
see the 503 section above.

## Container fails to start after mounting a volume over `/app`

Don't. The image contains the application; mounting over it hides the code.
Mount only `data/` (and `models/` if you swap artifacts without rebuilding).

---

## Overrides disappear after a rebuild

They live on the `cml-data` volume. `docker compose down -v` deletes it — use
`docker compose down` alone. Back up before a destructive operation:

```bash
docker compose cp api:/app/data/sme_overrides.json ./sme_overrides.backup.json
```

## Two people's overrides conflict

Concurrent writes on one host are safe: each read-modify-write cycle takes an
advisory lock, and the file is replaced atomically, so nothing interleaves and
no decision is lost. Posting the same CML id twice is a deliberate replace —
the newer decision wins, and the older one is gone.

The lock does **not** coordinate across hosts or over a network filesystem.
Run a single API replica until the store moves to a database
([DEPLOYMENT.md](DEPLOYMENT.md#future-work)).

---

## Predictions look wrong

Check what the model is actually being given:

```bash
curl -s localhost:8000/model/info | python -m json.tool
```

Two things routinely surprise people:

- **`elimination_probability` is not calibrated.** It is a Random Forest vote
  share. 0.9 does not mean "90% likely correct."
- **`risk_score` and `remaining_life_years` are used as supplied** when your
  file provides them, and only derived when absent — because training consumed
  them from the source file. If your `remaining_life_years` was computed with a
  different formula or cap than the training data's, the model sees a
  differently-scaled feature. Drop the column to have it derived consistently.

## A CML is flagged CRITICAL that used to be LOW

Risk classification now includes wall thickness, not just remaining life.
Anything thinner than 5 mm is CRITICAL regardless of how slowly it is
corroding, because a wall near its minimum has almost no material left. The
API and the dashboard previously disagreed here on 70% of CMLs; both now use
`app/risk.py`. See [ADR-0005](adr/0005-single-risk-classifier.md).

## The recommendation does not match the model prediction

That is an override doing its job. Check `sme_override` on the result:

```bash
curl -s http://localhost:8000/sme-override | python -m json.tool
```

`recommendation` is the decision to act on; `model_recommendation` is what
the model said on its own. Withdraw the override with
`DELETE /sme-override/{id}` to fall back to the model.

Before this was wired up, overrides were recorded but never applied — if
you are comparing against older output, that is the difference.

---

An unknown commodity or feature type does not error: `handle_unknown="ignore"`
means it contributes nothing to the prediction. Check
`validation.stats.commodity_distribution` from `/upload-cml-data` for
unexpected values or typos.
