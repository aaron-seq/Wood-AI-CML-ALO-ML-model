# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`, OpenAPI schema
at `/openapi.json`.

There is **no authentication**. Anything that can reach the port can score
data and write SME overrides — deploy behind an authenticating gateway.

## Conventions

Every response carries an `x-request-id` header. Supply your own to correlate a
client-side report with server logs; otherwise one is generated.

| Status | Meaning |
| --- | --- |
| `200` / `201` | Success |
| `400` | The request is wrong: bad format, too large, missing columns |
| `404` | No such SME override |
| `422` | JSON body failed schema validation (FastAPI's own shape) |
| `500` | Server fault — model inference or persistence failed |
| `503` | No model artifact is loaded |

Errors return `{"detail": "...", "request_id": "..."}`. A `400` always names
what to fix.

Upload endpoints accept `.csv`, `.xlsx` and `.xls` as `multipart/form-data`
under the field name `file`. Defaults: 25 MB and 100,000 rows.

---

## `GET /`

Service banner.

```json
{ "message": "Wood AI CML Optimization API", "version": "1.0.0", "documentation": "/docs" }
```

---

## `GET /health`

Liveness and readiness. Used by the Docker healthcheck and the dashboard.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/models/cml_elimination_model.joblib",
  "version": "1.0.0"
}
```

`status` is `"degraded"` with `model_loaded: false` when no artifact loaded.
The process stays up deliberately: an orchestrator can then report *why* the
container is unhealthy instead of crash-looping with no diagnostics.
Non-ML endpoints keep working in that state.

---

## `GET /model/info`

Describes the loaded estimator. Returns `503` if none is loaded.

```json
{
  "model_type": "Pipeline",
  "features_used": [
    "average_corrosion_rate", "thickness_mm", "corrosion_thickness_ratio",
    "days_since_inspection", "risk_score", "remaining_life_years",
    "commodity", "feature_type", "cml_shape"
  ],
  "model_path": "/app/models/cml_elimination_model.joblib"
}
```

`features_used` is read from the fitted estimator, in fitted order, so it
always reflects the model actually serving traffic.

---

## `POST /upload-cml-data`

Parse and validate without scoring — a dry run for checking a file's schema.

```bash
curl -X POST http://localhost:8000/upload-cml-data -F "file=@data/cml_sample_500.csv"
```

```json
{
  "filename": "cml_sample_500.csv",
  "rows": 500,
  "columns": ["id_number", "commodity", "..."],
  "preview": [{ "id_number": "CML-001", "thickness_mm": 11.8 }],
  "message": "Successfully parsed 500 records",
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": ["3 records with unusual thickness (expected 0.0-50.0 mm). Examples: [...]"],
    "stats": {
      "total_records": 500,
      "unique_cmls": 500,
      "avg_corrosion_rate": 0.121,
      "avg_thickness": 9.42,
      "commodity_distribution": { "Condensate": 66 }
    }
  }
}
```

This endpoint **reports** schema problems rather than rejecting them:
`validation.valid` is `false` with the reasons in `errors`, and the response is
still `200`. It returns `400` only when the file itself cannot be read.
Missing cells serialise as `null`.

---

## `POST /score-cml-data`

Score every CML for elimination. Requires a loaded model.

```bash
curl -X POST http://localhost:8000/score-cml-data -F "file=@data/cml_sample_500.csv"
```

```json
{
  "rows_scored": 500,
  "results": [
    {
      "id_number": "CML-001",
      "predicted_elimination_flag": 0,
      "elimination_probability": 0.23,
      "model_recommendation": "KEEP",
      "recommendation": "KEEP",
      "confidence": "HIGH",
      "sme_override": null
    }
  ],
  "total_results": 500,
  "results_truncated": true,
  "sme_overrides_applied": 0,
  "model_info": { "model_type": "Pipeline", "features_used": ["..."] },
  "message": "Successfully scored 500 CML records"
}
```

| Field | Notes |
| --- | --- |
| `predicted_elimination_flag` | Raw model output: `0` keep, `1` eliminate |
| `elimination_probability` | Random Forest vote share in `[0, 1]`. **Not calibrated** — do not read it as a true probability |
| `model_recommendation` | What the model alone said; always agrees with the flag |
| `recommendation` | **The decision to act on.** Equals `model_recommendation` unless an expert has overruled it |
| `sme_override` | `null`, or the expert decision with who recorded it, when and why |
| `sme_overrides_applied` | How many rows in the batch carried an override |
| `confidence` | `HIGH` when more than 0.3 from the 0.5 boundary, else `MODERATE`. A distance measure, not a validated interval |
| `results` | Capped at 100 entries |
| `total_results` | The true count, always |
| `results_truncated` | `true` when `results` is a partial view |

Rows whose corrosion rate is zero score normally. Errors: `400` (bad or
incomplete file), `503` (no model), `500` (inference failed).

---

## `POST /forecast-remaining-life`

Projects remaining life and a next-inspection date per CML using the API 570
formula `(t_actual − t_min) / corrosion_rate`. **No model required** — this is
deterministic arithmetic, so it works even when the API is degraded.

```json
[
  {
    "id_number": "CML-001",
    "remaining_life_years": 50.0,
    "next_inspection_date": "2032-07-28",
    "estimated_thickness_at_next_inspection": 9.08,
    "recommended_inspection_frequency_months": 72,
    "risk_level": "LOW"
  }
]
```

`risk_level` comes from `app/risk.py`, the single classifier shared with the
dashboard and the analytics charts. A CML takes a level if **any** of its
conditions holds:

| Level | Conditions |
| --- | --- |
| `CRITICAL` | remaining life < 1 year, **or** wall thinner than 5 mm |
| `HIGH` | remaining life < 3 years, **or** rate > 0.25 mm/yr |
| `MEDIUM` | remaining life < 7 years, **or** rate > 0.15 mm/yr |
| `LOW` | none of the above |

Thickness is part of the test on purpose: a wall near its minimum has
almost no material left however slowly it is corroding, and a
remaining-life-only rule scored exactly that case as `LOW`.

Intervals are clamped to 1–6 years after the safety factor, shortened for
CMLs corroding faster than 0.20 mm/yr and extended below 0.05 mm/yr. A row with unusable values is skipped and logged with
a count rather than failing the batch.

---

## `POST /generate-report`

Scores a file and returns an aggregated analysis. Requires a loaded model.

```json
{
  "summary": {
    "total_cmls": 500,
    "recommended_eliminations": 101,
    "recommended_keep": 399,
    "elimination_rate": 20.2
  },
  "confidence_distribution": { "HIGH": 402, "MODERATE": 98 },
  "elimination_by_commodity": { "Crude Oil": 18, "Natural Gas": 25 },
  "elimination_by_feature": { "Pipe": 44, "Elbow": 21 },
  "top_elimination_candidates": [
    { "id_number": "CML-233", "elimination_probability": 0.97, "thickness_mm": 12.4 }
  ],
  "marginal_cases": [
    { "id_number": "CML-118", "elimination_probability": 0.52, "recommendation": "ELIMINATE" }
  ],
  "generated_from": "cml_sample_500.csv"
}
```

`marginal_cases` — probability between 0.4 and 0.6 — is the list worth an
engineer's attention: the model is closest to indifferent there.

---

## SME overrides

Expert decisions that supersede the model, kept as an audit trail. Keyed by
CML id.

Overrides are applied automatically. Once an expert records a decision for
a CML, every later scoring run returns it as the `recommendation` while
still reporting the model's own view in `model_recommendation` — the
override does not erase the audit trail.

### `POST /sme-override` → `201`

```bash
curl -X POST http://localhost:8000/sme-override \
  -H "Content-Type: application/json" \
  -d '{
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "High-consequence area adjacent to a fired heater",
    "sme_name": "A. Engineer",
    "original_prediction": "ELIMINATE",
    "original_probability": 0.85
  }'
```

| Field | Required | Constraint |
| --- | --- | --- |
| `id_number` | yes | — |
| `sme_decision` | yes | `KEEP` or `ELIMINATE` |
| `reason` | yes | At least 10 characters |
| `sme_name` | yes | — |
| `original_prediction` | no | Feeds the agreement rate |
| `original_probability` | no | `[0, 1]` |

Posting the same `id_number` again **replaces** the previous decision. A
violated constraint returns `422`.

### `GET /sme-override`

```json
{
  "overrides": [
    {
      "id_number": "CML-042",
      "sme_decision": "KEEP",
      "reason": "High-consequence area adjacent to a fired heater",
      "sme_name": "A. Engineer",
      "override_date": "2026-07-30T10:15:00",
      "original_prediction": "ELIMINATE",
      "original_probability": 0.85
    }
  ],
  "statistics": {
    "total_overrides": 1,
    "keep_overrides": 1,
    "eliminate_overrides": 0,
    "sme_distribution": { "A. Engineer": 1 },
    "recent_overrides": [{ "id_number": "CML-042" }],
    "disagreements_with_ml": 1,
    "agreement_rate": 0.0
  }
}
```

`agreement_rate` is how often experts agreed with the model, over the
overrides that recorded an `original_prediction`. It is a measure of the
model's field credibility and is worth watching over time.

### `DELETE /sme-override/{id_number}`

`200` on removal, `404` if there was nothing to remove.

---

## Python usage

The modules are importable directly, without the HTTP layer:

```python
import pandas as pd
from app.features import engineer_features
from app.forecasting import CMLForecaster
from app.sme_override import SMEOverrideManager
from app.utils import validate_cml_dataframe

df = pd.read_csv("data/cml_sample_500.csv")

report = validate_cml_dataframe(df)
if not report["valid"]:
    raise SystemExit(report["errors"])

forecaster = CMLForecaster(minimum_thickness=3.0, safety_factor=1.5)
forecasts = forecaster.forecast_batch(df)
print(forecasts[["id_number", "remaining_life_years", "risk_level"]].head())
print(forecaster.generate_forecast_summary(df))

features = engineer_features(df)  # same code the API and trainer use

sme = SMEOverrideManager()
sme.add_override(
    id_number="CML-042",
    sme_decision="KEEP",
    reason="High-consequence area adjacent to a fired heater",
    sme_name="A. Engineer",
)
print(sme.get_override_statistics())
```

Where a source column collides with a forecast column, `forecast_batch` keeps
the forecast under the canonical name and preserves the input as
`<name>_input`.
