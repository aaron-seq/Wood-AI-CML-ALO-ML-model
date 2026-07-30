# Architecture

## System overview

Two processes: a FastAPI service that owns the model and all business logic,
and a Streamlit dashboard that talks to it over HTTP. The dashboard holds no
scoring logic of its own, so a CLI, a notebook or another front end gets the
same behaviour from the same endpoints.

```mermaid
graph TB
    subgraph Clients
        BROWSER[Browser]
        CURL[curl / scripts / other services]
    end

    subgraph Dashboard["Streamlit dashboard :8501"]
        UI[streamlit_app.py]
        CLIENT[api_client.py<br/>CML_API_URL]
    end

    subgraph API["FastAPI service :8000"]
        MW[Middleware<br/>CORS - request id]
        ROUTES[app/main.py<br/>routes]
        ING[app/ingestion.py<br/>size - format - parse]
        FEAT[app/features.py<br/>engineered features]
        UTIL[app/utils.py<br/>validation - reports]
        FC[app/forecasting.py<br/>remaining life]
        SME[app/sme_override.py<br/>override store]
        MODEL[(sklearn Pipeline<br/>loaded at startup)]
    end

    subgraph Storage["Filesystem"]
        JOBLIB[models/*.joblib]
        JSON[data/sme_overrides.json]
    end

    subgraph Training["Offline"]
        TRAIN[ml/train_enhanced.py]
        DATA[data/*.csv]
    end

    BROWSER --> UI --> CLIENT -->|HTTP| MW
    CURL -->|HTTP| MW
    MW --> ROUTES
    ROUTES --> ING
    ROUTES --> FEAT
    ROUTES --> UTIL
    ROUTES --> FC
    ROUTES --> SME
    ROUTES --> MODEL
    MODEL -.loads.-> JOBLIB
    SME <--> JSON
    DATA --> TRAIN --> JOBLIB
    TRAIN --> FEAT

    style FEAT fill:#00AEEF,color:#fff
    style MODEL fill:#2e7d32,color:#fff
```

`app/features.py` is deliberately shared by the API and the training script.
It is the one place the remaining-life and corrosion-ratio formulas are
defined, so training and serving cannot compute them differently — the
condition that previously caused train/serve skew (see
[ADR-0003](adr/0003-share-feature-engineering-between-training-and-serving.md)).

## Module responsibilities

| Module | Owns | Depends on |
| --- | --- | --- |
| `app/main.py` | Routes, HTTP status mapping, model lifecycle | everything below |
| `app/config.py` | Settings from env/`.env`, path resolution | — |
| `app/ingestion.py` | Upload size/format/parse; `UploadError` | pandas |
| `app/features.py` | Engineered feature columns | pandas, numpy |
| `app/utils.py` | Dataframe validation, inspection schedule, reports | pandas |
| `app/forecasting.py` | `CMLForecaster`: life, intervals, risk levels | `app.features` |
| `app/sme_override.py` | Override persistence and statistics | pandas |
| `app/advanced_analytics.py` | Plotly figures and dataset statistics | `app.features`, plotly |
| `app/schemas.py` | Request/response contracts | pydantic |

Dependencies point one way — routes depend on services, never the reverse —
so any module can be imported and tested without starting the API.

## Request lifecycle: scoring

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant M as Middleware
    participant R as /score-cml-data
    participant I as ingestion
    participant V as utils.validate
    participant F as features
    participant P as sklearn Pipeline

    C->>M: POST multipart file
    M->>M: assign x-request-id
    M->>R: forward
    R->>R: model loaded?
    alt no model
        R-->>C: 503 model not loaded
    end
    R->>I: read_upload(file, max_bytes, max_rows)
    alt too large / wrong format / unparseable
        I-->>R: raise UploadError
        R-->>C: 400 + detail + request_id
    end
    I-->>R: DataFrame
    R->>V: validate_cml_dataframe
    alt required columns missing
        V-->>R: valid=false, errors
        R-->>C: 400 + which columns
    end
    R->>F: engineer_features
    F-->>R: + ratio, remaining life, inspection age, risk score
    R->>P: predict + predict_proba
    alt inference raises
        P-->>R: exception
        R-->>C: 500 (logged with traceback)
    end
    P-->>R: flags + probabilities
    R->>R: zip positionally, truncate to 100
    R-->>C: 200 results, total_results, results_truncated
```

The distinction that matters: anything the caller can fix is a `400` carrying
the reason; only a genuine server fault is a `500`. Previously every rejected
upload surfaced as a `500` because `HTTPException` was raised inside a `try`
whose own `except Exception` re-wrapped it.

## Data flow

```mermaid
flowchart LR
    CSV[CSV / XLSX upload] --> PARSE[parse + size check]
    PARSE --> VAL{required<br/>columns?}
    VAL -->|no| REJ[400 with the missing names]
    VAL -->|yes| ENG[engineer features]

    ENG --> RATIO[corrosion_thickness_ratio<br/>guarded division]
    ENG --> LIFE[remaining_life_years<br/>supplied value wins]
    ENG --> AGE[days_since_inspection<br/>365 if unknown]
    ENG --> RISK[risk_score<br/>derived if absent]

    RATIO --> X[feature matrix<br/>ordered by model.feature_names_in_]
    LIFE --> X
    AGE --> X
    RISK --> X

    X --> PRED[predict / predict_proba]
    PRED --> OUT[flag - probability - recommendation - confidence]
    OUT --> SMEO{SME override<br/>on this CML?}
    SMEO -->|yes| FINAL[expert decision wins]
    SMEO -->|no| FINAL2[model recommendation stands]
```

Two columns behave the same way on purpose: `risk_score` and
`remaining_life_years` are used as supplied and only derived when absent,
because the training pipeline consumes both straight from the source file.

## Model pipeline

```mermaid
flowchart LR
    subgraph Numeric["StandardScaler"]
        N1[average_corrosion_rate]
        N2[thickness_mm]
        N3[corrosion_thickness_ratio]
        N4[days_since_inspection]
        N5[risk_score]
        N6[remaining_life_years]
    end
    subgraph Categorical["OneHotEncoder — handle_unknown=ignore"]
        C1[commodity]
        C2[feature_type]
        C3[cml_shape]
    end
    Numeric --> CT[ColumnTransformer] --> RF[RandomForestClassifier<br/>300 trees - depth 10<br/>balanced_subsample] --> P[elimination probability]
    Categorical --> CT
```

`handle_unknown="ignore"` means a commodity the model never saw does not
raise — it contributes nothing to the prediction. The serving code reads the
column list from `model.feature_names_in_` rather than a hardcoded list, so
retraining with a different feature set cannot silently desynchronise the API.

## Deployment topology

```mermaid
graph LR
    subgraph Host["Docker Compose"]
        subgraph AC["wood-cml-api"]
            A[uvicorn :8000<br/>non-root uid 10001<br/>healthcheck /health]
        end
        subgraph DC["wood-cml-dashboard"]
            D[streamlit :8501<br/>CML_API_URL=http://api:8000]
        end
        V[(cml-data volume<br/>SME overrides)]
    end
    U[User] -->|8501| D
    U -->|8000| A
    D -->|compose network| A
    A --- V
```

The dashboard waits on `service_healthy`, so it never starts against an API
that has not loaded its model. The source tree is baked into the image rather
than bind-mounted: what runs is what was built.

## Design decisions

Recorded as ADRs in [`adr/`](adr/):

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-single-canonical-api-entry-point.md) | One API module, not four parallel copies |
| [0002](adr/0002-pin-dependencies-to-the-model-artifact.md) | Pin dependencies to the model artifact's scikit-learn |
| [0003](adr/0003-share-feature-engineering-between-training-and-serving.md) | Share feature engineering across training and serving |
| [0004](adr/0004-centralise-upload-validation.md) | Centralise upload validation; client errors are 400s |

## Conventions

- Type hints on public functions; `from __future__ import annotations` in new
  modules.
- Google-style docstrings stating what a function returns and what it raises.
- Comments explain *why*. A comment restating the code is noise; one recording
  a constraint that is not visible locally is the point.
- Never swallow an exception silently. Either handle it and log something
  actionable, or let it propagate.
- Configuration comes from `app.config.settings`, never from a literal at a
  call site.
- Business logic lives in `app/`, never in `streamlit_app.py`.
