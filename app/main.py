"""FastAPI application for CML elimination scoring and inspection planning.

This is the single canonical API entry point. Start it with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.features import engineer_features
from app.forecasting import CMLForecaster
from app.ingestion import UploadError, read_upload
from app.schemas import (
    ForecastOutput,
    HealthResponse,
    ModelInfoResponse,
    ReportResponse,
    ScoreResponse,
    SMEOverride,
    SMEOverrideListResponse,
    SMEOverrideResponse,
    UploadResponse,
)
from app.sme_override import SMEOverrideManager
from app.utils import (
    calculate_inspection_schedule,
    generate_elimination_report,
    validate_cml_dataframe,
)

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fallback for a model whose pickle predates ``feature_names_in_``. The
# live column list is read off the fitted estimator so it cannot drift.
_FALLBACK_FEATURE_COLUMNS = [
    "average_corrosion_rate",
    "thickness_mm",
    "corrosion_thickness_ratio",
    "days_since_inspection",
    "risk_score",
    "remaining_life_years",
    "commodity",
    "feature_type",
    "cml_shape",
]

HIGH_CONFIDENCE_MARGIN = 0.3

#: Populated during startup; ``None`` means the API runs in degraded mode
#: where non-ML endpoints still work but scoring returns 503.
model: Any | None = None

sme_manager = SMEOverrideManager(settings.SME_OVERRIDE_FILE)
forecaster = CMLForecaster(
    minimum_thickness=settings.DEFAULT_MINIMUM_THICKNESS,
    safety_factor=settings.SAFETY_FACTOR,
)


def _load_model() -> Any | None:
    """Load the model artifact, returning ``None`` if it is unusable.

    A missing or corrupt artifact must not stop the process: the health
    endpoint has to stay reachable so an orchestrator can report *why* the
    container is unhealthy instead of crash-looping with no diagnostics.
    """
    path = settings.MODEL_PATH
    if not path.exists():
        logger.warning(
            "Model artifact not found at %s. Scoring endpoints will return 503. "
            "Train one with: python ml/train_enhanced.py data/cml_sample_500.csv",
            path,
        )
        return None
    try:
        loaded = joblib.load(path)
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None
    logger.info("Model loaded from %s (%s)", path, type(loaded).__name__)
    return loaded


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once per process, at startup rather than at import."""
    global model
    model = _load_model()
    yield
    model = None


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Tag each request so a log line can be traced back to a client report."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(UploadError)
async def upload_error_handler(request: Request, exc: UploadError) -> JSONResponse:
    """Report upload problems as 400s, with the request id for correlation."""
    request_id = getattr(request.state, "request_id", None)
    logger.warning("Rejected upload (request_id=%s): %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "request_id": request_id},
    )


def _require_model() -> Any:
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ML model is not loaded. Train or mount a model artifact and restart.",
        )
    return model


def _override_summary(override: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a stored override to the fields a scoring response needs."""
    if override is None:
        return None
    return {
        "sme_decision": override["sme_decision"],
        "sme_name": override.get("sme_name"),
        "reason": override.get("reason"),
        "override_date": override.get("override_date"),
    }


def _json_safe_preview(df: pd.DataFrame, rows: int = 5) -> list[dict[str, Any]]:
    """First few rows as JSON-serialisable records.

    NaN is not valid JSON, so missing cells are emitted as null rather
    than as a float the client cannot parse.
    """
    preview = df.head(rows).astype(object)
    return [
        {
            str(column): None
            if value is None or (isinstance(value, float) and pd.isna(value))
            else value
            for column, value in record.items()
        }
        for record in preview.to_dict("records")
    ]


def _feature_columns(loaded_model: Any) -> list[str]:
    names = getattr(loaded_model, "feature_names_in_", None)
    return list(names) if names is not None else list(_FALLBACK_FEATURE_COLUMNS)


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    """Service banner with a pointer to the interactive documentation."""
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> dict[str, Any]:
    """Liveness and readiness probe used by Docker, Compose and the dashboard."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(settings.MODEL_PATH) if settings.MODEL_PATH.exists() else None,
        "version": settings.API_VERSION,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info() -> dict[str, Any]:
    """Describe the loaded estimator and the features it consumes."""
    loaded = _require_model()
    return {
        "model_type": type(loaded).__name__,
        "features_used": _feature_columns(loaded),
        "model_path": str(settings.MODEL_PATH),
    }


@app.post("/upload-cml-data", response_model=UploadResponse, tags=["data"])
async def upload_cml_data(file: UploadFile = File(...)) -> dict[str, Any]:
    """Parse and validate a CML file without scoring it.

    Returns the detected schema, a short preview and a validation report so
    a user can correct their data before spending time on a scoring run.
    """
    df = await read_upload(file, settings.MAX_UPLOAD_BYTES, settings.MAX_UPLOAD_ROWS)
    validation = validate_cml_dataframe(df)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": _json_safe_preview(df),
        "message": f"Successfully parsed {len(df)} records",
        "validation": validation,
    }


@app.post("/score-cml-data", response_model=ScoreResponse, tags=["scoring"])
async def score_cml_data(file: UploadFile = File(...)) -> dict[str, Any]:
    """Score every CML in the uploaded file for elimination.

    Rows are scored as one batch. The response echoes at most
    ``MAX_RESULTS_IN_RESPONSE`` results; ``total_results`` always reports
    the true count.
    """
    loaded = _require_model()
    df = await read_upload(file, settings.MAX_UPLOAD_BYTES, settings.MAX_UPLOAD_ROWS)

    validation = validate_cml_dataframe(df)
    if not validation["valid"]:
        raise UploadError("; ".join(validation["errors"]))

    features = engineer_features(df, minimum_thickness_mm=settings.DEFAULT_MINIMUM_THICKNESS)

    columns = _feature_columns(loaded)
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise UploadError(f"Missing columns required by the model: {', '.join(missing)}")

    logger.info("Scoring %d CMLs from %s", len(features), file.filename)
    try:
        X = features[columns]
        predictions = loaded.predict(X)
        probabilities = loaded.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.exception("Model inference failed for %s", file.filename)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Model inference failed: {exc}"
        ) from exc

    # Positional access throughout: ``iterrows`` yields index *labels*,
    # which are only interchangeable with positions for a default
    # RangeIndex, and predictions are positional numpy arrays.
    ids = features["id_number"].astype(str).tolist()

    # Expert decisions supersede the model. Until now the override store
    # was written but never read back into a scoring run, so an engineer
    # could record "KEEP this CML" and the next run still said ELIMINATE.
    overrides = sme_manager.get_override_map()

    results = []
    for position in range(len(features)):
        model_recommendation = "ELIMINATE" if predictions[position] == 1 else "KEEP"
        override = overrides.get(ids[position])
        results.append(
            {
                "id_number": ids[position],
                "predicted_elimination_flag": int(predictions[position]),
                "elimination_probability": float(probabilities[position]),
                "model_recommendation": model_recommendation,
                # The safe default: a client that reads only
                # "recommendation" must not be shown a decision an
                # engineer has already overruled.
                "recommendation": override["sme_decision"] if override else model_recommendation,
                "confidence": "HIGH"
                if abs(probabilities[position] - 0.5) > HIGH_CONFIDENCE_MARGIN
                else "MODERATE",
                "sme_override": _override_summary(override),
            }
        )

    overridden = sum(1 for result in results if result["sme_override"] is not None)
    logger.info(
        "Scored %d CMLs from %s (%d overridden by an SME)",
        len(results),
        file.filename,
        overridden,
    )
    return {
        "rows_scored": len(results),
        "results": results[: settings.MAX_RESULTS_IN_RESPONSE],
        "total_results": len(results),
        "results_truncated": len(results) > settings.MAX_RESULTS_IN_RESPONSE,
        "sme_overrides_applied": overridden,
        "model_info": {
            "model_type": type(loaded).__name__,
            "features_used": columns,
        },
        "message": f"Successfully scored {len(results)} CML records",
    }


@app.post(
    "/forecast-remaining-life",
    response_model=list[ForecastOutput],
    tags=["forecasting"],
)
async def forecast_remaining_life(file: UploadFile = File(...)) -> list[ForecastOutput]:
    """Project remaining life and a next-inspection date for each CML."""
    df = await read_upload(file, settings.MAX_UPLOAD_BYTES, settings.MAX_UPLOAD_ROWS)

    validation = validate_cml_dataframe(df)
    if not validation["valid"]:
        raise UploadError("; ".join(validation["errors"]))

    logger.info("Forecasting %d CMLs from %s", len(df), file.filename)

    results: list[ForecastOutput] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            schedule = calculate_inspection_schedule(
                corrosion_rate=float(row["average_corrosion_rate"]),
                thickness=float(row["thickness_mm"]),
                min_thickness=settings.DEFAULT_MINIMUM_THICKNESS,
                safety_factor=settings.SAFETY_FACTOR,
            )
        except (ValueError, TypeError) as exc:
            # One unusable row must not fail the batch, but silently
            # dropping rows would misreport coverage -- count them.
            skipped += 1
            logger.warning("Skipping CML %s: %s", row.get("id_number", "<unknown>"), exc)
            continue

        results.append(
            ForecastOutput(
                id_number=str(row["id_number"]),
                remaining_life_years=schedule["remaining_life_years"],
                next_inspection_date=schedule["next_inspection_date"],
                estimated_thickness_at_next_inspection=schedule[
                    "estimated_thickness_at_next_inspection"
                ],
                recommended_inspection_frequency_months=schedule["inspection_interval_months"],
                risk_level=schedule["risk_level"],
            )
        )

    if skipped:
        logger.warning("Forecast for %s skipped %d unusable row(s)", file.filename, skipped)
    logger.info("Generated %d forecasts from %s", len(results), file.filename)
    return results


@app.post("/generate-report", response_model=ReportResponse, tags=["reporting"])
async def generate_report(file: UploadFile = File(...)) -> dict[str, Any]:
    """Score a file and summarise the results as an elimination report.

    Aggregates predictions by commodity and feature type and surfaces the
    strongest elimination candidates alongside the marginal cases that
    warrant an engineer's review.
    """
    loaded = _require_model()
    df = await read_upload(file, settings.MAX_UPLOAD_BYTES, settings.MAX_UPLOAD_ROWS)

    validation = validate_cml_dataframe(df)
    if not validation["valid"]:
        raise UploadError("; ".join(validation["errors"]))

    features = engineer_features(df, minimum_thickness_mm=settings.DEFAULT_MINIMUM_THICKNESS)
    columns = _feature_columns(loaded)
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise UploadError(f"Missing columns required by the model: {', '.join(missing)}")

    try:
        X = features[columns]
        features["predicted_elimination"] = loaded.predict(X)
        features["elimination_probability"] = loaded.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.exception("Model inference failed while generating a report")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Model inference failed: {exc}"
        ) from exc

    features["recommendation"] = features["predicted_elimination"].map({1: "ELIMINATE", 0: "KEEP"})

    # Fold in expert decisions so the totals describe what will actually
    # be acted on, not what the model would have said on its own.
    features = sme_manager.apply_overrides_to_predictions(features)
    if "final_decision" in features.columns:
        overruled = features["final_decision"].notna()
        features.loc[overruled, "predicted_elimination"] = features.loc[
            overruled, "final_decision"
        ].map({"ELIMINATE": 1, "KEEP": 0})

    features["confidence_level"] = (
        (features["elimination_probability"] - 0.5).abs() > HIGH_CONFIDENCE_MARGIN
    ).map({True: "HIGH", False: "MODERATE"})

    report = generate_elimination_report(features)
    report["generated_from"] = file.filename
    return report


@app.get("/sme-override", response_model=SMEOverrideListResponse, tags=["sme"])
async def list_sme_overrides() -> dict[str, Any]:
    """List every recorded expert override plus agreement statistics."""
    return {
        "overrides": sme_manager.get_all_overrides(),
        "statistics": sme_manager.get_override_statistics(),
    }


@app.post(
    "/sme-override",
    response_model=SMEOverrideResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["sme"],
)
async def create_sme_override(override: SMEOverride) -> dict[str, Any]:
    """Record an expert decision that supersedes the model's recommendation.

    Overrides are keyed by CML id: posting the same id again replaces the
    previous decision rather than accumulating duplicates.
    """
    try:
        stored = sme_manager.add_override(
            id_number=override.id_number,
            sme_decision=override.sme_decision,
            reason=override.reason,
            sme_name=override.sme_name,
            original_prediction=override.original_prediction,
            original_probability=override.original_probability,
        )
    except OSError as exc:
        logger.exception("Failed to persist SME override for %s", override.id_number)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not persist the override: {exc}"
        ) from exc

    logger.info("Recorded SME override for %s by %s", override.id_number, override.sme_name)
    return {"override": stored, "message": f"Override recorded for {override.id_number}"}


@app.delete("/sme-override/{id_number}", tags=["sme"])
async def delete_sme_override(id_number: str) -> dict[str, str]:
    """Withdraw a previously recorded override."""
    if not sme_manager.remove_override(id_number):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No override recorded for CML {id_number}")
    return {"message": f"Override removed for {id_number}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
