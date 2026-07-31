"""Pydantic schemas for API data validation and response models.

This module defines all request and response schemas used throughout
the Wood AI CML Optimization API, ensuring type safety and validation.
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SMEOverride(BaseModel):
    """Subject Matter Expert manual override record.

    Captures expert decisions that override ML model predictions,
    including detailed reasoning and metadata.
    """

    id_number: str = Field(..., description="CML identifier being overridden")
    sme_decision: str = Field(
        ..., pattern="^(KEEP|ELIMINATE)$", description="Expert decision: KEEP or ELIMINATE"
    )
    reason: str = Field(
        ..., min_length=10, description="Detailed explanation for override decision"
    )
    sme_name: str = Field(..., description="Name of subject matter expert")
    override_date: datetime | None = Field(
        default_factory=datetime.now, description="Timestamp of override decision"
    )
    original_prediction: str | None = Field(None, description="Original ML model prediction")
    original_probability: float | None = Field(
        None, ge=0.0, le=1.0, description="Original prediction probability"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id_number": "CML-042",
                "sme_decision": "KEEP",
                "reason": "Critical monitoring point for high-risk process area despite low corrosion rate",
                "sme_name": "Dr. John Smith",
                "original_prediction": "ELIMINATE",
                "original_probability": 0.85,
            }
        }
    )


class ForecastOutput(BaseModel):
    """Output from remaining life forecast calculation."""

    id_number: str = Field(..., description="CML identifier")
    remaining_life_years: float = Field(
        ..., ge=0.0, description="Estimated remaining life in years"
    )
    next_inspection_date: date = Field(..., description="Recommended date for next inspection")
    estimated_thickness_at_next_inspection: float = Field(
        ..., description="Projected wall thickness at next inspection (mm)"
    )
    recommended_inspection_frequency_months: int = Field(
        ..., ge=1, description="Recommended months between inspections"
    )
    risk_level: str = Field(..., description="Risk classification: LOW, MEDIUM, HIGH, or CRITICAL")


class HealthResponse(BaseModel):
    """API health check response."""

    # Several fields below start with "model_", which Pydantic reserves.
    # Renaming them would break every existing client, so the namespace is
    # opened up instead.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="API operational status: ok or degraded")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_path: str | None = Field(None, description="Path to model file")
    version: str = Field(..., description="API version")
    auth: str = Field(
        "disabled",
        description="Which endpoints require X-API-Key: 'disabled', 'writes' or 'all'",
    )


class UploadResponse(BaseModel):
    """File upload confirmation response."""

    filename: str = Field(..., description="Name of uploaded file")
    rows: int = Field(..., ge=0, description="Number of data rows")
    columns: list[str] = Field(..., description="Column names in dataset")
    preview: list[dict[str, Any]] = Field(..., description="Preview of first few rows")
    message: str | None = Field(None, description="Success message")
    validation: dict[str, Any] | None = Field(None, description="Optional validation results")


class ModelInfo(BaseModel):
    """Information about the ML model used for predictions."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="Type of ML model")
    features_used: list[str] = Field(..., description="List of features used for prediction")
    version: str | None = Field(None, description="Model version")
    trained_date: datetime | None = Field(None, description="When the model was trained")


class AppliedOverride(BaseModel):
    """The expert decision that superseded the model for one CML."""

    sme_decision: str = Field(..., description="The expert's decision: KEEP or ELIMINATE")
    sme_name: str | None = Field(None, description="Who recorded it")
    reason: str | None = Field(None, description="Why the model was overruled")
    override_date: str | None = Field(None, description="When it was recorded")


class PredictionResult(BaseModel):
    """Single prediction result for score endpoint."""

    id_number: str = Field(..., description="CML identifier")
    predicted_elimination_flag: int = Field(
        ..., ge=0, le=1, description="Raw model output: 0=Keep, 1=Eliminate"
    )
    elimination_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Uncalibrated probability of elimination"
    )
    model_recommendation: str = Field(
        ..., description="What the model alone recommended: KEEP or ELIMINATE"
    )
    recommendation: str = Field(
        ...,
        description=(
            "The decision to act on: the SME override when one exists, "
            "otherwise model_recommendation"
        ),
    )
    confidence: str = Field(..., description="Confidence level of the model prediction")
    sme_override: AppliedOverride | None = Field(
        None, description="Present only when an expert has overruled the model for this CML"
    )


class ScoreResponse(BaseModel):
    """Response from the CML scoring endpoint.

    Holds predictions for the uploaded data along with the model metadata
    and a processing summary.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "rows_scored": 200,
                "results": [
                    {
                        "id_number": "CML-001",
                        "predicted_elimination_flag": 0,
                        "elimination_probability": 0.23,
                        "recommendation": "KEEP",
                        "confidence": "HIGH",
                    }
                ],
                "total_results": 200,
                "results_truncated": True,
                "model_info": {
                    "model_type": "Pipeline",
                    "features_used": [
                        "average_corrosion_rate",
                        "thickness_mm",
                        "corrosion_thickness_ratio",
                        "days_since_inspection",
                        "risk_score",
                        "remaining_life_years",
                        "commodity",
                        "feature_type",
                        "cml_shape",
                    ],
                },
                "message": "Successfully scored 200 CML records",
            }
        },
    )

    rows_scored: int = Field(..., ge=0, description="Total number of rows processed")
    results: list[PredictionResult] = Field(
        ..., description="Prediction results, truncated to MAX_RESULTS_IN_RESPONSE"
    )
    total_results: int = Field(
        ..., ge=0, description="Total predictions generated, before truncation"
    )
    results_truncated: bool = Field(
        False, description="True when 'results' holds fewer entries than 'total_results'"
    )
    offset: int = Field(0, ge=0, description="Index of the first result in this page")
    limit: int = Field(0, ge=0, description="Maximum results this page could hold")
    sme_overrides_applied: int = Field(
        0, ge=0, description="How many scored CMLs carried an expert override"
    )
    model_info: ModelInfo = Field(..., description="Model metadata")
    message: str | None = Field(None, description="Processing message")


class ModelInfoResponse(BaseModel):
    """Metadata about the estimator currently serving predictions."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="Class name of the loaded estimator")
    features_used: list[str] = Field(
        ..., description="Feature columns the estimator consumes, in fitted order"
    )
    model_path: str = Field(..., description="Filesystem path of the loaded artifact")


class ReportResponse(BaseModel):
    """Aggregated elimination analysis for an uploaded dataset."""

    summary: dict[str, Any] = Field(..., description="Totals and the overall elimination rate")
    confidence_distribution: dict[str, Any] | None = Field(
        None, description="Count of predictions per confidence level"
    )
    elimination_by_commodity: dict[str, Any] | None = Field(
        None, description="Elimination counts grouped by commodity"
    )
    elimination_by_feature: dict[str, Any] | None = Field(
        None, description="Elimination counts grouped by feature type"
    )
    top_elimination_candidates: list[dict[str, Any]] | None = Field(
        None, description="Highest-probability elimination candidates"
    )
    marginal_cases: list[dict[str, Any]] | None = Field(
        None, description="Predictions near the decision boundary, for engineer review"
    )
    generated_from: str | None = Field(
        None, description="Name of the file the report was generated from"
    )


class SMEOverrideResponse(BaseModel):
    """Confirmation returned after recording an expert override."""

    override: dict[str, Any] = Field(..., description="The stored override record")
    message: str = Field(..., description="Human-readable confirmation")


class SMEOverrideListResponse(BaseModel):
    """All recorded overrides plus their agreement statistics."""

    overrides: list[dict[str, Any]] = Field(..., description="Stored override records")
    statistics: dict[str, Any] = Field(
        ..., description="Counts, per-SME distribution and ML agreement rate"
    )
