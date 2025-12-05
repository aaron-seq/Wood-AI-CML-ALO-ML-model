"""Pydantic schemas for API data validation and response models.

This module defines all request and response schemas used throughout
the Wood AI CML Optimization API, ensuring type safety and validation.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CMLShape(str, Enum):
    """CML monitoring shape/location types."""
    INTERNAL = "Internal"
    EXTERNAL = "External"
    BOTH = "Both"


class FeatureType(str, Enum):
    """Piping feature types for CML monitoring."""
    PIPE = "Pipe"
    ELBOW = "Elbow"
    TEE = "Tee"
    REDUCER = "Reducer"
    FLANGE = "Flange"
    NOZZLE = "Nozzle"
    HEADER = "Header"
    BEND = "Bend"
    WELD = "Weld"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class CMLDataInput(BaseModel):
    """Input schema for CML data validation.
    
    Represents a single Condition Monitoring Location with all
    required parameters for ML model prediction and analysis.
    """
    id_number: str = Field(..., description="Unique CML identifier")
    average_corrosion_rate: float = Field(
        ...,
        ge=0,
        le=5.0,
        description="Average corrosion rate in mm/year"
    )
    thickness_mm: float = Field(
        ...,
        gt=0,
        le=50.0,
        description="Current wall thickness in mm"
    )
    commodity: str = Field(..., min_length=1, description="Commodity type")
    feature_type: str = Field(..., description="Piping feature type")
    cml_shape: str = Field(..., description="CML monitoring location")
    last_inspection_date: Optional[str] = Field(
        None,
        description="Date of last inspection (YYYY-MM-DD)"
    )
    next_inspection_date: Optional[str] = Field(
        None,
        description="Scheduled next inspection date (YYYY-MM-DD)"
    )
    remaining_life_years: Optional[float] = Field(
        None,
        ge=0,
        description="Calculated remaining life in years"
    )
    risk_score: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Risk score (0-100)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id_number": "CML-001",
                "average_corrosion_rate": 0.12,
                "thickness_mm": 9.5,
                "commodity": "Crude Oil",
                "feature_type": "Pipe",
                "cml_shape": "Both",
                "last_inspection_date": "2023-06-15",
                "next_inspection_date": "2026-06-15",
                "risk_score": 25
            }
        }


class CMLPrediction(BaseModel):
    """Output schema for single CML elimination prediction.
    
    Contains ML model prediction results including elimination
    recommendation and confidence metrics.
    """
    id_number: str = Field(..., description="CML identifier")
    predicted_elimination: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary prediction: 0=Keep, 1=Eliminate"
    )
    elimination_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of elimination (0-1)"
    )
    keep_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of keeping (0-1)"
    )
    recommendation: str = Field(..., description="Human-readable: KEEP or ELIMINATE")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )
    confidence_level: str = Field(
        ...,
        description="Confidence category: LOW, MODERATE, or HIGH"
    )


class CMLBatchInput(BaseModel):
    """Batch input for multiple CML predictions."""
    cmls: List[CMLDataInput] = Field(
        ...,
        description="List of CML data records to process"
    )


class CMLBatchPrediction(BaseModel):
    """Batch prediction output with summary statistics."""
    predictions: List[CMLPrediction] = Field(
        ...,
        description="Individual predictions for each CML"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Aggregated statistics across all predictions"
    )


class SMEOverride(BaseModel):
    """Subject Matter Expert manual override record.
    
    Captures expert decisions that override ML model predictions,
    including detailed reasoning and metadata.
    """
    id_number: str = Field(..., description="CML identifier being overridden")
    sme_decision: str = Field(
        ...,
        pattern="^(KEEP|ELIMINATE)$",
        description="Expert decision: KEEP or ELIMINATE"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Detailed explanation for override decision"
    )
    sme_name: str = Field(..., description="Name of subject matter expert")
    override_date: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Timestamp of override decision"
    )
    original_prediction: Optional[str] = Field(
        None,
        description="Original ML model prediction"
    )
    original_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Original prediction probability"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id_number": "CML-042",
                "sme_decision": "KEEP",
                "reason": "Critical monitoring point for high-risk process area despite low corrosion rate",
                "sme_name": "Dr. John Smith",
                "original_prediction": "ELIMINATE",
                "original_probability": 0.85
            }
        }


class ForecastInput(BaseModel):
    """Input parameters for remaining life forecasting."""
    id_number: str = Field(..., description="CML identifier")
    average_corrosion_rate: float = Field(
        ...,
        ge=0.0,
        description="Corrosion rate in mm/year"
    )
    thickness_mm: float = Field(
        ...,
        gt=0.0,
        description="Current wall thickness in mm"
    )
    minimum_required_thickness: float = Field(
        default=3.0,
        gt=0.0,
        description="Minimum safe thickness in mm"
    )
    inspection_interval_months: Optional[int] = Field(
        default=36,
        ge=1,
        le=120,
        description="Preferred inspection interval in months"
    )


class ForecastOutput(BaseModel):
    """Output from remaining life forecast calculation."""
    id_number: str = Field(..., description="CML identifier")
    remaining_life_years: float = Field(
        ...,
        ge=0.0,
        description="Estimated remaining life in years"
    )
    next_inspection_date: date = Field(
        ...,
        description="Recommended date for next inspection"
    )
    estimated_thickness_at_next_inspection: float = Field(
        ...,
        description="Projected wall thickness at next inspection (mm)"
    )
    recommended_inspection_frequency_months: int = Field(
        ...,
        ge=1,
        description="Recommended months between inspections"
    )
    risk_level: str = Field(
        ...,
        description="Risk classification: LOW, MEDIUM, HIGH, or CRITICAL"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="API operational status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_path: Optional[str] = Field(None, description="Path to model file")
    version: str = Field(..., description="API version")


class UploadResponse(BaseModel):
    """File upload confirmation response."""
    filename: str = Field(..., description="Name of uploaded file")
    rows: int = Field(..., ge=0, description="Number of data rows")
    columns: List[str] = Field(..., description="Column names in dataset")
    preview: List[Dict[str, Any]] = Field(
        ...,
        description="Preview of first few rows"
    )
    message: Optional[str] = Field(None, description="Success message")
    validation: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional validation results"
    )


class ModelInfo(BaseModel):
    """Information about the ML model used for predictions."""
    model_type: str = Field(..., description="Type of ML model")
    features_used: List[str] = Field(
        ...,
        description="List of features used for prediction"
    )
    version: Optional[str] = Field(None, description="Model version")
    trained_date: Optional[datetime] = Field(
        None,
        description="When the model was trained"
    )


class PredictionResult(BaseModel):
    """Single prediction result for score endpoint."""
    id_number: str = Field(..., description="CML identifier")
    predicted_elimination_flag: int = Field(
        ...,
        ge=0,
        le=1,
        description="0=Keep, 1=Eliminate"
    )
    elimination_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of elimination"
    )
    recommendation: str = Field(..., description="KEEP or ELIMINATE")
    confidence: str = Field(..., description="Confidence level")


class ScoreResponse(BaseModel):
    """Response from CML data scoring endpoint.
    
    Contains predictions for uploaded CML data along with
    model information and processing summary.
    """
    rows_scored: int = Field(
        ...,
        ge=0,
        description="Total number of rows processed"
    )
    results: List[PredictionResult] = Field(
        ...,
        description="Prediction results (limited to first 100)"
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Total number of predictions generated"
    )
    model_info: ModelInfo = Field(..., description="Model metadata")
    message: Optional[str] = Field(None, description="Processing message")

    class Config:
        json_schema_extra = {
            "example": {
                "rows_scored": 200,
                "results": [
                    {
                        "id_number": "CML-001",
                        "predicted_elimination_flag": 0,
                        "elimination_probability": 0.23,
                        "recommendation": "KEEP",
                        "confidence": "HIGH"
                    }
                ],
                "total_results": 200,
                "model_info": {
                    "model_type": "RandomForestClassifier",
                    "features_used": [
                        "average_corrosion_rate",
                        "thickness_mm",
                        "commodity",
                        "feature_type",
                        "cml_shape"
                    ]
                },
                "message": "Successfully scored 200 CML records"
            }
        }
