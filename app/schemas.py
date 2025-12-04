"""Pydantic schemas for data validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import date, datetime
from enum import Enum


class CMLShape(str, Enum):
    """CML monitoring shape types."""
    INTERNAL = "Internal"
    EXTERNAL = "External"
    BOTH = "Both"


class FeatureType(str, Enum):
    """Piping feature types."""
    PIPE = "Pipe"
    ELBOW = "Elbow"
    TEE = "Tee"
    REDUCER = "Reducer"
    FLANGE = "Flange"
    NOZZLE = "Nozzle"
    HEADER = "Header"
    BEND = "Bend"
    WELD = "Weld"


class CMLDataInput(BaseModel):
    """Input schema for CML data validation."""
    id_number: str = Field(..., description="Unique CML identifier")
    average_corrosion_rate: float = Field(
        ..., ge=0, le=5.0, description="Average corrosion rate in mm/year"
    )
    thickness_mm: float = Field(
        ..., gt=0, le=50.0, description="Current wall thickness in mm"
    )
    commodity: str = Field(..., min_length=1, description="Commodity type")
    feature_type: str = Field(..., description="Piping feature type")
    cml_shape: str = Field(..., description="CML monitoring location")
    last_inspection_date: Optional[str] = Field(None, description="Date of last inspection")
    next_inspection_date: Optional[str] = Field(None, description="Scheduled next inspection")
    remaining_life_years: Optional[float] = Field(None, ge=0, description="Calculated remaining life")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="Risk score (0-100)")

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
    """Output schema for CML elimination prediction."""
    id_number: str
    predicted_elimination: int = Field(..., ge=0, le=1, description="0=Keep, 1=Eliminate")
    elimination_probability: float = Field(..., ge=0, le=1)
    keep_probability: float = Field(..., ge=0, le=1)
    recommendation: str = Field(..., description="KEEP or ELIMINATE")
    confidence: float = Field(..., ge=0, le=1)
    confidence_level: str = Field(..., description="LOW, MEDIUM, or HIGH")


class CMLBatchInput(BaseModel):
    """Batch input for multiple CMLs."""
    cmls: List[CMLDataInput]


class CMLBatchPrediction(BaseModel):
    """Batch prediction output."""
    predictions: List[CMLPrediction]
    summary: dict


class SMEOverride(BaseModel):
    """SME manual override record."""
    id_number: str
    sme_decision: str = Field(..., pattern="^(KEEP|ELIMINATE)$")
    reason: str = Field(..., min_length=10)
    sme_name: str
    override_date: Optional[datetime] = Field(default_factory=datetime.now)
    original_prediction: Optional[str] = None
    original_probability: Optional[float] = None

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
    """Input for remaining life forecasting."""
    id_number: str
    average_corrosion_rate: float
    thickness_mm: float
    minimum_required_thickness: float = Field(default=3.0, description="Minimum safe thickness")
    inspection_interval_months: Optional[int] = Field(default=36, ge=1, le=120)


class ForecastOutput(BaseModel):
    """Output for remaining life forecast."""
    id_number: str
    remaining_life_years: float
    next_inspection_date: date
    estimated_thickness_at_next_inspection: float
    recommended_inspection_frequency_months: int
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class UploadResponse(BaseModel):
    """File upload response."""
    filename: str
    rows: int
    columns: List[str]
    preview: List[dict]
    validation: Optional[dict] = None