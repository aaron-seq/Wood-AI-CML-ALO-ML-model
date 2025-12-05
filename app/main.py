"""FastAPI application for Wood AI CML ALO ML Model.

This module provides the main REST API endpoints for CML optimization,
including health checks, data upload, scoring, and batch predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, UploadResponse, ScoreResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Wood AI CML ALO API",
    version="0.3.0",
    description="Machine Learning API for Condition Monitoring Location optimization",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cml_elimination_model.joblib"

# Required columns for CML data
REQUIRED_COLUMNS = [
    "id_number",
    "average_corrosion_rate",
    "thickness_mm",
    "commodity",
    "feature_type",
    "cml_shape",
    "last_inspection_date",
    "next_inspection_date",
]

# Feature columns used for model predictions
FEATURE_COLUMNS = [
    "average_corrosion_rate",
    "thickness_mm",
    "commodity",
    "feature_type",
    "cml_shape",
]

# File upload limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Model loading
model: Optional[Any] = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint providing API information.
    
    Returns:
        Dictionary with welcome message and documentation links
    """
    return {
        "message": "Wood AI CML Optimization API",
        "version": "0.3.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health() -> Dict[str, Any]:
    """Health check endpoint.
    
    Verifies API status and model availability.
    
    Returns:
        Dictionary containing:
        - status: API operational status
        - model_loaded: Whether ML model is loaded
        - model_path: Path to model file
        - version: API version
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "version": "0.3.0"
    }


def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}. "
                   f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def read_uploaded_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into pandas DataFrame.
    
    Args:
        file: Uploaded file (CSV or Excel)
        
    Returns:
        Parsed DataFrame
        
    Raises:
        HTTPException: If file cannot be parsed
    """
    try:
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file.file)
        elif file_ext in {'.xlsx', '.xls'}:
            df = pd.read_excel(file.file)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        if df.empty:
            raise ValueError("Uploaded file contains no data")
        
        logger.info(f"Successfully parsed {file.filename}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/upload-cml-data", tags=["Data"], response_model=UploadResponse)
async def upload_cml_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and validate CML data file.
    
    Accepts CSV or Excel files containing CML data and returns
    basic information about the uploaded dataset.
    
    Args:
        file: Uploaded CSV or Excel file
        
    Returns:
        Dictionary containing:
        - filename: Name of uploaded file
        - rows: Number of data rows
        - columns: List of column names
        - preview: First 5 rows as dictionary records
        
    Raises:
        HTTPException: If file is invalid or cannot be processed
    """
    validate_file_upload(file)
    df = read_uploaded_dataframe(file)
    
    # Generate preview (limit to first 5 rows)
    preview_df = df.head(5)
    preview_data = preview_df.to_dict(orient="records")
    
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview_data,
        "message": f"Successfully uploaded {file.filename} with {len(df)} records"
    }


@app.post("/score-cml-data", tags=["Predictions"], response_model=ScoreResponse)
async def score_cml_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Score CML data with trained ML model.
    
    Loads uploaded CML data, validates required columns, and generates
    elimination predictions with probability scores.
    
    Args:
        file: Uploaded CSV or Excel file with CML data
        
    Returns:
        Dictionary containing:
        - rows_scored: Number of records processed
        - results: List of predictions with probabilities (limited to 100)
        - model_info: Information about the model used
        
    Raises:
        HTTPException: If model is not loaded, file is invalid,
                      or required columns are missing
    """
    # Check model availability
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Please train the model first using ml/train_enhanced.py"
        )
    
    # Validate and read file
    validate_file_upload(file)
    df = read_uploaded_dataframe(file)
    
    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing required columns",
                "missing_columns": missing_cols,
                "required_columns": REQUIRED_COLUMNS
            }
        )
    
    # Validate feature columns for model
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing feature columns for model prediction",
                "missing_features": missing_features,
                "required_features": FEATURE_COLUMNS
            }
        )
    
    try:
        # Prepare features for prediction
        X = df[FEATURE_COLUMNS]
        
        # Generate predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of elimination
        
        # Prepare results
        results: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            results.append({
                "id_number": str(row["id_number"]),
                "predicted_elimination_flag": int(predictions[idx]),
                "elimination_probability": float(probabilities[idx]),
                "recommendation": "ELIMINATE" if predictions[idx] == 1 else "KEEP",
                "confidence": "HIGH" if abs(probabilities[idx] - 0.5) > 0.3 else "MODERATE"
            })
        
        logger.info(f"Successfully scored {len(df)} records from {file.filename}")
        
        # Return results (limit to first 100 for response size)
        return {
            "rows_scored": len(df),
            "results": results[:100],
            "total_results": len(results),
            "model_info": {
                "model_type": type(model).__name__,
                "features_used": FEATURE_COLUMNS
            },
            "message": f"Successfully scored {len(df)} CML records. "
                      f"Showing first 100 results. Download full results for complete data."
        }
    
    except Exception as e:
        logger.error(f"Prediction error for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error_type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
