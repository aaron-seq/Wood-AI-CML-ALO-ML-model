"""Enhanced FastAPI application with full CML optimization features."""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
from pathlib import Path
import joblib
from typing import List, Optional
import io
from datetime import datetime

# Import local modules
from app.schemas import (
    CMLDataInput, CMLPrediction, SMEOverride, ForecastInput, ForecastOutput,
    HealthResponse, UploadResponse
)
from app.config import settings
from app.utils import validate_cml_dataframe, generate_elimination_report
from app.forecasting import CMLForecaster
from app.sme_override import SMEOverrideManager

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Global instances
model = None
forecaster = CMLForecaster(
    minimum_thickness=settings.DEFAULT_MINIMUM_THICKNESS,
    safety_factor=settings.SAFETY_FACTOR
)
sme_manager = SMEOverrideManager(settings.SME_OVERRIDE_FILE)

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    model_path = settings.MODEL_DIR / "cml_elimination_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"No model found at {model_path}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now()
    }


@app.post("/upload-cml-data", response_model=UploadResponse)
async def upload_cml_data(file: UploadFile = File(...)):
    """Upload and validate CML data file."""
    try:
        # Try reading as Excel first
        df = pd.read_excel(file.file)
    except Exception:
        # Fall back to CSV
        file.file.seek(0)
        df = pd.read_csv(file.file)
    
    # Validate dataframe
    validation = validate_cml_dataframe(df)
    
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
        "validation": validation
    }


@app.post("/score-cml-data")
async def score_cml_data(file: UploadFile = File(...)):
    """Score CML data for elimination recommendations."""
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Train a model first using /train endpoint or ml/train_enhanced.py"
        )
    
    try:
        df = pd.read_excel(file.file)
    except Exception:
        file.file.seek(0)
        df = pd.read_csv(file.file)
    
    required_cols = [
        "average_corrosion_rate", "thickness_mm",
        "commodity", "feature_type", "cml_shape"
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )
    
    # Engineer features
    df['corrosion_thickness_ratio'] = df['average_corrosion_rate'] / df['thickness_mm']
    
    # Prepare features for prediction
    features = required_cols + ['corrosion_thickness_ratio']
    X = df[features]
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Build results
    results = []
    for idx, row in df.iterrows():
        elim_prob = float(probabilities[idx, 1])
        keep_prob = float(probabilities[idx, 0])
        
        results.append({
            "id_number": row.get('id_number', f"CML-{idx+1}"),
            "predicted_elimination": int(predictions[idx]),
            "elimination_probability": elim_prob,
            "keep_probability": keep_prob,
            "recommendation": "ELIMINATE" if predictions[idx] == 1 else "KEEP",
            "confidence": max(elim_prob, keep_prob),
            "confidence_level": "HIGH" if max(elim_prob, keep_prob) > 0.8 else "MEDIUM" if max(elim_prob, keep_prob) > 0.6 else "LOW"
        })
    
    # Generate summary
    results_df = pd.DataFrame(results)
    summary = {
        "total_cmls": len(df),
        "recommended_eliminations": int((predictions == 1).sum()),
        "recommended_keep": int((predictions == 0).sum()),
        "elimination_rate": float((predictions == 1).sum() / len(predictions)),
        "high_confidence_eliminations": int(sum(1 for r in results if r['predicted_elimination'] == 1 and r['confidence_level'] == 'HIGH'))
    }
    
    return {
        "rows_scored": len(df),
        "results": results[:100],  # Limit to 100 for response size
        "summary": summary
    }


@app.post("/forecast-remaining-life")
async def forecast_remaining_life(file: UploadFile = File(...)):
    """Forecast remaining life and inspection schedules."""
    try:
        df = pd.read_excel(file.file)
    except Exception:
        file.file.seek(0)
        df = pd.read_csv(file.file)
    
    required_cols = ['average_corrosion_rate', 'thickness_mm']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )
    
    # Generate forecasts
    forecast_df = forecaster.forecast_batch(df)
    forecast_summary = forecaster.generate_forecast_summary(df)
    
    # Convert to records
    forecasts = forecast_df[[
        'id_number', 'remaining_life_years', 'next_inspection_date',
        'recommended_inspection_frequency_months', 'risk_level'
    ]].to_dict('records') if 'id_number' in forecast_df.columns else []
    
    return {
        "forecasts": forecasts[:100],
        "summary": forecast_summary
    }


@app.post("/sme-override")
async def add_sme_override(override: SMEOverride):
    """Add SME manual override for a CML decision."""
    result = sme_manager.add_override(
        id_number=override.id_number,
        sme_decision=override.sme_decision,
        reason=override.reason,
        sme_name=override.sme_name,
        original_prediction=override.original_prediction,
        original_probability=override.original_probability
    )
    
    return {
        "status": "success",
        "override": result
    }


@app.get("/sme-override/{id_number}")
async def get_sme_override(id_number: str):
    """Get SME override for a specific CML."""
    override = sme_manager.get_override(id_number)
    
    if override is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SME override found for {id_number}"
        )
    
    return override


@app.get("/sme-overrides/statistics")
async def get_sme_statistics():
    """Get statistics about SME overrides."""
    return sme_manager.get_override_statistics()


@app.post("/generate-report")
async def generate_report(file: UploadFile = File(...)):
    """Generate comprehensive elimination report."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        df = pd.read_excel(file.file)
    except Exception:
        file.file.seek(0)
        df = pd.read_csv(file.file)
    
    # Score the data first
    required_cols = [
        "average_corrosion_rate", "thickness_mm",
        "commodity", "feature_type", "cml_shape"
    ]
    
    df['corrosion_thickness_ratio'] = df['average_corrosion_rate'] / df['thickness_mm']
    X = df[required_cols + ['corrosion_thickness_ratio']]
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to dataframe
    df['predicted_elimination'] = predictions
    df['elimination_probability'] = probabilities[:, 1]
    df['recommendation'] = ['ELIMINATE' if p == 1 else 'KEEP' for p in predictions]
    df['confidence_level'] = ['HIGH' if max(p) > 0.8 else 'MEDIUM' if max(p) > 0.6 else 'LOW' for p in probabilities]
    
    # Apply SME overrides
    df_with_overrides = sme_manager.apply_overrides_to_predictions(df)
    
    # Generate report
    report = generate_elimination_report(df_with_overrides)
    
    return report


@app.get("/download-predictions/{filename}")
async def download_predictions(filename: str):
    """Download predictions as CSV file."""
    file_path = settings.UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type='text/csv',
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)