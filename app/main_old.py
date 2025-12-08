from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from datetime import datetime

from app.schemas import HealthResponse, UploadResponse, ScoreResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title='Wood AI CML ALO API',
    version='0.3.0',
    description='Machine Learning API for CML optimization'
)

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / 'models' / 'cml_elimination_model.joblib'

# Load model
model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f'Model loaded successfully from {MODEL_PATH}')
    else:
        logger.warning(f'Model file not found at {MODEL_PATH}')
except Exception as e:
    logger.error(f'Failed to load model: {e}')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Engineer features to match training pipeline.'''
    # Corrosion-thickness ratio
    df['corrosion_thickness_ratio'] = df['average_corrosion_rate'] / df['thickness_mm']
    
    # Remaining life calculation
    min_thickness = 3.0
    df['remaining_life_years'] = (df['thickness_mm'] - min_thickness) / df['average_corrosion_rate']
    df['remaining_life_years'] = df['remaining_life_years'].clip(lower=0)
    
    # Days since inspection
    if 'last_inspection_date' in df.columns:
        df['last_inspection_date'] = pd.to_datetime(df['last_inspection_date'], errors='coerce')
        df['days_since_inspection'] = (pd.Timestamp.now() - df['last_inspection_date']).dt.days
        df['days_since_inspection'] = df['days_since_inspection'].fillna(365)
    else:
        df['days_since_inspection'] = 365
    
    # Risk score
    if 'risk_score' not in df.columns:
        df['risk_score'] = (
            df['average_corrosion_rate'] * 20 + 
            (10 - df['thickness_mm']) * 5
        ).clip(0, 100)
    
    return df


@app.get('/')
async def root():
    return {
        'message': 'Wood AI CML Optimization API',
        'version': '0.3.0',
        'documentation': '/docs'
    }


@app.get('/health', response_model=HealthResponse)
async def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH) if MODEL_PATH.exists() else None,
        'version': '0.3.0'
    }


@app.post('/upload-cml-data', response_model=UploadResponse)
async def upload_cml_data(file: UploadFile = File(...)):
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, 'Unsupported file format')
        
        logger.info(f'Successfully parsed {file.filename}: {len(df)} rows, {len(df.columns)} columns')
        
        return {
            'filename': file.filename,
            'rows': len(df),
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records'),
            'message': f'Successfully uploaded {len(df)} records'
        }
    except Exception as e:
        logger.error(f'Error reading file: {e}')
        raise HTTPException(500, f'Error processing file: {str(e)}')


@app.post('/score-cml-data', response_model=ScoreResponse)
async def score_cml_data(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, 'ML model not loaded')
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, 'Unsupported file format')
        
        logger.info(f'Scoring {len(df)} CMLs from {file.filename}')
        
        # Engineer features (IMPORTANT!)
        df = engineer_features(df)
        
        # Get feature columns (must match training)
        feature_cols = [
            'average_corrosion_rate',
            'thickness_mm',
            'commodity',
            'feature_type',
            'cml_shape',
            'remaining_life_years',
            'corrosion_thickness_ratio',
            'risk_score',
            'days_since_inspection'
        ]
        
        # Validate required columns
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        
        X = df[feature_cols]
        
        # Predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Build results
        results = []
        for idx, row in df.iterrows():
            results.append({
                'id_number': str(row['id_number']),
                'predicted_elimination_flag': int(predictions[idx]),
                'elimination_probability': float(probabilities[idx]),
                'recommendation': 'ELIMINATE' if predictions[idx] == 1 else 'KEEP',
                'confidence': 'HIGH' if abs(probabilities[idx] - 0.5) > 0.3 else 'MODERATE'
            })
        
        logger.info(f'Successfully scored {len(df)} CMLs')
        
        return {
            'rows_scored': len(df),
            'results': results[:100],
            'total_results': len(results),
            'model_info': {
                'model_type': type(model).__name__,
                'features_used': feature_cols
            },
            'message': f'Successfully scored {len(df)} CML records'
        }
        
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(500, f'Error during prediction: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
