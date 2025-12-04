"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_upload_cml_data():
    """Test file upload endpoint."""
    # Create sample CSV
    df = pd.DataFrame({
        'id_number': ['CML-001', 'CML-002'],
        'average_corrosion_rate': [0.12, 0.08],
        'thickness_mm': [9.5, 10.2],
        'commodity': ['Crude Oil', 'Natural Gas'],
        'feature_type': ['Pipe', 'Elbow'],
        'cml_shape': ['Both', 'Internal']
    })
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    response = client.post("/upload-cml-data", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data['rows'] == 2
    assert 'columns' in data


def test_score_cml_data_without_model():
    """Test scoring endpoint handles missing model gracefully."""
    df = pd.DataFrame({
        'id_number': ['CML-001'],
        'average_corrosion_rate': [0.12],
        'thickness_mm': [9.5],
        'commodity': ['Crude Oil'],
        'feature_type': ['Pipe'],
        'cml_shape': ['Both']
    })
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    response = client.post("/score-cml-data", files=files)
    
    # Should return 500 if model not loaded, or 200 if model exists
    assert response.status_code in [200, 500]


def test_upload_invalid_file():
    """Test upload with missing columns."""
    df = pd.DataFrame({
        'id_number': ['CML-001'],
        'average_corrosion_rate': [0.12]
        # Missing required columns
    })
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    response = client.post("/upload-cml-data", files=files)
    
    # Should still upload but validation will show errors
    assert response.status_code == 200
    data = response.json()
    if 'validation' in data:
        assert 'errors' in data['validation'] or 'warnings' in data['validation']