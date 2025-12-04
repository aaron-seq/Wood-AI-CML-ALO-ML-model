"""Tests for utility functions."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils import validate_cml_dataframe, calculate_inspection_schedule


def test_validate_cml_dataframe_valid():
    """Test validation with valid dataframe."""
    df = pd.DataFrame({
        'id_number': ['CML-001', 'CML-002'],
        'average_corrosion_rate': [0.12, 0.08],
        'thickness_mm': [9.5, 10.2],
        'commodity': ['Crude Oil', 'Natural Gas'],
        'feature_type': ['Pipe', 'Elbow'],
        'cml_shape': ['Both', 'Internal']
    })
    
    result = validate_cml_dataframe(df)
    
    assert result['valid'] == True
    assert result['stats']['total_records'] == 2
    assert len(result['errors']) == 0


def test_validate_cml_dataframe_missing_columns():
    """Test validation with missing required columns."""
    df = pd.DataFrame({
        'id_number': ['CML-001'],
        'average_corrosion_rate': [0.12]
    })
    
    result = validate_cml_dataframe(df)
    
    assert result['valid'] == False
    assert len(result['errors']) > 0


def test_validate_cml_dataframe_duplicates():
    """Test validation detects duplicate IDs."""
    df = pd.DataFrame({
        'id_number': ['CML-001', 'CML-001'],  # Duplicate
        'average_corrosion_rate': [0.12, 0.08],
        'thickness_mm': [9.5, 10.2],
        'commodity': ['Crude Oil', 'Natural Gas'],
        'feature_type': ['Pipe', 'Elbow'],
        'cml_shape': ['Both', 'Internal']
    })
    
    result = validate_cml_dataframe(df)
    
    assert len(result['warnings']) > 0


def test_calculate_inspection_schedule():
    """Test inspection schedule calculation."""
    result = calculate_inspection_schedule(
        corrosion_rate=0.12,
        thickness=9.5,
        min_thickness=3.0,
        safety_factor=1.5
    )
    
    assert 'remaining_life_years' in result
    assert 'inspection_interval_months' in result
    assert 'risk_level' in result
    assert result['remaining_life_years'] > 0
    assert result['inspection_interval_months'] > 0


def test_calculate_inspection_schedule_high_risk():
    """Test inspection schedule for high-risk CML."""
    result = calculate_inspection_schedule(
        corrosion_rate=0.35,  # High corrosion
        thickness=5.0,        # Thin wall
        min_thickness=3.0,
        safety_factor=1.5
    )
    
    assert result['risk_level'] in ['HIGH', 'CRITICAL']
    assert result['remaining_life_years'] < 10