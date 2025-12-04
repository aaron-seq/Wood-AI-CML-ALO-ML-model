"""Tests for forecasting module."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.forecasting import CMLForecaster


def test_forecaster_initialization():
    """Test forecaster can be initialized."""
    forecaster = CMLForecaster(
        minimum_thickness=3.0,
        safety_factor=1.5
    )
    
    assert forecaster.minimum_thickness == 3.0
    assert forecaster.safety_factor == 1.5


def test_calculate_remaining_life():
    """Test remaining life calculation."""
    forecaster = CMLForecaster()
    
    remaining_life = forecaster.calculate_remaining_life(
        current_thickness=10.0,
        corrosion_rate=0.10,
        min_thickness=3.0
    )
    
    # (10.0 - 3.0) / 0.10 = 70 years, capped at 50
    assert remaining_life == 50.0


def test_calculate_remaining_life_zero_corrosion():
    """Test remaining life with zero corrosion."""
    forecaster = CMLForecaster()
    
    remaining_life = forecaster.calculate_remaining_life(
        current_thickness=10.0,
        corrosion_rate=0.0
    )
    
    assert remaining_life == 50.0  # Maximum


def test_calculate_inspection_interval():
    """Test inspection interval calculation."""
    forecaster = CMLForecaster()
    
    interval = forecaster.calculate_inspection_interval(
        remaining_life_years=20.0,
        corrosion_rate=0.10
    )
    
    assert interval >= forecaster.min_inspection_interval
    assert interval <= forecaster.max_inspection_interval


def test_calculate_risk_level():
    """Test risk level classification."""
    forecaster = CMLForecaster()
    
    # Critical risk
    risk = forecaster.calculate_risk_level(
        remaining_life_years=0.5,
        corrosion_rate=0.30,
        current_thickness=4.0
    )
    assert risk == "CRITICAL"
    
    # Low risk
    risk = forecaster.calculate_risk_level(
        remaining_life_years=20.0,
        corrosion_rate=0.05,
        current_thickness=12.0
    )
    assert risk == "LOW"


def test_forecast_single_cml():
    """Test single CML forecast."""
    forecaster = CMLForecaster()
    
    forecast = forecaster.forecast_single_cml(
        id_number="CML-TEST-001",
        current_thickness=10.0,
        corrosion_rate=0.12,
        last_inspection_date=datetime.now()
    )
    
    assert forecast['id_number'] == "CML-TEST-001"
    assert 'remaining_life_years' in forecast
    assert 'next_inspection_date' in forecast
    assert 'risk_level' in forecast


def test_forecast_batch():
    """Test batch forecasting."""
    forecaster = CMLForecaster()
    
    df = pd.DataFrame({
        'id_number': ['CML-001', 'CML-002'],
        'thickness_mm': [10.0, 8.5],
        'average_corrosion_rate': [0.12, 0.18]
    })
    
    forecast_df = forecaster.forecast_batch(df)
    
    assert len(forecast_df) == 2
    assert 'remaining_life_years' in forecast_df.columns
    assert 'risk_level' in forecast_df.columns