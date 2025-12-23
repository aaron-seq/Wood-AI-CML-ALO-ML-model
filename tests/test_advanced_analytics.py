"""Tests for the advanced analytics module."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.advanced_analytics import (
    create_risk_matrix_heatmap,
    create_remaining_life_distribution,
    create_corrosion_by_commodity_chart,
    create_inspection_priority_scatter,
    create_feature_type_analysis,
    create_corrosion_trend_gauge,
    calculate_advanced_statistics,
    create_timeline_forecast_chart,
)


@pytest.fixture
def sample_cml_dataframe():
    """Create a sample CML dataframe for testing."""
    return pd.DataFrame(
        {
            "id_number": ["CML-001", "CML-002", "CML-003", "CML-004", "CML-005"],
            "average_corrosion_rate": [0.12, 0.08, 0.25, 0.05, 0.18],
            "thickness_mm": [9.5, 10.2, 6.0, 12.0, 7.5],
            "commodity": ["Crude Oil", "Natural Gas", "Steam", "Fuel Gas", "Crude Oil"],
            "feature_type": ["Pipe", "Elbow", "Reducer", "Pipe", "Tee"],
            "cml_shape": ["Both", "Internal", "External", "Internal", "Both"],
            "remaining_life_years": [54.2, 90.0, 12.0, 150.0, 25.0],
        }
    )


def test_risk_matrix_heatmap(sample_cml_dataframe):
    """Test risk matrix heatmap generation."""
    fig = create_risk_matrix_heatmap(sample_cml_dataframe)
    assert fig is not None
    assert fig.layout.title.text == "Risk Matrix: Corrosion Rate vs Wall Thickness"


def test_remaining_life_distribution(sample_cml_dataframe):
    """Test remaining life distribution chart."""
    fig = create_remaining_life_distribution(sample_cml_dataframe)
    assert fig is not None
    assert "Remaining Life Distribution" in fig.layout.title.text


def test_corrosion_by_commodity_chart(sample_cml_dataframe):
    """Test corrosion by commodity box plot."""
    fig = create_corrosion_by_commodity_chart(sample_cml_dataframe)
    assert fig is not None
    assert "Corrosion Rate Distribution by Commodity" in fig.layout.title.text


def test_inspection_priority_scatter(sample_cml_dataframe):
    """Test inspection priority scatter plot."""
    fig = create_inspection_priority_scatter(sample_cml_dataframe)
    assert fig is not None
    assert "Inspection Priority Matrix" in fig.layout.title.text


def test_feature_type_analysis(sample_cml_dataframe):
    """Test feature type sunburst chart."""
    fig = create_feature_type_analysis(sample_cml_dataframe)
    assert fig is not None


def test_corrosion_trend_gauge(sample_cml_dataframe):
    """Test portfolio health gauge indicators."""
    fig = create_corrosion_trend_gauge(sample_cml_dataframe)
    assert fig is not None
    assert "Portfolio Health Indicators" in fig.layout.title.text


def test_timeline_forecast_chart(sample_cml_dataframe):
    """Test inspection timeline forecast chart."""
    fig = create_timeline_forecast_chart(sample_cml_dataframe)
    assert fig is not None
    assert "Inspection Timeline Forecast" in fig.layout.title.text


def test_calculate_advanced_statistics(sample_cml_dataframe):
    """Test advanced statistics calculation."""
    stats = calculate_advanced_statistics(sample_cml_dataframe)

    assert "total_cmls" in stats
    assert stats["total_cmls"] == 5

    assert "remaining_life" in stats
    assert "mean" in stats["remaining_life"]
    assert "median" in stats["remaining_life"]

    assert "corrosion_rate" in stats
    assert "mean" in stats["corrosion_rate"]

    assert "risk_distribution" in stats
    assert "critical" in stats["risk_distribution"]
    assert "high" in stats["risk_distribution"]
    assert "medium" in stats["risk_distribution"]
    assert "low" in stats["risk_distribution"]

    assert "inspection_scheduling" in stats
    assert "immediate_attention" in stats["inspection_scheduling"]


def test_statistics_with_missing_remaining_life():
    """Test statistics calculation when remaining_life_years is missing."""
    df = pd.DataFrame(
        {
            "id_number": ["CML-001", "CML-002"],
            "average_corrosion_rate": [0.12, 0.08],
            "thickness_mm": [9.5, 10.2],
            "commodity": ["Crude Oil", "Natural Gas"],
            "feature_type": ["Pipe", "Elbow"],
            "cml_shape": ["Both", "Internal"],
        }
    )

    stats = calculate_advanced_statistics(df)

    assert "total_cmls" in stats
    assert stats["total_cmls"] == 2
    assert "remaining_life" in stats
