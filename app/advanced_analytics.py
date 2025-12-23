"""Advanced Analytics and Visualization Module for CML Reports.

This module provides enhanced data visualization capabilities for CML
condition monitoring, including:
- API 570 compliant remaining life calculations
- Risk matrix heatmaps
- Corrosion rate trend analysis
- Statistical distribution analysis
- Commodity and feature type breakdowns
- Inspection scheduling optimization insights
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


def create_risk_matrix_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a risk matrix heatmap based on corrosion rate and thickness.

    Follows API 570 guidelines for risk-based inspection planning.
    Risk is determined by remaining life which is calculated from
    (current_thickness - min_thickness) / corrosion_rate.

    Args:
        df: DataFrame with columns 'average_corrosion_rate' and 'thickness_mm'

    Returns:
        Plotly Figure object with the risk matrix heatmap
    """
    # Create risk categories
    df = df.copy()
    df["risk_category"] = pd.cut(
        df["remaining_life_years"]
        if "remaining_life_years" in df.columns
        else (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01),
        bins=[0, 2, 5, 10, 20, float("inf")],
        labels=["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"],
    )

    # Create corrosion bins
    df["corrosion_bin"] = pd.cut(
        df["average_corrosion_rate"],
        bins=[0, 0.05, 0.10, 0.15, 0.20, float("inf")],
        labels=["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", ">0.20"],
    )

    # Create thickness bins
    df["thickness_bin"] = pd.cut(
        df["thickness_mm"],
        bins=[0, 5, 7, 9, 11, float("inf")],
        labels=["<5mm", "5-7mm", "7-9mm", "9-11mm", ">11mm"],
    )

    # Create pivot table for heatmap
    pivot = df.groupby(["thickness_bin", "corrosion_bin"]).size().unstack(fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0, "#1a472a"],  # Dark green - minimal risk
                [0.25, "#2e7d32"],  # Green - low risk
                [0.5, "#ffc107"],  # Yellow - medium risk
                [0.75, "#ff5722"],  # Orange - high risk
                [1, "#b71c1c"],  # Red - critical risk
            ],
            text=pivot.values,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            hoverongaps=False,
            hovertemplate=(
                "Thickness: %{y}<br>Corrosion Rate: %{x}<br>Count: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": "Risk Matrix: Corrosion Rate vs Wall Thickness",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        xaxis_title="Corrosion Rate (mm/year)",
        yaxis_title="Wall Thickness",
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        height=400,
    )

    return fig


def create_remaining_life_distribution(df: pd.DataFrame) -> go.Figure:
    """Create a histogram showing the distribution of remaining life.

    Based on API 570 remaining life formula:
    Remaining Life = (t_actual - t_minimum) / corrosion_rate

    Args:
        df: DataFrame with remaining_life_years column

    Returns:
        Plotly Figure with histogram and statistical markers
    """
    remaining_life = (
        df["remaining_life_years"]
        if "remaining_life_years" in df.columns
        else (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01)
    ).clip(upper=50)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=remaining_life,
            nbinsx=20,
            marker_color="#00AEEF",
            opacity=0.7,
            name="CML Count",
        )
    )

    # Add mean line
    mean_life = remaining_life.mean()
    fig.add_vline(
        x=mean_life,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text=f"Mean: {mean_life:.1f} years",
        annotation_position="top",
    )

    # Add critical threshold
    fig.add_vline(
        x=2,
        line_dash="dot",
        line_color="#f44336",
        annotation_text="Critical (<2 years)",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title={
            "text": "Remaining Life Distribution (API 570)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        xaxis_title="Remaining Life (Years)",
        yaxis_title="Number of CMLs",
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        showlegend=True,
        height=350,
    )

    return fig


def create_corrosion_by_commodity_chart(df: pd.DataFrame) -> go.Figure:
    """Create a box plot showing corrosion rate distribution by commodity.

    This visualization helps identify which commodities have higher
    corrosion rates, enabling targeted inspection programs.

    Args:
        df: DataFrame with 'commodity' and 'average_corrosion_rate' columns

    Returns:
        Plotly Figure with box plots per commodity
    """
    # Sort commodities by median corrosion rate
    commodity_order = (
        df.groupby("commodity")["average_corrosion_rate"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = px.box(
        df,
        x="commodity",
        y="average_corrosion_rate",
        color="commodity",
        category_orders={"commodity": commodity_order},
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )

    fig.update_layout(
        title={
            "text": "Corrosion Rate Distribution by Commodity",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        xaxis_title="Commodity Type",
        yaxis_title="Corrosion Rate (mm/year)",
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        showlegend=False,
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def create_inspection_priority_scatter(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot for inspection priority analysis.

    Plots corrosion rate vs thickness with size representing risk score.
    Color coding indicates remaining life categories.

    Args:
        df: DataFrame with corrosion, thickness, and risk data

    Returns:
        Plotly Figure with interactive scatter plot
    """
    df = df.copy()

    # Calculate remaining life if not present
    if "remaining_life_years" not in df.columns:
        df["remaining_life_years"] = (
            (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01)
        ).clip(0, 50)

    # Create risk categories
    df["risk_category"] = pd.cut(
        df["remaining_life_years"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    )

    color_map = {
        "CRITICAL": "#b71c1c",
        "HIGH": "#ff5722",
        "MEDIUM": "#ffc107",
        "LOW": "#4CAF50",
    }

    fig = px.scatter(
        df,
        x="average_corrosion_rate",
        y="thickness_mm",
        color="risk_category",
        size="remaining_life_years",
        hover_data=["id_number", "commodity", "feature_type"],
        color_discrete_map=color_map,
        category_orders={"risk_category": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
    )

    fig.update_layout(
        title={
            "text": "Inspection Priority Matrix",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        xaxis_title="Corrosion Rate (mm/year)",
        yaxis_title="Current Thickness (mm)",
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        legend_title="Risk Level",
        height=450,
    )

    # Add min thickness line
    fig.add_hline(
        y=3.0,
        line_dash="dash",
        line_color="#f44336",
        annotation_text="Minimum Thickness (3.0mm)",
    )

    return fig


def create_feature_type_analysis(df: pd.DataFrame) -> go.Figure:
    """Create a sunburst chart showing feature type distribution.

    Helps identify which component types are most prevalent and
    their associated risk levels.

    Args:
        df: DataFrame with 'feature_type' and 'commodity' columns

    Returns:
        Plotly Figure with sunburst chart
    """
    # Create hierarchical data
    df = df.copy()
    if "risk_category" not in df.columns:
        remaining_life = (
            df["remaining_life_years"]
            if "remaining_life_years" in df.columns
            else (df["thickness_mm"] - 3.0)
            / df["average_corrosion_rate"].clip(lower=0.01)
        ).clip(0, 50)
        df["risk_category"] = pd.cut(
            remaining_life,
            bins=[0, 2, 5, 10, float("inf")],
            labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )

    fig = px.sunburst(
        df,
        path=["feature_type", "commodity"],
        color="feature_type",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )

    fig.update_layout(
        title={
            "text": "CML Distribution by Feature Type and Commodity",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        height=450,
    )

    return fig


def create_corrosion_trend_gauge(df: pd.DataFrame) -> go.Figure:
    """Create gauge charts showing overall portfolio health.

    Displays key metrics as gauge indicators for executive dashboards.

    Args:
        df: DataFrame with CML data

    Returns:
        Plotly Figure with gauge indicators
    """
    # Calculate metrics
    remaining_life = (
        df["remaining_life_years"]
        if "remaining_life_years" in df.columns
        else (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01)
    ).clip(0, 50)

    avg_remaining_life = remaining_life.mean()
    critical_pct = (remaining_life < 2).sum() / len(df) * 100
    avg_corrosion = df["average_corrosion_rate"].mean()

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Avg Remaining Life", "Critical CMLs", "Avg Corrosion Rate"),
    )

    # Gauge 1: Average Remaining Life
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_remaining_life,
            number={"suffix": " years", "font": {"size": 24, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 50], "tickcolor": "#FFFFFF"},
                "bar": {"color": "#00AEEF"},
                "bgcolor": "#333333",
                "bordercolor": "#444444",
                "steps": [
                    {"range": [0, 5], "color": "#b71c1c"},
                    {"range": [5, 15], "color": "#ff5722"},
                    {"range": [15, 30], "color": "#ffc107"},
                    {"range": [30, 50], "color": "#4CAF50"},
                ],
            },
        ),
        row=1,
        col=1,
    )

    # Gauge 2: Critical CML Percentage
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=critical_pct,
            number={"suffix": "%", "font": {"size": 24, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FFFFFF"},
                "bar": {"color": "#f44336" if critical_pct > 10 else "#4CAF50"},
                "bgcolor": "#333333",
                "bordercolor": "#444444",
                "steps": [
                    {"range": [0, 5], "color": "#4CAF50"},
                    {"range": [5, 15], "color": "#ffc107"},
                    {"range": [15, 100], "color": "#b71c1c"},
                ],
            },
        ),
        row=1,
        col=2,
    )

    # Gauge 3: Average Corrosion Rate
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_corrosion,
            number={"suffix": " mm/yr", "font": {"size": 20, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 0.5], "tickcolor": "#FFFFFF"},
                "bar": {"color": "#00AEEF"},
                "bgcolor": "#333333",
                "bordercolor": "#444444",
                "steps": [
                    {"range": [0, 0.1], "color": "#4CAF50"},
                    {"range": [0.1, 0.2], "color": "#ffc107"},
                    {"range": [0.2, 0.5], "color": "#b71c1c"},
                ],
            },
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title={
            "text": "Portfolio Health Indicators",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        height=300,
    )

    return fig


def calculate_advanced_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive statistics for the CML dataset.

    Computes API 570 compliant metrics including:
    - Remaining life statistics
    - Corrosion rate analysis
    - Risk distribution
    - Inspection recommendations

    Args:
        df: DataFrame with CML data

    Returns:
        Dictionary containing all computed statistics
    """
    # Calculate remaining life if not present
    if "remaining_life_years" not in df.columns:
        remaining_life = (
            (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01)
        ).clip(0, 50)
    else:
        remaining_life = df["remaining_life_years"].clip(0, 50)

    # Risk categorization
    risk_counts = pd.cut(
        remaining_life,
        bins=[0, 2, 5, 10, float("inf")],
        labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    ).value_counts()

    # Calculate inspection intervals based on API 570
    inspection_intervals = (
        remaining_life.apply(lambda x: min(x / 1.5, 6) if x > 0 else 0.5) * 12
    )  # Convert to months

    stats = {
        "total_cmls": len(df),
        "remaining_life": {
            "mean": float(remaining_life.mean()),
            "median": float(remaining_life.median()),
            "std": float(remaining_life.std()),
            "min": float(remaining_life.min()),
            "max": float(remaining_life.max()),
            "percentile_10": float(remaining_life.quantile(0.1)),
            "percentile_25": float(remaining_life.quantile(0.25)),
            "percentile_75": float(remaining_life.quantile(0.75)),
            "percentile_90": float(remaining_life.quantile(0.9)),
        },
        "corrosion_rate": {
            "mean": float(df["average_corrosion_rate"].mean()),
            "median": float(df["average_corrosion_rate"].median()),
            "std": float(df["average_corrosion_rate"].std()),
            "max": float(df["average_corrosion_rate"].max()),
            "high_corrosion_count": int((df["average_corrosion_rate"] > 0.15).sum()),
        },
        "thickness": {
            "mean": float(df["thickness_mm"].mean()),
            "min": float(df["thickness_mm"].min()),
            "below_min_count": int((df["thickness_mm"] < 3.0).sum()),
            "thin_wall_count": int((df["thickness_mm"] < 5.0).sum()),
        },
        "risk_distribution": {
            "critical": int(risk_counts.get("CRITICAL", 0)),
            "high": int(risk_counts.get("HIGH", 0)),
            "medium": int(risk_counts.get("MEDIUM", 0)),
            "low": int(risk_counts.get("LOW", 0)),
        },
        "inspection_scheduling": {
            "immediate_attention": int((remaining_life < 1).sum()),
            "next_6_months": int((remaining_life < 2).sum()),
            "next_12_months": int((remaining_life < 3).sum()),
            "avg_interval_months": float(inspection_intervals.mean()),
        },
        "commodity_risk": df.groupby("commodity")
        .apply(
            lambda x: {
                "count": len(x),
                "avg_corrosion": float(x["average_corrosion_rate"].mean()),
                "critical_count": int(
                    (
                        (x["thickness_mm"] - 3.0)
                        / x["average_corrosion_rate"].clip(lower=0.01)
                        < 2
                    ).sum()
                ),
            }
        )
        .to_dict(),
    }

    return stats


def create_timeline_forecast_chart(df: pd.DataFrame) -> go.Figure:
    """Create a timeline showing projected inspection needs.

    Visualizes when CMLs will require inspection based on remaining life
    and recommended inspection intervals.

    Args:
        df: DataFrame with CML data

    Returns:
        Plotly Figure with timeline visualization
    """
    df = df.copy()

    # Calculate remaining life and next inspection
    if "remaining_life_years" not in df.columns:
        df["remaining_life_years"] = (
            (df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01)
        ).clip(0, 50)

    # Calculate inspection months
    df["inspection_months"] = (
        df["remaining_life_years"].apply(lambda x: min(x / 1.5, 6) if x > 0 else 0.5)
        * 12
    )

    # Create time buckets
    time_buckets = (
        pd.cut(
            df["inspection_months"],
            bins=[0, 6, 12, 24, 36, 72, float("inf")],
            labels=["0-6mo", "6-12mo", "1-2yr", "2-3yr", "3-6yr", ">6yr"],
        )
        .value_counts()
        .sort_index()
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=time_buckets.index.tolist(),
                y=time_buckets.values,
                marker_color=[
                    "#b71c1c",
                    "#ff5722",
                    "#ffc107",
                    "#8bc34a",
                    "#4CAF50",
                    "#00AEEF",
                ],
                text=time_buckets.values,
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Inspection Timeline Forecast",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#00AEEF"},
        },
        xaxis_title="Time Until Next Inspection",
        yaxis_title="Number of CMLs",
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        height=350,
    )

    return fig
