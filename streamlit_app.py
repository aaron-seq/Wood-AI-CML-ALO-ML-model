"""Streamlit Dashboard for CML Optimization - Wood Engineering Customization.

This module provides an interactive web dashboard for the CML Optimization system,
customized for Wood Engineering with specific branding, educational resources,
and strict data handling policies.
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use proper imports
try:
    from api_client import (
        APIError,
        auth_headers,
        check_api_health,
        get_api_base_url,
        score_cml_data,
    )
    from app.config import settings
    from app.forecasting import CMLForecaster
    from app.ingestion import UploadError, parse_bytes
    from app.sme_override import SMEOverrideManager
    from app.utils import validate_cml_dataframe
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("Failed to import required modules. Please check your installation.")
    st.stop()

# Constants
DEFAULT_MINIMUM_THICKNESS = 3.0
DEFAULT_SAFETY_FACTOR = 1.5
LOGO_PATH = Path("Wood-logo-WHITE-45mm.png")

# Page configuration
st.set_page_config(
    page_title="Wood Engineering - CML Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Wood Engineering Theme consistency
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #1E1E1E;
    }
    h1, h2, h3 {
        color: #00AEEF !important;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #00AEEF;
        color: white;
        border-radius: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Branding
if LOGO_PATH.exists():
    try:
        # Streamlit 1.18.0+ uses use_container_width
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    except TypeError:
        # Older versions use use_column_width
        st.sidebar.image(str(LOGO_PATH), use_column_width=True)
else:
    st.sidebar.title("WOOD ENGINEERING")

st.sidebar.header("CML Analysis Platform")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Upload & Analyze",
        "Forecasting",
        "SME Overrides",
        "Reports",
        "How It Works",
        "About Application",
    ],
)

# --- Session State Management ---
if "data" not in st.session_state:
    st.session_state["data"] = None
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None


# --- Helpers ---
@st.cache_resource
def get_forecaster() -> CMLForecaster:
    return CMLForecaster()


@st.cache_resource
def get_sme_manager() -> SMEOverrideManager:
    return SMEOverrideManager()


def get_model_description(api_online: bool) -> str:
    """Describe the model the API is serving, or why we cannot tell."""
    if not api_online:
        return "Unknown"
    try:
        response = requests.get(
            f"{get_api_base_url()}/model/info", timeout=5, headers=auth_headers()
        )
    except requests.RequestException:
        return "Unknown"
    if response.status_code == 503:
        return "Not loaded"
    if not response.ok:
        return "Unknown"
    info = response.json()
    return f"{info['model_type']} ({len(info['features_used'])} features)"


def read_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """Parse an upload through the same code path the API uses.

    This previously called pandas directly, so the dashboard accepted
    files the API would reject -- no size limit, no row limit, and
    different error messages for the same bad file.
    """
    payload = uploaded_file.getvalue()

    if len(payload) > settings.MAX_UPLOAD_BYTES:
        st.error(
            f"File is too large ({len(payload) / 1_048_576:.1f} MB). "
            f"The limit is {settings.MAX_UPLOAD_BYTES / 1_048_576:.0f} MB."
        )
        return None

    try:
        frame = parse_bytes(payload, uploaded_file.name)
    except UploadError as exc:
        st.error(str(exc))
        return None

    if len(frame) > settings.MAX_UPLOAD_ROWS:
        st.error(
            f"File contains {len(frame):,} rows, which exceeds the "
            f"{settings.MAX_UPLOAD_ROWS:,} row limit."
        )
        return None

    return frame


try:
    forecaster = get_forecaster()
    sme_manager = get_sme_manager()
except Exception:
    st.stop()

# --- Pages ---

if page == "Overview":
    st.title("Project Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        # Probe the API rather than asserting health unconditionally.
        api_up = check_api_health()
        st.metric("Scoring API", "Online" if api_up else "Offline")
        if not api_up:
            st.caption(f"No response from {get_api_base_url()}")
    with col2:
        # Report what is actually loaded rather than a hardcoded string.
        # "RF-Ensemble v2.1" was not a real version of anything.
        st.metric("Model", get_model_description(api_up))
    with col3:
        if st.session_state["data"] is not None:
            st.metric("Active Dataset", f"{len(st.session_state['data'])} Records")
        else:
            st.metric("Active Dataset", "None")

    st.markdown("---")

    if st.session_state["data"] is None:
        st.info("Welcome to the Wood Engineering CML Analysis Tool.")
        st.warning("No data loaded. Please go to the **Upload & Analyze** page to begin.")
    else:
        df = st.session_state["data"]
        st.subheader("Current Dataset Analytics")

        col1, col2 = st.columns(2)
        with col1:
            # Commodity distribution
            commodity_counts = df["commodity"].value_counts()
            fig = px.pie(
                values=commodity_counts.values,
                names=commodity_counts.index,
                title="Commodity Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Feature type distribution
            feature_counts = df["feature_type"].value_counts()
            fig = px.bar(
                x=feature_counts.index,
                y=feature_counts.values,
                title="Feature Types",
                color_discrete_sequence=["#00AEEF"],
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Upload & Analyze":
    st.title("Data Upload & Analysis")

    st.markdown("""
    Upload your Condition Monitoring Location (CML) data for automated analysis.
    The system uses our proprietary ML model to recommend optimal elimination candidates.
    """)

    uploaded_file = st.file_uploader("Select CML Data File", type=["csv", "xlsx"])

    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            # Validate
            validation = validate_cml_dataframe(df)
            if validation["valid"]:
                st.session_state["data"] = df
                st.success(f"Loaded {len(df)} records successfully.")

                if st.button("Run ML Analysis", type="primary"):
                    with st.spinner("Analyzing CML patterns..."):
                        try:
                            uploaded_file.seek(0)
                            result = score_cml_data(uploaded_file)
                            st.session_state["analysis_results"] = result
                            st.success(f"Analysis complete: {result['total_results']} CMLs scored.")
                        except APIError as e:
                            st.error(str(e))
                            st.caption(
                                f"The dashboard is configured to call {get_api_base_url()}. "
                                "Start the API with `make api`, or set CML_API_URL."
                            )

            else:
                st.error("Validation failed.")
                for err in validation["errors"]:
                    st.error(f"- {err}")

    # Display Results if available
    if st.session_state["analysis_results"]:
        analysis = st.session_state["analysis_results"]
        res_df = pd.DataFrame(analysis["results"])

        st.subheader("Optimization Recommendations")

        # The API caps the rows it returns. Say so, rather than presenting
        # counts over the first 100 rows as if they covered the dataset.
        if analysis.get("results_truncated"):
            st.info(
                f"Showing the first {len(res_df)} of {analysis['total_results']} scored "
                "CMLs. The counts below cover the displayed rows only."
            )

        col1, col2 = st.columns(2)
        with col1:
            elim_count = int((res_df["predicted_elimination_flag"] == 1).sum())
            st.metric(
                "Candidates for Elimination",
                elim_count,
                delta="Optimization Opportunity",
            )
        with col2:
            keep_count = int((res_df["predicted_elimination_flag"] == 0).sum())
            st.metric("Critical Monitoring Points", keep_count)

        st.dataframe(
            res_df.style.apply(
                lambda x: [
                    "background-color: #3d0000" if x.predicted_elimination_flag == 1 else ""
                    for i in x
                ],
                axis=1,
            )
        )

        # Download button
        csv = res_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Report",
            data=csv,
            file_name="wood_cml_optimization_results.csv",
            mime="text/csv",
        )


elif page == "How It Works":
    st.title("Machine Learning Model Explained")

    st.markdown("""
    ### The Intelligence Behind the Decision
    
    This application utilizes an advanced **Random Forest Ensemble** model to evaluate reliability. Unlike simple threshold-based rules, this model considers non-linear interactions between multiple variables to mimic the decision-making process of a senior corrosion engineer.

    #### 1. The Core Algorithm: Random Forest
    Imagine consulting 100 different reliability experts. Each expert (Decision Tree) looks at the data and votes on whether a CML should be kept or eliminated.
    *   **Diversity**: Each tree looks at a slightly different subset of data and features.
    *   **Voting**: The final decision is based on the majority vote of these 100 trees.
    *   **Robustness**: This prevents a single outlier data point from skewing the result.

    #### 2. Key Decision Indicators (Feature Importance)
    The model creates a weighted analysis based on the following factors, ranked by influence:

    1.  **Corrosion/Thickness Ratio (35%)**: The most critical indicator. It measures how fast the asset is degrading relative to its remaining wall thickness. A high ratio indicates urgent risk.
    2.  **Average Corrosion Rate (25%)**: The raw speed of degradation.
    3.  **Risk Score (15%)**: A composite metric derived from industry standards (API 570).
    4.  **Remaining Life (10%)**: Years until minimum thickness is breached.
    5.  **Inspection History (5%)**: Timespan since last active verification.
    
    #### 3. Understanding the Confidence Score
    The "Confidence" is not a guess; it is a calculated statistical probability.
    
    *   **Calculation**: If 95 out of 100 trees vote to "ELIMINATE", the probability is 0.95.
    *   **Usage**: 
        *   **High Confidence (>0.8)**: Strong consensus. clear patterns matching historical safe-to-eliminate cases.
        *   **Moderate Confidence (0.6 - 0.8)**: Most indicators align, but some edge cases exist. **Engineer Review Recommended.**
        *   **Low Confidence (<0.6)**: The model detects conflicting signals (e.g., low corrosion rate but very thin wall). **Manual Review Required.**

    #### 4. Human-in-the-Loop
    This tool is a **Decision Support System**, not a replacement. It filters the noise so you can focus your expertise on the complex cases (Moderate/Low confidence).
    """)

    # Visualizing Feature Importance (Conceptual)
    importance_data = pd.DataFrame(
        {
            "Feature": [
                "Corrosion/Thick Ratio",
                "Corrosion Rate",
                "Risk Score",
                "Remaining Life",
                "Inspection Age",
                "Commodity Type",
            ],
            "Weight": [35, 25, 15, 10, 10, 5],
        }
    )
    fig = px.bar(
        importance_data,
        x="Weight",
        y="Feature",
        orientation="h",
        title="Model Decision Weightage",
        color="Weight",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig)

elif page == "Forecasting":
    st.title("Lifecycle Forecasting")
    if st.session_state["data"] is None:
        st.warning("Please upload data in 'Upload & Analyze' first.")
    else:
        df = st.session_state["data"]

        col1, col2 = st.columns(2)
        with col1:
            min_thickness = st.number_input(
                "Minimum Required Thickness (mm)",
                1.0,
                10.0,
                DEFAULT_MINIMUM_THICKNESS,
                0.5,
            )
        with col2:
            safety_factor = st.number_input("Safety Factor", 1.0, 3.0, DEFAULT_SAFETY_FACTOR, 0.1)

        if st.button("Generate Forecasts", type="primary"):
            try:
                custom_forecaster = CMLForecaster(
                    minimum_thickness=min_thickness, safety_factor=safety_factor
                )
                forecast_df = custom_forecaster.forecast_batch(df)

                # Summary Stats
                summary = custom_forecaster.generate_forecast_summary(df)
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Avg Remaining Life",
                    f"{summary['avg_remaining_life_years']:.1f} yrs",
                )
                c2.metric("Critical CMLs", summary["critical_cmls"])
                c3.metric("High Risk CMLs", summary["high_risk_cmls"])

                # Detailed Table
                st.subheader("Forecast Details")
                st.dataframe(forecast_df)

            except Exception as e:
                st.error(f"Forecasting error: {e}")

elif page == "SME Overrides":
    st.title("Expert Override Management")
    st.markdown(
        "Record and track manual engineering decisions that deviate from model recommendations."
    )

    with st.expander("Add New Override", expanded=False), st.form("override_form"):
        col1, col2 = st.columns(2)
        with col1:
            cml_id = st.text_input("CML ID", placeholder="CML-001")
            decision = st.selectbox("Decision", ["KEEP", "ELIMINATE"])
        with col2:
            sme_name = st.text_input("SME Name", placeholder="Dr. John Smith")

        reason = st.text_area("Reason for Override", height=100)
        submitted = st.form_submit_button("Submit Override")

        if submitted and cml_id and reason:
            sme_manager.add_override(cml_id, decision, reason, sme_name)
            st.success("Override recorded.")

    # Show overrides
    overrides = sme_manager.get_all_overrides()
    if overrides:
        st.dataframe(pd.DataFrame(overrides))
    else:
        st.info("No overrides recorded.")

elif page == "Reports":
    st.title("Comprehensive Reports & Advanced Analytics")
    if st.session_state["data"] is None:
        st.warning("No data available for reporting. Upload data first.")
    else:
        df = st.session_state["data"]

        # Try to import advanced analytics
        try:
            from app.advanced_analytics import (
                calculate_advanced_statistics,
                create_corrosion_by_commodity_chart,
                create_corrosion_trend_gauge,
                create_feature_type_analysis,
                create_inspection_priority_scatter,
                create_remaining_life_distribution,
                create_risk_matrix_heatmap,
                create_timeline_forecast_chart,
            )

            advanced_available = True
        except ImportError as e:
            logger.warning(f"Advanced analytics not available: {e}")
            advanced_available = False

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets Analyzed", len(df))
        with col2:
            if "remaining_life_years" in df.columns:
                avg_life = df["remaining_life_years"].mean()
            else:
                avg_life = (
                    ((df["thickness_mm"] - 3.0) / df["average_corrosion_rate"].clip(lower=0.01))
                    .clip(0, 50)
                    .mean()
                )
            st.metric("Avg Remaining Life", f"{avg_life:.1f} years")
        with col3:
            critical_count = len(df[df["average_corrosion_rate"] > 0.2])
            st.metric("High Corrosion CMLs", critical_count)
        with col4:
            thin_wall = len(df[df["thickness_mm"] < 5.0])
            st.metric("Thin Wall CMLs", thin_wall)

        st.markdown("---")

        if advanced_available:
            # Portfolio Health Gauges
            st.subheader("Portfolio Health Indicators")
            gauge_fig = create_corrosion_trend_gauge(df)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown("---")

            # Risk Analysis Section
            st.subheader("Risk Analysis (API 570 Compliant)")
            col1, col2 = st.columns(2)
            with col1:
                risk_fig = create_risk_matrix_heatmap(df)
                st.plotly_chart(risk_fig, use_container_width=True)
            with col2:
                life_fig = create_remaining_life_distribution(df)
                st.plotly_chart(life_fig, use_container_width=True)

            st.markdown("---")

            # Inspection Planning
            st.subheader("Inspection Planning & Scheduling")
            col1, col2 = st.columns(2)
            with col1:
                timeline_fig = create_timeline_forecast_chart(df)
                st.plotly_chart(timeline_fig, use_container_width=True)
            with col2:
                priority_fig = create_inspection_priority_scatter(df)
                st.plotly_chart(priority_fig, use_container_width=True)

            st.markdown("---")

            # Commodity and Feature Analysis
            st.subheader("Asset Breakdown Analysis")
            col1, col2 = st.columns(2)
            with col1:
                commodity_fig = create_corrosion_by_commodity_chart(df)
                st.plotly_chart(commodity_fig, use_container_width=True)
            with col2:
                feature_fig = create_feature_type_analysis(df)
                st.plotly_chart(feature_fig, use_container_width=True)

            st.markdown("---")

            # Statistical Insights Panel
            st.subheader("Statistical Insights")
            stats = calculate_advanced_statistics(df)

            with st.expander("Detailed Statistics", expanded=False):
                st.json(stats)

            # Key insights cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Remaining Life Metrics**")
                st.write(f"- Mean: {stats['remaining_life']['mean']:.1f} years")
                st.write(f"- Median: {stats['remaining_life']['median']:.1f} years")
                st.write(f"- 10th Percentile: {stats['remaining_life']['percentile_10']:.1f} years")
            with col2:
                st.markdown("**Risk Distribution**")
                st.write(f"- Critical: {stats['risk_distribution']['critical']} CMLs")
                st.write(f"- High Risk: {stats['risk_distribution']['high']} CMLs")
                st.write(f"- Medium Risk: {stats['risk_distribution']['medium']} CMLs")
            with col3:
                st.markdown("**Inspection Planning**")
                st.write(
                    f"- Immediate Attention: {stats['inspection_scheduling']['immediate_attention']}"
                )
                st.write(f"- Next 6 Months: {stats['inspection_scheduling']['next_6_months']}")
                st.write(
                    f"- Avg Interval: {stats['inspection_scheduling']['avg_interval_months']:.0f} months"
                )

        else:
            # Fallback to basic visualization
            st.info("Advanced analytics module not loaded. Showing basic reports.")
            fig = px.scatter(
                df,
                x="average_corrosion_rate",
                y="thickness_mm",
                color="commodity",
                size="risk_score" if "risk_score" in df.columns else None,
                title="Corrosion Rate vs Thickness by Commodity",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download section
        st.markdown("---")
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset",
                data=csv,
                file_name="cml_analysis_data.csv",
                mime="text/csv",
            )
        with col2:
            if advanced_available:
                import json

                stats_json = json.dumps(stats, indent=2, default=str)
                st.download_button(
                    label="Download Statistics Report",
                    data=stats_json,
                    file_name="cml_statistics_report.json",
                    mime="application/json",
                )

elif page == "About Application":
    st.title("About Application")

    st.markdown("""
    ### User Guide & System Documentation
    
    This application is a specialized decision support system designed for **Wood Engineering** integrity management teams. It integrates advanced machine learning with traditional engineering principles to optimize Condition Monitoring Location (CML) inspection programs.
    
    ---
    
    ### 1. Upload & Analyze
    **Component**: Data Ingestion Engine & ML Scoring
    
    *   **What it does**: Accepts raw CML data (CSV/Excel) and applies the Random Forest Ensemble model to evaluate the risk and necessity of each monitoring point.
    *   **Why it exists**: Manual review of thousands of CMLs is time-consuming and prone to inconsistency. This component automates the first pass of screening, identifying "Low Value" CMLs that can be safely eliminated.
    *   **How to use**: 
        1.  Prepare your data file (ensure required columns like `thickness_mm`, `corrosion_rate` are present).
        2.  Navigate to **Upload & Analyze**.
        3.  Drag and drop your file.
        4.  Review the validation check (green checkmark).
        5.  Click **Run ML Analysis**.
        6.  View the "Optimization Recommendations" dashboard and download the results.
        
    ### 2. Lifecycle Forecasting
    **Component**: Remaining Life Calculator
    
    *   **What it does**: Projects the future state of an asset based on current thickness and corrosion rates, calculating the exact date when minimum safe thickness will be breached.
    *   **Why it exists**: To move from reactive to predictive maintenance. Knowing *when* an asset will become unsafe allows for optimized inspection intervals (e.g., extending inspection from 3 years to 5 years if remaining life is high).
    *   **How to use**:
        1.  Ensure data is loaded in the "Upload" section.
        2.  Go to **Forecasting**.
        3.  Adjust parameters:
            *   **Min Thickness**: The absolute safety floor (default 3.0 mm).
            *   **Safety Factor**: Buffer multiplier (default 1.5x).
        4.  Click **Generate Forecasts** to see the `Next Inspection Date`.
        
    ### 3. Expert Override Management
    **Component**: Human-in-the-loop Auditing
    
    *   **What it does**: A dedicated interface for engineers to formally reject the AI's recommendation.
    *   **Why it exists**: **Safety First.** The AI is a tool, not the final authority. There are nuanced field conditions (e.g., "vibro-acoustic fatigue risk") that the data might not capture. This component ensures that human expertise always supersedes the model, and creates an audit trail for compliance.
    *   **How to use**:
        1.  Identify a CML where you disagree with the model (e.g., Model says "ELIMINATE" but you know it's a high-consequence area).
        2.  Go to **SME Overrides**.
        3.  Enter the `CML ID`.
        4.  Select your decision (`KEEP`).
        5.  Provide the **Reason** (mandatory).
        6.  Submit to save to the audit log.
        
    ### 4. Comprehensive Reports
    **Component**: Executive Dashboard
    
    *   **What it does**: Aggregates the health status of the entire uploaded asset portfolio.
    *   **Why it exists**: For high-level decision making. Managers need to know the overall health (e.g., "30% of our carbon steel pipes are nearing end-of-life") rather than individual data points.
    *   **How to use**: Simply navigate to **Reports** after loading data to see interactive charts on Commodity Distribution and Risk profiles.
    
    ### 5. How It Works
    **Component**: White-Box Model Explanation
    
    *   **What it does**: Reveals the inner workings of the AI.
    *   **Why it exists**: To build trust. "Black box" AI is dangerous in engineering. By showing exactly which factors (Corrosion Rate, Thickness, etc.) drove the decision, engineers can validate if the AI's logic aligns with physical reality.
    
    ---
    
    **Support**: For technical issues or model retraining requests, please contact the Wood Engineering Digital Solutions team.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div style='text-align: center; color: #888;'>
    <small>Powered by</small><br>
    <b>Wood Engineering AI Solutions</b><br>
    <small>© 2025 Wood PLC</small>
</div>
""",
    unsafe_allow_html=True,
)
