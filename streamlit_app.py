"""Streamlit Dashboard for CML Optimization.

This module provides an interactive web dashboard for the Wood AI CML Optimization system.
It enables users to upload CML data, generate predictions, forecast remaining life,
manage SME overrides, and generate comprehensive reports.
"""

from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use proper imports instead of sys.path manipulation
try:
    from app.forecasting import CMLForecaster
    from app.sme_override import SMEOverrideManager
    from app.utils import validate_cml_dataframe
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("Failed to import required modules. Please check your installation.")
    st.stop()

# Constants
SAMPLE_DATA_PATH = Path("data/sample_cml_data.csv")
DEFAULT_MINIMUM_THICKNESS = 3.0
DEFAULT_SAFETY_FACTOR = 1.5
MAX_PREVIEW_ROWS = 10
MAX_RESULTS_DISPLAY = 100

# Page configuration
st.set_page_config(
    page_title="Wood AI - CML Optimization",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Wood AI CML Optimization Dashboard")
st.markdown("**Condition Monitoring Location (CML) Elimination & Optimization System**")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Upload & Score",
        "Forecasting",
        "SME Overrides",
        "Reports"
    ]
)


@st.cache_resource
def get_forecaster() -> CMLForecaster:
    """Initialize and cache the CML forecaster.
    
    Returns:
        CMLForecaster: Cached forecaster instance
    """
    try:
        return CMLForecaster()
    except Exception as e:
        logger.error(f"Failed to initialize forecaster: {e}")
        st.error("Failed to initialize forecasting module. Please check your configuration.")
        raise


@st.cache_resource
def get_sme_manager() -> SMEOverrideManager:
    """Initialize and cache the SME override manager.
    
    Returns:
        SMEOverrideManager: Cached SME manager instance
    """
    try:
        return SMEOverrideManager()
    except Exception as e:
        logger.error(f"Failed to initialize SME manager: {e}")
        st.error("Failed to initialize SME override manager. Please check your configuration.")
        raise


@st.cache_data
def load_sample_data() -> Optional[pd.DataFrame]:
    """Load sample CML data from CSV file.
    
    Returns:
        Optional[pd.DataFrame]: Loaded dataframe or None if file doesn't exist
    """
    try:
        if SAMPLE_DATA_PATH.exists():
            return pd.read_csv(SAMPLE_DATA_PATH)
        else:
            logger.warning(f"Sample data file not found at {SAMPLE_DATA_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return None


def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Read uploaded CSV or Excel file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Optional[pd.DataFrame]: Parsed dataframe or None on error
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return None
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid data file.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing file: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        logger.error(f"File read error for {uploaded_file.name}: {e}")
        return None


# Initialize managers
try:
    forecaster = get_forecaster()
    sme_manager = get_sme_manager()
except Exception as e:
    logger.critical(f"Failed to initialize application: {e}")
    st.stop()


# PAGE: Overview
if page == "Overview":
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Status", "Online", "Healthy")
    with col2:
        st.metric("Model Status", "Loaded", "v1.0")
    with col3:
        st.metric("Total CMLs", "200", "Sample Dataset")
    
    st.markdown("---")
    
    df = load_sample_data()
    
    if df is not None:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Commodity distribution
            commodity_counts = df['commodity'].value_counts()
            fig = px.pie(
                values=commodity_counts.values,
                names=commodity_counts.index,
                title="CMLs by Commodity"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature type distribution
            feature_counts = df['feature_type'].value_counts()
            fig = px.bar(
                x=feature_counts.index,
                y=feature_counts.values,
                title="CMLs by Feature Type",
                labels={'x': 'Feature Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrosion rate distribution
        st.subheader("Corrosion Rate Analysis")
        fig = px.histogram(
            df,
            x='average_corrosion_rate',
            nbins=30,
            title="Distribution of Corrosion Rates",
            labels={'average_corrosion_rate': 'Corrosion Rate (mm/year)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Elimination statistics
        if 'elimination_flag' in df.columns:
            st.subheader("Elimination Statistics")
            elimination_count = df['elimination_flag'].sum()
            total_count = len(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total CMLs", total_count)
            with col2:
                st.metric("Recommended Eliminations", elimination_count)
            with col3:
                elimination_rate = (elimination_count / total_count * 100)
                st.metric("Elimination Rate", f"{elimination_rate:.1f}%")
    else:
        st.info("No sample data found. Upload data to view statistics.")


# PAGE: Upload & Score
elif page == "Upload & Score":
    st.header("Upload and Score CML Data")
    
    st.markdown("""
    Upload your CML data file (CSV or Excel) to get elimination recommendations.
    
    **Required Columns:**
    - `id_number`: Unique CML identifier
    - `average_corrosion_rate`: Corrosion rate (mm/year)
    - `thickness_mm`: Current wall thickness (mm)
    - `commodity`: Commodity type
    - `feature_type`: Piping feature type
    - `cml_shape`: Monitoring location (Internal/External/Both)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with CML data"
    )
    
    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file)
        
        if df is not None:
            st.success(f"Successfully loaded {len(df)} records from {uploaded_file.name}")
            
            # Validate data
            try:
                validation = validate_cml_dataframe(df)
                
                if validation['valid']:
                    st.success("Data validation passed")
                else:
                    st.error("Data validation errors found:")
                    for error in validation['errors']:
                        st.error(f"- {error}")
                
                if validation['warnings']:
                    st.warning("Validation warnings:")
                    for warning in validation['warnings']:
                        st.warning(f"- {warning}")
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df.head(MAX_PREVIEW_ROWS))
                
                # Display statistics
                st.subheader("Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                
                stats = validation['stats']
                with col1:
                    st.metric("Total Records", stats['total_records'])
                with col2:
                    st.metric("Avg Corrosion Rate", f"{stats['avg_corrosion_rate']:.3f} mm/yr")
                with col3:
                    st.metric("Avg Thickness", f"{stats['avg_thickness']:.2f} mm")
                
                # Score button
                if st.button("Score Data", type="primary"):
                    st.info("Scoring functionality requires running API server. See documentation for setup.")
            
            except Exception as e:
                st.error(f"Validation error: {e}")
                logger.error(f"Validation error: {e}")


# PAGE: Forecasting
elif page == "Forecasting":
    st.header("Remaining Life Forecasting")
    
    st.markdown("""
    Forecast remaining life and recommended inspection schedules for CMLs.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CML data for forecasting",
        type=['csv', 'xlsx', 'xls'],
        key="forecast_upload"
    )
    
    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file)
        
        if df is not None:
            st.success(f"Successfully loaded {len(df)} records")
            
            # Forecasting parameters
            col1, col2 = st.columns(2)
            with col1:
                min_thickness = st.number_input(
                    "Minimum Required Thickness (mm)",
                    min_value=1.0,
                    max_value=10.0,
                    value=DEFAULT_MINIMUM_THICKNESS,
                    step=0.5
                )
            with col2:
                safety_factor = st.number_input(
                    "Safety Factor",
                    min_value=1.0,
                    max_value=3.0,
                    value=DEFAULT_SAFETY_FACTOR,
                    step=0.1
                )
            
            if st.button("Generate Forecasts", type="primary"):
                try:
                    with st.spinner("Generating forecasts..."):
                        # Create forecaster with custom parameters
                        custom_forecaster = CMLForecaster(
                            minimum_thickness=min_thickness,
                            safety_factor=safety_factor
                        )
                        
                        # Generate forecasts
                        forecast_df = custom_forecaster.forecast_batch(df)
                        
                        st.success("Forecasts generated successfully")
                        
                        # Summary
                        st.subheader("Forecast Summary")
                        summary = custom_forecaster.generate_forecast_summary(df)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Remaining Life", f"{summary['avg_remaining_life_years']:.1f} yrs")
                        with col2:
                            st.metric("Critical CMLs", summary['critical_cmls'])
                        with col3:
                            st.metric("High Risk CMLs", summary['high_risk_cmls'])
                        with col4:
                            st.metric("Inspections Next 12mo", summary['inspections_needed_next_12_months'])
                        
                        # Risk distribution
                        st.subheader("Risk Distribution")
                        risk_dist = pd.DataFrame(
                            list(summary['risk_distribution'].items()),
                            columns=['Risk Level', 'Count']
                        )
                        fig = px.bar(
                            risk_dist,
                            x='Risk Level',
                            y='Count',
                            color='Risk Level',
                            color_discrete_map={
                                'CRITICAL': 'red',
                                'HIGH': 'orange',
                                'MEDIUM': 'yellow',
                                'LOW': 'green'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results
                        st.subheader("Detailed Forecast Results")
                        display_cols = [
                            'id_number',
                            'remaining_life_years',
                            'next_inspection_date',
                            'recommended_inspection_frequency_months',
                            'risk_level'
                        ]
                        available_cols = [col for col in display_cols if col in forecast_df.columns]
                        st.dataframe(forecast_df[available_cols])
                        
                        # Download button
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast Results",
                            data=csv,
                            file_name="cml_forecasts.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
                    logger.error(f"Forecasting error: {e}")


# PAGE: SME Overrides
elif page == "SME Overrides":
    st.header("Subject Matter Expert (SME) Overrides")
    
    st.markdown("""
    Track manual decision overrides by Subject Matter Experts.
    """)
    
    # Add new override
    with st.expander("Add New Override", expanded=False):
        with st.form("override_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                cml_id = st.text_input("CML ID", placeholder="CML-001")
                decision = st.selectbox("Decision", ["KEEP", "ELIMINATE"])
            
            with col2:
                sme_name = st.text_input("SME Name", placeholder="Dr. John Smith")
            
            reason = st.text_area(
                "Reason for Override",
                placeholder="Explain the rationale for this decision...",
                height=100
            )
            
            submitted = st.form_submit_button("Add Override")
            
            if submitted:
                if cml_id and decision and sme_name and reason:
                    try:
                        sme_manager.add_override(
                            id_number=cml_id,
                            sme_decision=decision,
                            reason=reason,
                            sme_name=sme_name
                        )
                        st.success(f"Override successfully added for {cml_id}")
                    except Exception as e:
                        st.error(f"Error adding override: {e}")
                        logger.error(f"Override addition error: {e}")
                else:
                    st.error("Please fill in all fields")
    
    # Display statistics
    st.subheader("Override Statistics")
    try:
        stats = sme_manager.get_override_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Overrides", stats['total_overrides'])
        with col2:
            st.metric("Keep Decisions", stats['keep_overrides'])
        with col3:
            st.metric("Eliminate Decisions", stats['eliminate_overrides'])
    except Exception as e:
        st.error(f"Error loading statistics: {e}")
        logger.error(f"Statistics error: {e}")
    
    # Display all overrides
    st.subheader("All Overrides")
    try:
        overrides = sme_manager.get_all_overrides()
        
        if overrides:
            df_overrides = pd.DataFrame(overrides)
            st.dataframe(df_overrides, use_container_width=True)
            
            # Download button
            csv = df_overrides.to_csv(index=False)
            st.download_button(
                label="Download Overrides",
                data=csv,
                file_name="sme_overrides.csv",
                mime="text/csv"
            )
        else:
            st.info("No overrides recorded yet.")
    except Exception as e:
        st.error(f"Error loading overrides: {e}")
        logger.error(f"Override loading error: {e}")


# PAGE: Reports
elif page == "Reports":
    st.header("Comprehensive Reports")
    
    st.markdown("""
    Generate comprehensive CML optimization reports.
    """)
    
    # Sample data option
    use_sample = st.checkbox("Use sample data", value=True)
    
    df = None
    if use_sample:
        df = load_sample_data()
        if df is not None:
            st.success(f"Successfully loaded {len(df)} sample records")
        else:
            st.error("Sample data not found")
    else:
        uploaded_file = st.file_uploader(
            "Upload CML data for reporting",
            type=['csv', 'xlsx', 'xls'],
            key="report_upload"
        )
        
        if uploaded_file:
            df = read_uploaded_file(uploaded_file)
            if df is not None:
                st.success(f"Successfully loaded {len(df)} records")
    
    if df is not None and st.button("Generate Report", type="primary"):
        try:
            st.subheader("CML Optimization Report")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total CMLs", len(df))
            with col2:
                avg_corr = df['average_corrosion_rate'].mean()
                st.metric("Avg Corrosion Rate", f"{avg_corr:.3f} mm/yr")
            with col3:
                avg_thick = df['thickness_mm'].mean()
                st.metric("Avg Thickness", f"{avg_thick:.2f} mm")
            
            # Commodity breakdown
            st.subheader("Commodity Analysis")
            commodity_stats = df.groupby('commodity').agg({
                'id_number': 'count',
                'average_corrosion_rate': 'mean',
                'thickness_mm': 'mean'
            }).round(3)
            commodity_stats.columns = ['Count', 'Avg Corrosion Rate', 'Avg Thickness']
            st.dataframe(commodity_stats)
            
            # Visualization
            fig = px.scatter(
                df,
                x='average_corrosion_rate',
                y='thickness_mm',
                color='commodity',
                size='risk_score' if 'risk_score' in df.columns else None,
                hover_data=['id_number'],
                title="Corrosion Rate vs Thickness by Commodity",
                labels={
                    'average_corrosion_rate': 'Corrosion Rate (mm/year)',
                    'thickness_mm': 'Thickness (mm)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Report generated successfully")
        
        except Exception as e:
            st.error(f"Error generating report: {e}")
            logger.error(f"Report generation error: {e}")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Wood AI CML Optimization**  
Version 1.0.0  
Copyright 2024 Smarter.Codes.AI
""")
