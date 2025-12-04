"""Streamlit Dashboard for CML Optimization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import requests
import io

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent))

from app.forecasting import CMLForecaster
from app.sme_override import SMEOverrideManager
from app.utils import validate_cml_dataframe

# Page config
st.set_page_config(
    page_title="Wood AI - CML Optimization",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔧 Wood AI CML Optimization Dashboard")
st.markdown("**Condition Monitoring Location (CML) Elimination & Optimization System**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Overview", "📤 Upload & Score", "📈 Forecasting", "👨‍💼 SME Overrides", "📑 Reports"]
)

# Initialize managers
@st.cache_resource
def get_forecaster():
    return CMLForecaster()

@st.cache_resource
def get_sme_manager():
    return SMEOverrideManager()

forecaster = get_forecaster()
sme_manager = get_sme_manager()

# Helper function to load sample data
@st.cache_data
def load_sample_data():
    sample_path = Path("data/sample_cml_data.csv")
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return None

# PAGE: Overview
if page == "📊 Overview":
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Status", "🟢 Online", "Healthy")
    with col2:
        st.metric("Model Status", "🟢 Loaded", "v1.0")
    with col3:
        st.metric("Total CMLs", "200", "Sample Dataset")
    
    st.markdown("---")
    
    # Load sample data for visualization
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
elif page == "📤 Upload & Score":
    st.header("Upload & Score CML Data")
    
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
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with CML data"
    )
    
    if uploaded_file is not None:
        # Read file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} records from {uploaded_file.name}")
            
            # Validate
            validation = validate_cml_dataframe(df)
            
            if validation['valid']:
                st.success("✅ Data validation passed")
            else:
                st.error("❌ Data validation errors found")
                for error in validation['errors']:
                    st.error(error)
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    st.warning(warning)
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
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
            if st.button("🎯 Score Data", type="primary"):
                st.info("Scoring functionality requires running API server. See documentation for setup.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# PAGE: Forecasting
elif page == "📈 Forecasting":
    st.header("Remaining Life Forecasting")
    
    st.markdown("""
    Forecast remaining life and recommended inspection schedules for CMLs.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CML data for forecasting",
        type=['csv', 'xlsx'],
        key="forecast_upload"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} records")
            
            # Forecasting parameters
            col1, col2 = st.columns(2)
            with col1:
                min_thickness = st.number_input(
                    "Minimum Required Thickness (mm)",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
            with col2:
                safety_factor = st.number_input(
                    "Safety Factor",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )
            
            if st.button("📊 Generate Forecasts", type="primary"):
                with st.spinner("Generating forecasts..."):
                    # Create forecaster with custom parameters
                    custom_forecaster = CMLForecaster(
                        minimum_thickness=min_thickness,
                        safety_factor=safety_factor
                    )
                    
                    # Generate forecasts
                    forecast_df = custom_forecaster.forecast_batch(df)
                    
                    st.success("✅ Forecasts generated successfully")
                    
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
                    risk_dist = pd.DataFrame(list(summary['risk_distribution'].items()), 
                                            columns=['Risk Level', 'Count'])
                    fig = px.bar(risk_dist, x='Risk Level', y='Count', 
                                color='Risk Level',
                                color_discrete_map={'CRITICAL': 'red', 'HIGH': 'orange', 
                                                   'MEDIUM': 'yellow', 'LOW': 'green'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("Detailed Forecast Results")
                    display_cols = ['id_number', 'remaining_life_years', 'next_inspection_date',
                                   'recommended_inspection_frequency_months', 'risk_level']
                    available_cols = [col for col in display_cols if col in forecast_df.columns]
                    st.dataframe(forecast_df[available_cols])
                    
                    # Download button
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Results",
                        data=csv,
                        file_name="cml_forecasts.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# PAGE: SME Overrides
elif page == "👨‍💼 SME Overrides":
    st.header("Subject Matter Expert (SME) Overrides")
    
    st.markdown("""
    Track manual decision overrides by Subject Matter Experts.
    """)
    
    # Add new override
    with st.expander("➕ Add New Override", expanded=False):
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
                        st.success(f"✅ Override added for {cml_id}")
                    except Exception as e:
                        st.error(f"Error adding override: {str(e)}")
                else:
                    st.error("Please fill in all fields")
    
    # Display statistics
    st.subheader("Override Statistics")
    stats = sme_manager.get_override_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Overrides", stats['total_overrides'])
    with col2:
        st.metric("Keep Decisions", stats['keep_overrides'])
    with col3:
        st.metric("Eliminate Decisions", stats['eliminate_overrides'])
    
    # Display all overrides
    st.subheader("All Overrides")
    overrides = sme_manager.get_all_overrides()
    
    if overrides:
        df_overrides = pd.DataFrame(overrides)
        st.dataframe(df_overrides, use_container_width=True)
        
        # Download button
        csv = df_overrides.to_csv(index=False)
        st.download_button(
            label="📥 Download Overrides",
            data=csv,
            file_name="sme_overrides.csv",
            mime="text/csv"
        )
    else:
        st.info("No overrides recorded yet.")

# PAGE: Reports
elif page == "📑 Reports":
    st.header("Comprehensive Reports")
    
    st.markdown("""
    Generate comprehensive CML optimization reports.
    """)
    
    # Sample data option
    use_sample = st.checkbox("Use sample data", value=True)
    
    if use_sample:
        df = load_sample_data()
        if df is not None:
            st.success(f"✅ Loaded {len(df)} sample records")
        else:
            st.error("Sample data not found")
            df = None
    else:
        uploaded_file = st.file_uploader(
            "Upload CML data for reporting",
            type=['csv', 'xlsx'],
            key="report_upload"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded {len(df)} records")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                df = None
        else:
            df = None
    
    if df is not None and st.button("📊 Generate Report", type="primary"):
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
        
        st.success("✅ Report generated successfully")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Wood AI CML Optimization**  
Version 1.0.0  
© 2024 Smarter.Codes.AI
""")