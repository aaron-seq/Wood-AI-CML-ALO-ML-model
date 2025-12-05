"""Utility functions for CML data validation, processing, and reporting.

This module provides core utility functions for validating CML dataframes,
managing SME overrides, calculating inspection schedules, and generating reports.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for validation thresholds
MAX_CORROSION_RATE_MM_PER_YEAR = 5.0
MIN_CORROSION_RATE_MM_PER_YEAR = 0.0
MAX_THICKNESS_MM = 50.0
MIN_THICKNESS_MM = 0.0

# Constants for inspection scheduling
DEFAULT_MIN_THICKNESS_MM = 3.0
DEFAULT_SAFETY_FACTOR = 1.5
MIN_INSPECTION_INTERVAL_YEARS = 1
MAX_INSPECTION_INTERVAL_YEARS = 6
MAX_REMAINING_LIFE_YEARS = 50.0

# Risk level thresholds (years)
CRITICAL_RISK_THRESHOLD_YEARS = 2
HIGH_RISK_THRESHOLD_YEARS = 5
MEDIUM_RISK_THRESHOLD_YEARS = 10

# Required column names
REQUIRED_CML_COLUMNS = [
    'id_number',
    'average_corrosion_rate',
    'thickness_mm',
    'commodity',
    'feature_type',
    'cml_shape'
]


def validate_cml_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CML dataframe structure and data quality.
    
    Performs comprehensive validation including:
    - Checking for required columns
    - Identifying duplicate CML IDs
    - Validating numeric ranges for corrosion rate and thickness
    - Computing dataset statistics
    
    Args:
        df: DataFrame containing CML data to validate
        
    Returns:
        Dictionary containing:
        - valid (bool): Overall validation status
        - errors (List[str]): Critical validation errors
        - warnings (List[str]): Non-critical data quality warnings
        - stats (Dict): Computed statistics about the dataset
        
    Example:
        >>> df = pd.DataFrame({'id_number': ['CML-001'], ...})
        >>> result = validate_cml_dataframe(df)
        >>> if result['valid']:
        ...     print(f"Dataset has {result['stats']['total_records']} records")
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Check for empty dataframe
        if df.empty:
            validation_results['valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
        
        # Check required columns
        missing_cols = set(REQUIRED_CML_COLUMNS) - set(df.columns)
        if missing_cols:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {', '.join(sorted(missing_cols))}"
            )
            return validation_results
        
        # Check for duplicates
        if df['id_number'].duplicated().any():
            duplicates = df[df['id_number'].duplicated()]['id_number'].tolist()
            validation_results['warnings'].append(
                f"Found {len(duplicates)} duplicate CML IDs. "
                f"Examples: {', '.join(map(str, duplicates[:5]))}"
            )
        
        # Check for null values in required columns
        null_counts = df[REQUIRED_CML_COLUMNS].isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            validation_results['warnings'].append(
                f"Null values found: {null_cols.to_dict()}"
            )
        
        # Validate corrosion rate range
        invalid_corr = df[
            (df['average_corrosion_rate'] < MIN_CORROSION_RATE_MM_PER_YEAR) | 
            (df['average_corrosion_rate'] > MAX_CORROSION_RATE_MM_PER_YEAR)
        ]
        if len(invalid_corr) > 0:
            validation_results['warnings'].append(
                f"{len(invalid_corr)} records with unusual corrosion rates "
                f"(expected {MIN_CORROSION_RATE_MM_PER_YEAR}-{MAX_CORROSION_RATE_MM_PER_YEAR} mm/year). "
                f"Examples: {invalid_corr['id_number'].head(3).tolist()}"
            )
        
        # Validate thickness range
        invalid_thick = df[
            (df['thickness_mm'] <= MIN_THICKNESS_MM) | 
            (df['thickness_mm'] > MAX_THICKNESS_MM)
        ]
        if len(invalid_thick) > 0:
            validation_results['warnings'].append(
                f"{len(invalid_thick)} records with unusual thickness "
                f"(expected {MIN_THICKNESS_MM}-{MAX_THICKNESS_MM} mm). "
                f"Examples: {invalid_thick['id_number'].head(3).tolist()}"
            )
        
        # Compute statistics
        validation_results['stats'] = {
            'total_records': len(df),
            'unique_cmls': df['id_number'].nunique(),
            'avg_corrosion_rate': float(df['average_corrosion_rate'].mean()),
            'avg_thickness': float(df['thickness_mm'].mean()),
            'min_corrosion_rate': float(df['average_corrosion_rate'].min()),
            'max_corrosion_rate': float(df['average_corrosion_rate'].max()),
            'min_thickness': float(df['thickness_mm'].min()),
            'max_thickness': float(df['thickness_mm'].max()),
            'commodity_distribution': df['commodity'].value_counts().to_dict(),
            'feature_type_distribution': df['feature_type'].value_counts().to_dict()
        }
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        validation_results['valid'] = False
        validation_results['errors'].append(f"Validation exception: {str(e)}")
    
    return validation_results


def save_predictions_to_csv(
    predictions_df: pd.DataFrame,
    output_path: Path
) -> Path:
    """Save predictions to CSV file with error handling.
    
    Args:
        predictions_df: DataFrame containing predictions to save
        output_path: Path where CSV file should be saved
        
    Returns:
        Path object pointing to the saved file
        
    Raises:
        ValueError: If predictions_df is empty
        IOError: If file cannot be written
        
    Example:
        >>> df = pd.DataFrame({'id': [1], 'prediction': [0]})
        >>> path = save_predictions_to_csv(df, Path('output/predictions.csv'))
        >>> print(f"Saved to {path}")
    """
    if predictions_df.empty:
        raise ValueError("Cannot save empty predictions DataFrame")
    
    output_path = Path(output_path)
    
    try:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        return output_path
    except IOError as e:
        logger.error(f"Failed to save predictions to {output_path}: {e}")
        raise IOError(f"Cannot write to {output_path}: {e}")


def load_sme_overrides(override_file: Path) -> List[Dict[str, Any]]:
    """Load SME override decisions from JSON file.
    
    Args:
        override_file: Path to JSON file containing override records
        
    Returns:
        List of override dictionaries, empty list if file doesn't exist
        
    Raises:
        json.JSONDecodeError: If file contains invalid JSON
        
    Example:
        >>> overrides = load_sme_overrides(Path('data/sme_overrides.json'))
        >>> print(f"Loaded {len(overrides)} overrides")
    """
    if not override_file.exists():
        logger.info(f"Override file {override_file} does not exist, returning empty list")
        return []
    
    try:
        with open(override_file, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
            logger.info(f"Loaded {len(overrides)} override records from {override_file}")
            return overrides
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in override file {override_file}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to read override file {override_file}: {e}")
        return []


def save_sme_override(
    override_data: Dict[str, Any],
    override_file: Path
) -> None:
    """Save new SME override decision to JSON file.
    
    Args:
        override_data: Dictionary containing override information
        override_file: Path where overrides should be saved
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> override = {
        ...     'id_number': 'CML-001',
        ...     'decision': 'KEEP',
        ...     'reason': 'Critical location'
        ... }
        >>> save_sme_override(override, Path('data/sme_overrides.json'))
    """
    try:
        overrides = load_sme_overrides(override_file)
        
        # Convert datetime to string if present
        if 'override_date' in override_data:
            if isinstance(override_data['override_date'], datetime):
                override_data['override_date'] = override_data['override_date'].isoformat()
        
        overrides.append(override_data)
        
        override_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(override_file, 'w', encoding='utf-8') as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Override saved for CML {override_data.get('id_number', 'unknown')}")
    
    except IOError as e:
        logger.error(f"Failed to save override to {override_file}: {e}")
        raise IOError(f"Cannot write override to {override_file}: {e}")


def calculate_inspection_schedule(
    corrosion_rate: float,
    thickness: float,
    min_thickness: float = DEFAULT_MIN_THICKNESS_MM,
    safety_factor: float = DEFAULT_SAFETY_FACTOR
) -> Dict[str, Any]:
    """Calculate recommended inspection schedule based on corrosion parameters.
    
    Uses remaining life calculation and safety factors to determine optimal
    inspection intervals and risk classifications.
    
    Args:
        corrosion_rate: Average corrosion rate in mm/year
        thickness: Current wall thickness in mm
        min_thickness: Minimum allowable thickness in mm (default: 3.0)
        safety_factor: Safety factor for inspection intervals (default: 1.5)
        
    Returns:
        Dictionary containing:
        - remaining_life_years: Estimated years until minimum thickness
        - inspection_interval_months: Recommended months between inspections
        - next_inspection_date: Calculated date for next inspection
        - risk_level: Risk classification (CRITICAL/HIGH/MEDIUM/LOW)
        - estimated_thickness_at_next_inspection: Projected thickness at next inspection
        
    Raises:
        ValueError: If input parameters are invalid
        
    Example:
        >>> schedule = calculate_inspection_schedule(
        ...     corrosion_rate=0.5,
        ...     thickness=10.0
        ... )
        >>> print(f"Risk level: {schedule['risk_level']}")
    """
    # Input validation
    if thickness <= 0:
        raise ValueError(f"Invalid thickness: {thickness}. Must be positive.")
    
    if corrosion_rate < 0:
        raise ValueError(f"Invalid corrosion rate: {corrosion_rate}. Must be non-negative.")
    
    if safety_factor < 1.0:
        raise ValueError(f"Invalid safety factor: {safety_factor}. Must be >= 1.0.")
    
    # Calculate remaining life
    available_thickness = thickness - min_thickness
    
    if available_thickness <= 0:
        remaining_life_years = 0.0
    elif corrosion_rate <= 0:
        remaining_life_years = MAX_REMAINING_LIFE_YEARS
    else:
        remaining_life_years = available_thickness / corrosion_rate
        remaining_life_years = min(remaining_life_years, MAX_REMAINING_LIFE_YEARS)
    
    # Calculate inspection interval with safety factor
    if remaining_life_years == 0:
        inspection_interval_years = 0.5  # Immediate attention needed
    else:
        inspection_interval_years = remaining_life_years / safety_factor
        inspection_interval_years = max(
            MIN_INSPECTION_INTERVAL_YEARS,
            min(inspection_interval_years, MAX_INSPECTION_INTERVAL_YEARS)
        )
    
    # Calculate next inspection date
    days_to_inspection = int(inspection_interval_years * 365)
    next_inspection = datetime.now() + timedelta(days=days_to_inspection)
    
    # Determine risk level
    if remaining_life_years < CRITICAL_RISK_THRESHOLD_YEARS:
        risk_level = "CRITICAL"
    elif remaining_life_years < HIGH_RISK_THRESHOLD_YEARS:
        risk_level = "HIGH"
    elif remaining_life_years < MEDIUM_RISK_THRESHOLD_YEARS:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Calculate estimated thickness at next inspection
    thickness_loss = corrosion_rate * inspection_interval_years
    estimated_thickness = thickness - thickness_loss
    
    return {
        'remaining_life_years': round(remaining_life_years, 1),
        'inspection_interval_months': int(inspection_interval_years * 12),
        'next_inspection_date': next_inspection.date(),
        'risk_level': risk_level,
        'estimated_thickness_at_next_inspection': round(estimated_thickness, 2)
    }


def generate_elimination_report(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive elimination analysis report.
    
    Analyzes prediction results and generates statistical summaries,
    distributions, and identification of key cases.
    
    Args:
        predictions_df: DataFrame containing prediction results
        
    Returns:
        Dictionary containing:
        - summary: Overall statistics (total, eliminations, keep, rate)
        - confidence_distribution: Distribution of confidence levels
        - elimination_by_commodity: Elimination counts by commodity type
        - elimination_by_feature: Elimination counts by feature type
        - top_elimination_candidates: Highest probability elimination cases
        - marginal_cases: Predictions close to decision boundary
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> report = generate_elimination_report(predictions_df)
        >>> print(f"Elimination rate: {report['summary']['elimination_rate']}%")
    """
    if predictions_df.empty:
        return {'error': 'DataFrame is empty'}
    
    if 'predicted_elimination' not in predictions_df.columns:
        raise ValueError('Column "predicted_elimination" not found in DataFrame')
    
    total_cmls = len(predictions_df)
    eliminations = predictions_df[predictions_df['predicted_elimination'] == 1]
    keep_cmls = predictions_df[predictions_df['predicted_elimination'] == 0]
    
    report = {
        'summary': {
            'total_cmls': total_cmls,
            'recommended_eliminations': len(eliminations),
            'recommended_keep': len(keep_cmls),
            'elimination_rate': round(
                len(eliminations) / total_cmls * 100, 1
            ) if total_cmls > 0 else 0
        }
    }
    
    # Optional fields with safe extraction
    if 'confidence_level' in predictions_df.columns:
        report['confidence_distribution'] = (
            predictions_df['confidence_level'].value_counts().to_dict()
        )
    
    if 'commodity' in eliminations.columns and len(eliminations) > 0:
        report['elimination_by_commodity'] = (
            eliminations.groupby('commodity').size().to_dict()
        )
    
    if 'feature_type' in eliminations.columns and len(eliminations) > 0:
        report['elimination_by_feature'] = (
            eliminations.groupby('feature_type').size().to_dict()
        )
    
    # Top elimination candidates
    if 'elimination_probability' in eliminations.columns and len(eliminations) > 0:
        top_candidates_cols = [
            'id_number', 'elimination_probability', 'average_corrosion_rate', 'thickness_mm'
        ]
        available_cols = [col for col in top_candidates_cols if col in eliminations.columns]
        
        if available_cols:
            report['top_elimination_candidates'] = (
                eliminations.nlargest(20, 'elimination_probability')[available_cols]
                .to_dict('records')
            )
    
    # Marginal cases (close to decision boundary)
    if 'elimination_probability' in predictions_df.columns:
        marginal_cols = ['id_number', 'elimination_probability']
        if 'recommendation' in predictions_df.columns:
            marginal_cols.append('recommendation')
        
        available_marginal_cols = [
            col for col in marginal_cols if col in predictions_df.columns
        ]
        
        if available_marginal_cols:
            marginal_cases = predictions_df[
                (predictions_df['elimination_probability'] > 0.4) & 
                (predictions_df['elimination_probability'] < 0.6)
            ]
            report['marginal_cases'] = (
                marginal_cases[available_marginal_cols].to_dict('records')
            )
    
    return report
