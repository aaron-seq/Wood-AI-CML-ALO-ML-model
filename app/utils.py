"""Utility functions."""

import pandas as pd
from typing import Dict, List
from pathlib import Path
from datetime import datetime, timedelta
import json


def validate_cml_dataframe(df: pd.DataFrame) -> Dict[str, any]:
    """Validate CML dataframe structure and data quality."""
    required_columns = [
        'id_number', 'average_corrosion_rate', 'thickness_mm',
        'commodity', 'feature_type', 'cml_shape'
    ]
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        return validation_results
    
    # Check for duplicates
    if df['id_number'].duplicated().any():
        duplicates = df[df['id_number'].duplicated()]['id_number'].tolist()
        validation_results['warnings'].append(f"Duplicate CML IDs found: {duplicates[:5]}")
    
    # Check corrosion rate range
    invalid_corr = df[
        (df['average_corrosion_rate'] < 0) | 
        (df['average_corrosion_rate'] > 5.0)
    ]
    if len(invalid_corr) > 0:
        validation_results['warnings'].append(
            f"{len(invalid_corr)} records with unusual corrosion rates (expected 0-5 mm/year)"
        )
    
    # Check thickness range
    invalid_thick = df[(df['thickness_mm'] <= 0) | (df['thickness_mm'] > 50)]
    if len(invalid_thick) > 0:
        validation_results['warnings'].append(
            f"{len(invalid_thick)} records with unusual thickness (expected 0-50 mm)"
        )
    
    # Statistics
    validation_results['stats'] = {
        'total_records': len(df),
        'unique_cmls': df['id_number'].nunique(),
        'avg_corrosion_rate': float(df['average_corrosion_rate'].mean()),
        'avg_thickness': float(df['thickness_mm'].mean()),
        'commodity_distribution': df['commodity'].value_counts().to_dict(),
        'feature_type_distribution': df['feature_type'].value_counts().to_dict()
    }
    
    return validation_results


def save_predictions_to_csv(predictions_df: pd.DataFrame, output_path: Path) -> Path:
    """Save predictions to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    predictions_df.to_csv(output_path, index=False)
    
    return output_path


def load_sme_overrides(override_file: Path) -> List[Dict]:
    """Load SME override decisions from JSON file."""
    if not override_file.exists():
        return []
    
    with open(override_file, 'r') as f:
        return json.load(f)


def save_sme_override(override_data: Dict, override_file: Path):
    """Save new SME override decision."""
    overrides = load_sme_overrides(override_file)
    
    # Convert datetime to string if needed
    if 'override_date' in override_data and isinstance(override_data['override_date'], datetime):
        override_data['override_date'] = override_data['override_date'].isoformat()
    
    overrides.append(override_data)
    
    override_file.parent.mkdir(exist_ok=True, parents=True)
    with open(override_file, 'w') as f:
        json.dump(overrides, f, indent=2)


def calculate_inspection_schedule(
    corrosion_rate: float,
    thickness: float,
    min_thickness: float = 3.0,
    safety_factor: float = 1.5
) -> Dict:
    """Calculate recommended inspection schedule."""
    # Calculate remaining life
    available_thickness = thickness - min_thickness
    if corrosion_rate <= 0:
        remaining_life_years = 50.0
    else:
        remaining_life_years = available_thickness / corrosion_rate
    
    # Apply safety factor for inspection frequency
    inspection_interval_years = remaining_life_years / safety_factor
    
    # Cap at reasonable limits
    inspection_interval_years = max(1, min(inspection_interval_years, 6))
    
    # Calculate next inspection date
    next_inspection = datetime.now() + timedelta(days=int(inspection_interval_years * 365))
    
    # Risk classification
    if remaining_life_years < 2:
        risk_level = "CRITICAL"
    elif remaining_life_years < 5:
        risk_level = "HIGH"
    elif remaining_life_years < 10:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'remaining_life_years': round(remaining_life_years, 1),
        'inspection_interval_months': int(inspection_interval_years * 12),
        'next_inspection_date': next_inspection.date(),
        'risk_level': risk_level,
        'estimated_thickness_at_next_inspection': round(
            thickness - (corrosion_rate * inspection_interval_years), 2
        )
    }


def generate_elimination_report(predictions_df: pd.DataFrame) -> Dict:
    """Generate comprehensive elimination report."""
    total_cmls = len(predictions_df)
    
    if 'predicted_elimination' not in predictions_df.columns:
        return {'error': 'Predictions not found in dataframe'}
    
    eliminations = predictions_df[predictions_df['predicted_elimination'] == 1]
    keep_cmls = predictions_df[predictions_df['predicted_elimination'] == 0]
    
    report = {
        'summary': {
            'total_cmls': total_cmls,
            'recommended_eliminations': len(eliminations),
            'recommended_keep': len(keep_cmls),
            'elimination_rate': round(len(eliminations) / total_cmls * 100, 1) if total_cmls > 0 else 0
        },
        'confidence_distribution': predictions_df['confidence_level'].value_counts().to_dict() if 'confidence_level' in predictions_df.columns else {},
        'elimination_by_commodity': eliminations.groupby('commodity').size().to_dict() if 'commodity' in eliminations.columns else {},
        'elimination_by_feature': eliminations.groupby('feature_type').size().to_dict() if 'feature_type' in eliminations.columns else {},
        'top_elimination_candidates': eliminations.nlargest(20, 'elimination_probability')[[
            'id_number', 'elimination_probability', 'confidence_level', 'average_corrosion_rate', 'thickness_mm'
        ]].to_dict('records') if 'id_number' in eliminations.columns and len(eliminations) > 0 else [],
        'marginal_cases': predictions_df[
            (predictions_df['elimination_probability'] > 0.4) & 
            (predictions_df['elimination_probability'] < 0.6)
        ][[
            'id_number', 'elimination_probability', 'recommendation'
        ]].to_dict('records') if 'id_number' in predictions_df.columns else []
    }
    
    return report