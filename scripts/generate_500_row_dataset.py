#!/usr/bin/env python3
"""
Generate 500-row synthetic CML dataset for training and testing.

Usage:
    python scripts/generate_500_row_dataset.py

Output:
    data/cml_sample_500.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RECORDS = 500

# Define possible values
commodities = [
    'Crude Oil', 'Natural Gas', 'Steam', 'Fuel Gas',
    'Potable Water', 'Process Water', 'Condensate',
    'Refined Product', 'Gas'
]

feature_types = [
    'Pipe', 'Elbow', 'Tee', 'Reducer', 'Flange',
    'Nozzle', 'Header', 'Bend', 'Weld'
]

cml_shapes = ['Internal', 'External', 'Both']

def generate_cml_data(num_records=500):
    """
    Generate synthetic CML data with realistic distributions.
    """
    data = []
    
    for i in range(1, num_records + 1):
        # Generate base properties
        commodity = random.choice(commodities)
        feature_type = random.choice(feature_types)
        cml_shape = random.choice(cml_shapes)
        
        # Corrosion rate varies by commodity (realistic patterns)
        if commodity in ['Crude Oil', 'Condensate']:
            avg_corrosion = np.random.uniform(0.05, 0.20)
        elif commodity in ['Natural Gas', 'Fuel Gas', 'Gas']:
            avg_corrosion = np.random.uniform(0.03, 0.15)
        elif commodity == 'Steam':
            avg_corrosion = np.random.uniform(0.08, 0.25)
        else:  # Water types
            avg_corrosion = np.random.uniform(0.02, 0.12)
        
        # Thickness varies by feature type
        if feature_type in ['Header', 'Pipe']:
            thickness = np.random.uniform(8.0, 12.0)
        elif feature_type in ['Flange', 'Reducer']:
            thickness = np.random.uniform(9.0, 11.5)
        else:
            thickness = np.random.uniform(6.0, 10.5)
        
        # Generate dates
        days_ago = random.randint(365, 1460)  # 1-4 years ago
        last_inspection = datetime.now() - timedelta(days=days_ago)
        
        inspection_interval = random.randint(365, 1095)  # 1-3 years
        next_inspection = last_inspection + timedelta(days=inspection_interval)
        days_to_next = (next_inspection - datetime.now()).days
        
        # Calculate remaining life
        min_thickness = 3.0
        if avg_corrosion > 0:
            remaining_life = (thickness - min_thickness) / avg_corrosion
        else:
            remaining_life = 999.9
        remaining_life = min(remaining_life, 300.0)  # Cap at 300 years
        
        # Calculate risk score (0-100)
        risk_score = 0
        if avg_corrosion > 0.15:
            risk_score += 40
        elif avg_corrosion > 0.10:
            risk_score += 25
        else:
            risk_score += 10
        
        if thickness < 8.0:
            risk_score += 30
        elif thickness < 9.5:
            risk_score += 15
        
        if remaining_life < 20:
            risk_score += 30
        elif remaining_life < 50:
            risk_score += 15
        
        risk_score = min(risk_score, 100)
        
        # Determine elimination flag (target variable)
        elimination_flag = 0
        
        # Low corrosion + thick walls = likely eliminate (redundant)
        if avg_corrosion < 0.08 and thickness > 9.5:
            elimination_flag = 1 if random.random() > 0.3 else 0
        
        # Very low corrosion = eliminate
        elif avg_corrosion < 0.05:
            elimination_flag = 1 if random.random() > 0.2 else 0
        
        # High risk = keep for monitoring
        elif risk_score > 70:
            elimination_flag = 0
        
        # Medium risk scenarios
        elif risk_score > 50:
            elimination_flag = 1 if random.random() > 0.8 else 0
        
        # Random edge cases
        else:
            elimination_flag = 1 if random.random() > 0.7 else 0
        
        # Build record
        record = {
            'id_number': f'CML-{i:03d}',
            'commodity': commodity,
            'feature_type': feature_type,
            'cml_shape': cml_shape,
            'average_corrosion_rate': round(avg_corrosion, 2),
            'thickness_mm': round(thickness, 1),
            'last_inspection_date': last_inspection.strftime('%Y-%m-%d'),
            'next_inspection_date': next_inspection.strftime('%Y-%m-%d'),
            'remaining_life_years': round(remaining_life, 1),
            'days_to_next_inspection': days_to_next,
            'risk_score': risk_score,
            'elimination_flag': elimination_flag
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    print('Generating 500-row synthetic CML dataset...')
    
    # Generate data
    df = generate_cml_data(NUM_RECORDS)
    
    # Statistics
    print(f'\nDataset statistics:')
    print(f'  Total records: {len(df)}')
    print(f'  Elimination rate: {df["elimination_flag"].mean():.1%}')
    print(f'  Unique commodities: {df["commodity"].nunique()}')
    print(f'  Unique feature types: {df["feature_type"].nunique()}')
    print(f'\nCorrosion rate range: {df["average_corrosion_rate"].min():.3f} - {df["average_corrosion_rate"].max():.3f}')
    print(f'Thickness range: {df["thickness_mm"].min():.1f} - {df["thickness_mm"].max():.1f} mm')
    print(f'Risk score range: {df["risk_score"].min()} - {df["risk_score"].max()}')
    
    # Save to CSV
    output_path = 'data/cml_sample_500.csv'
    df.to_csv(output_path, index=False)
    print(f'\nDataset saved to: {output_path}')
    print('\nNext steps:')
    print('  1. Train model: python ml/train_enhanced.py data/cml_sample_500.csv')
    print('  2. Start API: uvicorn app.main:app --reload')
    print('  3. Start dashboard: streamlit run streamlit_app.py')
