# Add this to the top of app/main.py after imports
import numpy as np
from datetime import datetime

def engineer_features(df):
    '''Engineer features matching the training pipeline.'''
    # Calculate corrosion-thickness ratio
    df['corrosion_thickness_ratio'] = df['average_corrosion_rate'] / df['thickness_mm']
    
    # Calculate remaining life
    min_thickness = 3.0  # Default minimum thickness
    df['remaining_life_years'] = (df['thickness_mm'] - min_thickness) / df['average_corrosion_rate']
    df['remaining_life_years'] = df['remaining_life_years'].clip(lower=0)
    
    # Calculate days since inspection
    if 'last_inspection_date' in df.columns:
        df['last_inspection_date'] = pd.to_datetime(df['last_inspection_date'], errors='coerce')
        df['days_since_inspection'] = (pd.Timestamp.now() - df['last_inspection_date']).dt.days
        df['days_since_inspection'] = df['days_since_inspection'].fillna(df['days_since_inspection'].median())
    else:
        df['days_since_inspection'] = 365  # Default 1 year
    
    # Use risk_score if available, otherwise calculate a basic one
    if 'risk_score' not in df.columns:
        df['risk_score'] = (
            df['average_corrosion_rate'] * 20 + 
            (10 - df['thickness_mm']) * 5
        ).clip(0, 100)
    
    return df
