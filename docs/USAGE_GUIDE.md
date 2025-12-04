# Usage Guide

## Wood AI CML Optimization System

Complete guide for using the CML (Condition Monitoring Location) optimization system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Using the API](#using-the-api)
5. [Dashboard](#dashboard)
6. [SME Overrides](#sme-overrides)
7. [Advanced Features](#advanced-features)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python ml/train_enhanced.py data/sample_cml_data.csv
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

### 4. Open the Dashboard

```bash
streamlit run streamlit_app.py
```

---

## Data Preparation

### Required Columns

Your CML data must include:

| Column | Type | Description | Example |
|--------|------|-------------|----------|
| `id_number` | string | Unique CML identifier | "CML-001" |
| `average_corrosion_rate` | float | Corrosion rate (mm/year) | 0.12 |
| `thickness_mm` | float | Current wall thickness (mm) | 9.5 |
| `commodity` | string | Commodity type | "Crude Oil" |
| `feature_type` | string | Piping feature type | "Pipe" |
| `cml_shape` | string | Monitoring location | "Both" |

### Optional Columns

- `last_inspection_date`: Date of last inspection (YYYY-MM-DD)
- `next_inspection_date`: Scheduled next inspection
- `remaining_life_years`: Calculated remaining life
- `risk_score`: Risk assessment score (0-100)
- `elimination_flag`: Target variable for training (0=Keep, 1=Eliminate)

### Data Format

Supported formats:
- CSV (`.csv`)
- Excel (`.xlsx`)

### Example CSV

```csv
id_number,average_corrosion_rate,thickness_mm,commodity,feature_type,cml_shape
CML-001,0.12,9.5,Crude Oil,Pipe,Both
CML-002,0.08,10.2,Natural Gas,Elbow,Internal
```

---

## Training the Model

### Basic Training

```bash
python ml/train_enhanced.py data/your_data.csv
```

### What Happens During Training:

1. **Data Loading**: Validates required columns
2. **Feature Engineering**: Creates derived features
3. **Preprocessing**: Scales numerical, encodes categorical
4. **Hyperparameter Tuning**: Grid search with cross-validation
5. **Evaluation**: Classification report, ROC-AUC, F1 score
6. **Model Saving**: Saves to `models/` directory

### Training Output

```
Loading data from data/sample_cml_data.csv
Loaded 200 records
Engineering features...
Training model...

Classification Report:
              precision    recall  f1-score   support

        Keep       0.92      0.95      0.93        31
   Eliminate       0.78      0.70      0.74         9

    accuracy                           0.90        40
   macro avg       0.85      0.82      0.83        40
weighted avg       0.89      0.90      0.89        40

ROC-AUC Score: 0.9234
F1 Score: 0.7400

Model saved to models/cml_elimination_model_20241204_103045.joblib
```

---

## Using the API

### Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload and Score Data

```bash
curl -X POST "http://localhost:8000/score-cml-data" \
  -F "file=@data/cml_data.csv"
```

### Forecast Remaining Life

```bash
curl -X POST "http://localhost:8000/forecast-remaining-life" \
  -F "file=@data/cml_data.csv"
```

### Add SME Override

```bash
curl -X POST "http://localhost:8000/sme-override" \
  -H "Content-Type: application/json" \
  -d '{
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "Critical monitoring point",
    "sme_name": "Dr. Smith"
  }'
```

---

## Dashboard

### Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

### Dashboard Pages

1. **Overview**: Dataset statistics and visualizations
2. **Upload & Score**: Upload data for predictions
3. **Forecasting**: Generate remaining life forecasts
4. **SME Overrides**: Manage expert overrides
5. **Reports**: Comprehensive analysis reports

---

## SME Overrides

Subject Matter Expert overrides allow manual decision tracking.

### Add Override (Dashboard)

1. Navigate to "SME Overrides" page
2. Expand "Add New Override"
3. Fill in:
   - CML ID
   - Decision (KEEP/ELIMINATE)
   - SME Name
   - Reason (detailed explanation)
4. Click "Add Override"

### Add Override (API)

```python
import requests

data = {
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "Critical monitoring for safety-critical area",
    "sme_name": "Dr. John Smith",
    "original_prediction": "ELIMINATE",
    "original_probability": 0.85
}

response = requests.post(
    "http://localhost:8000/sme-override",
    json=data
)
```

### View Overrides

Overrides are stored in `data/sme_overrides.json` and can be viewed in:
- Dashboard (SME Overrides page)
- API endpoint: `GET /sme-overrides/statistics`

---

## Advanced Features

### Custom Forecasting Parameters

```python
from app.forecasting import CMLForecaster

forecaster = CMLForecaster(
    minimum_thickness=2.5,  # Custom minimum
    safety_factor=2.0,       # More conservative
    min_inspection_interval_months=6,
    max_inspection_interval_months=60
)

forecast = forecaster.forecast_single_cml(
    id_number="CML-001",
    current_thickness=10.0,
    corrosion_rate=0.12
)
```

### Batch Processing

```python
import pandas as pd
from app.forecasting import CMLForecaster

df = pd.read_csv("data/large_dataset.csv")
forecaster = CMLForecaster()

forecast_df = forecaster.forecast_batch(df)
forecast_df.to_csv("output/forecasts.csv", index=False)
```

### Custom Model Training

```python
from ml.train_enhanced import EnhancedCMLModelTrainer

trainer = EnhancedCMLModelTrainer(
    data_path="data/custom_data.csv",
    model_output_dir="custom_models"
)

df = trainer.load_data()
df = trainer.engineer_features(df)
X, y, preprocessor = trainer.prepare_features(df)
metrics = trainer.train_model(X, y, preprocessor)
model_path = trainer.save_model(metrics)
```

---

## Troubleshooting

### Model Not Loading

```
Error: Model not loaded
```

**Solution**: Train a model first
```bash
python ml/train_enhanced.py data/sample_cml_data.csv
```

### Missing Columns Error

```
Missing required columns: ['thickness_mm']
```

**Solution**: Ensure your data has all required columns listed in [Data Preparation](#data-preparation)

### Port Already in Use

```
Error: Port 8000 already in use
```

**Solution**: Use different port
```bash
uvicorn app.main:app --port 8001
```

---

## Support

For questions or issues:
- Email: aaron@smarter.codes.ai
- GitHub Issues: [Create Issue](https://github.com/aaron-seq/wood-ai-cml-alo-ml-model/issues)