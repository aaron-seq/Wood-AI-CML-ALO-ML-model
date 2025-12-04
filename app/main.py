from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from pathlib import Path
import joblib
from typing import List, Any

app = FastAPI(title="Wood AI CML ALO API", version="0.2.0")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cml_elimination_model.joblib"

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

REQUIRED_COLUMNS = [
    "id_number",
    "average_corrosion_rate",
    "thickness_mm",
    "commodity",
    "feature_type",
    "cml_shape",
    "last_inspection_date",
    "next_inspection_date",
]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/upload-cml-data")
async def upload_cml_data(file: UploadFile = File(...)):
    try:
        df = pd.read_excel(file.file)
    except Exception:
        file.file.seek(0)
        df = pd.read_csv(file.file)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/score-cml-data")
async def score_cml_data(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train it first.")

    try:
        df = pd.read_excel(file.file)
    except Exception:
        file.file.seek(0)
        df = pd.read_csv(file.file)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    features = [
        "average_corrosion_rate",
        "thickness_mm",
        "commodity",
        "feature_type",
        "cml_shape",
    ]
    X = df[features]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    results: List[dict[str, Any]] = []
    for idx, row in df.iterrows():
        results.append(
            {
                "id_number": row["id_number"],
                "predicted_elimination_flag": int(preds[idx]),
                "elimination_probability": float(probs[idx]),
            }
        )

    return {
        "rows_scored": len(df),
        "results": results[:100],
    }
