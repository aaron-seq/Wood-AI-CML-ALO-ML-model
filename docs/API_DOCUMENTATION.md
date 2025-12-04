# API Documentation

## Wood AI CML Optimization API

Comprehensive API for Condition Monitoring Location (CML) elimination prediction and optimization.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Check API and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2024-12-04T10:30:00"
}
```

---

### Upload CML Data

**POST** `/upload-cml-data`

Upload and validate CML data file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File (CSV or Excel)

**Response:**
```json
{
  "filename": "cml_data.csv",
  "rows": 200,
  "columns": ["id_number", "average_corrosion_rate", ...],
  "preview": [{...}, {...}],
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": [],
    "stats": {...}
  }
}
```

---

### Score CML Data

**POST** `/score-cml-data`

Generate elimination predictions for CML data.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File (CSV or Excel)

**Required Columns:**
- `average_corrosion_rate`
- `thickness_mm`
- `commodity`
- `feature_type`
- `cml_shape`

**Response:**
```json
{
  "rows_scored": 200,
  "results": [
    {
      "id_number": "CML-001",
      "predicted_elimination": 0,
      "elimination_probability": 0.23,
      "keep_probability": 0.77,
      "recommendation": "KEEP",
      "confidence": 0.77,
      "confidence_level": "MEDIUM"
    }
  ],
  "summary": {
    "total_cmls": 200,
    "recommended_eliminations": 42,
    "recommended_keep": 158,
    "elimination_rate": 0.21
  }
}
```

---

### Forecast Remaining Life

**POST** `/forecast-remaining-life`

Forecast remaining life and inspection schedules.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File (CSV or Excel)

**Required Columns:**
- `average_corrosion_rate`
- `thickness_mm`

**Response:**
```json
{
  "forecasts": [
    {
      "id_number": "CML-001",
      "remaining_life_years": 50.0,
      "next_inspection_date": "2028-01-25",
      "recommended_inspection_frequency_months": 36,
      "risk_level": "LOW"
    }
  ],
  "summary": {
    "total_cmls": 200,
    "avg_remaining_life_years": 45.2,
    "critical_cmls": 0,
    "high_risk_cmls": 8,
    "inspections_needed_next_12_months": 23
  }
}
```

---

### Add SME Override

**POST** `/sme-override`

Add Subject Matter Expert manual decision override.

**Request Body:**
```json
{
  "id_number": "CML-042",
  "sme_decision": "KEEP",
  "reason": "Critical monitoring point for high-risk area",
  "sme_name": "Dr. John Smith",
  "original_prediction": "ELIMINATE",
  "original_probability": 0.85
}
```

**Response:**
```json
{
  "status": "success",
  "override": {...}
}
```

---

### Get SME Override

**GET** `/sme-override/{id_number}`

Retrieve SME override for specific CML.

**Response:**
```json
{
  "id_number": "CML-042",
  "sme_decision": "KEEP",
  "reason": "...",
  "sme_name": "Dr. John Smith",
  "override_date": "2024-12-04T10:30:00"
}
```

---

### Get SME Statistics

**GET** `/sme-overrides/statistics`

Get statistics about all SME overrides.

**Response:**
```json
{
  "total_overrides": 15,
  "keep_overrides": 10,
  "eliminate_overrides": 5,
  "disagreements_with_ml": 8,
  "agreement_rate": 46.7
}
```

---

### Generate Report

**POST** `/generate-report`

Generate comprehensive elimination report with SME overrides applied.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File (CSV or Excel)

**Response:**
```json
{
  "summary": {
    "total_cmls": 200,
    "recommended_eliminations": 42,
    "elimination_rate": 21.0
  },
  "confidence_distribution": {...},
  "elimination_by_commodity": {...},
  "top_elimination_candidates": [...],
  "marginal_cases": [...]
}
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message description"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error

## Authentication

Currently, the API does not require authentication. For production deployment, implement API key authentication.

## Rate Limiting

No rate limiting currently implemented. Recommended for production.

## Support

For API support, contact: aaron@smarter.codes.ai