"""
FastAPI application for ChurnOps.

This API serves the trained churn prediction model.

Endpoints:
1. GET /health
2. POST /predict
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerData, HealthResponse, PredictionResponse


# ============================================================
# Model path
# ============================================================

MODEL_PATH = Path("models/churn_model.pkl")


# ============================================================
# Unified retention policy
# ============================================================
# This aligns the API with the dashboard threshold analysis.
#
# Low Risk:    churn_probability < 0.40
# Medium Risk: 0.40 <= churn_probability < 0.60
# High Risk:   churn_probability >= 0.60
#
# The 0.60 threshold is the recommended operational retention
# threshold from the threshold analysis report.

MEDIUM_RISK_THRESHOLD = 0.40
RETENTION_THRESHOLD = 0.60


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="ChurnOps API",
    description=(
        "End-to-End MLOps API for Telecom Customer Churn Prediction. "
        "The API returns churn probability, predicted label, risk level, "
        "and recommended business action."
    ),
    version="1.0.0",
)

model = None


# ============================================================
# Model loading
# ============================================================

def load_model():
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/models/train_model.py first."
        )

    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    """Load model when API starts."""
    global model
    model = load_model()


# ============================================================
# Business decision helpers
# ============================================================

def get_risk_level(churn_probability: float) -> str:
    """
    Convert churn probability into business risk level.

    Risk policy:
    - Low: probability < 0.40
    - Medium: 0.40 <= probability < 0.60
    - High: probability >= 0.60
    """
    if churn_probability >= RETENTION_THRESHOLD:
        return "High"

    if churn_probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "Low"


def get_business_action(risk_level: str) -> str:
    """Recommend business action based on risk level."""
    if risk_level == "High":
        return "Priority retention outreach."

    if risk_level == "Medium":
        return "Monitor customer or send low-cost engagement offer."

    return "No immediate retention action needed."


# ============================================================
# API endpoints
# ============================================================

@app.get("/", tags=["Root"])
def root():
    """Root endpoint."""
    return {
        "message": "Welcome to ChurnOps API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "risk_policy": {
            "low_risk": f"churn_probability < {MEDIUM_RISK_THRESHOLD}",
            "medium_risk": (
                f"{MEDIUM_RISK_THRESHOLD} <= churn_probability < "
                f"{RETENTION_THRESHOLD}"
            ),
            "high_risk": f"churn_probability >= {RETENTION_THRESHOLD}",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if API and model are available."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerData):
    """
    Predict churn risk for one customer.

    Returns:
    - prediction: 1 means customer is above the operational retention threshold
    - prediction_label: Churn Risk or No Churn Risk
    - churn_probability
    - risk_level
    - business_action
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded.",
        )

    try:
        input_df = pd.DataFrame([customer.model_dump()])

        churn_probability = float(model.predict_proba(input_df)[:, 1][0])

        # Aligned with threshold analysis:
        # customer is treated as a priority churn risk at >= 0.60
        prediction = int(churn_probability >= RETENTION_THRESHOLD)

        prediction_label = "Churn Risk" if prediction == 1 else "No Churn Risk"
        risk_level = get_risk_level(churn_probability)
        business_action = get_business_action(risk_level)

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "churn_probability": round(churn_probability, 4),
            "risk_level": risk_level,
            "business_action": business_action,
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}",
        )