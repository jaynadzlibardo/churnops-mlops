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

from src.api.schemas import CustomerData, PredictionResponse, HealthResponse


MODEL_PATH = Path("models/churn_model.pkl")

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


def get_risk_level(churn_probability: float) -> str:
    """Convert churn probability into business risk level."""
    if churn_probability < 0.30:
        return "Low"
    if churn_probability <= 0.60:
        return "Medium"
    return "High"


def get_business_action(risk_level: str) -> str:
    """Recommend business action based on risk level."""
    if risk_level == "Low":
        return "No immediate retention action needed. Continue normal engagement."
    if risk_level == "Medium":
        return "Monitor customer and consider low-cost retention engagement."
    return "Prioritize for retention campaign or proactive customer support."


@app.get("/", tags=["Root"])
def root():
    """Root endpoint."""
    return {
        "message": "Welcome to ChurnOps API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
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
    - prediction: 1 means churn, 0 means no churn
    - prediction_label: Churn or No Churn
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
        prediction = int(churn_probability >= 0.50)

        prediction_label = "Churn" if prediction == 1 else "No Churn"
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