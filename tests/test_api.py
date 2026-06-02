"""
API tests for ChurnOps FastAPI application.

These tests use TestClient as a context manager so FastAPI startup events run
and the trained model is loaded before requests are tested.
"""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    """Test that the health endpoint returns model status."""
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "model_path" in data


def test_predict_endpoint():
    """Test that the prediction endpoint returns churn prediction output."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.70,
        "TotalCharges": 478.50,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "prediction_label" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "business_action" in data

        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["Churn", "No Churn"]
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ["Low", "Medium", "High"]