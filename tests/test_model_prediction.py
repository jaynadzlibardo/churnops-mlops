from pathlib import Path
import joblib
import pandas as pd


def test_model_file_exists():
    assert Path("models/churn_model.pkl").exists()


def test_model_can_predict():
    model = joblib.load("models/churn_model.pkl")

    sample = pd.DataFrame(
        [
            {
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
        ]
    )

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    assert prediction.shape[0] == 1
    assert probability.shape == (1, 2)
    assert 0 <= probability[0, 1] <= 1