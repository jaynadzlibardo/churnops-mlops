"""
Streamlit dashboard for ChurnOps.

This dashboard provides:
1. Business problem overview
2. Manual customer churn prediction
3. Model metrics summary
4. Feature importance visualization
5. Threshold analysis visualization
6. Monitoring status summary
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/churn_model.pkl")
TEST_METRICS_PATH = Path("reports/test_metrics_report.json")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.png")
THRESHOLD_ANALYSIS_PATH = Path("reports/threshold_analysis.csv")
THRESHOLD_PLOT_PATH = Path("reports/threshold_analysis.png")
MONITORING_SUMMARY_PATH = Path("reports/monitoring_summary.json")

st.set_page_config(
    page_title="ChurnOps Dashboard",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load trained churn model."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_json(path: Path):
    """Load JSON file if available."""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_risk_level(churn_probability: float) -> str:
    """Convert churn probability to risk level."""
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


def build_customer_input():
    """Create customer input form."""
    st.subheader("Manual Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure", min_value=0, max_value=72, value=5)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )

    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=95.70,
            step=0.10,
        )

    with col5:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=478.50,
            step=0.10,
        )

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return customer


def show_metrics():
    """Display final model metrics."""
    st.subheader("Final Test Metrics")

    metrics_report = load_json(TEST_METRICS_PATH)

    if metrics_report is None:
        st.warning("Test metrics report not found. Run evaluate_model.py first.")
        return

    metrics = metrics_report["metrics"]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col3.metric("Precision", f"{metrics['precision']:.4f}")
    col4.metric("Recall", f"{metrics['recall']:.4f}")
    col5.metric("F1-score", f"{metrics['f1']:.4f}")

    st.info(
        "Recall is important because missed churners represent possible lost revenue. "
        "Precision matters because false positives may waste retention offers."
    )


def show_monitoring():
    """Display monitoring summary."""
    st.subheader("Monitoring Summary")

    monitoring_summary = load_json(MONITORING_SUMMARY_PATH)

    if monitoring_summary is None:
        st.warning("Monitoring summary not found. Run generate_drift_report.py first.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Drifted Features", monitoring_summary["drifted_features"])
    col2.metric(
        "Drift Share",
        f"{monitoring_summary['drifted_feature_share']:.2%}",
    )
    col3.metric("Prediction Drift", str(monitoring_summary["prediction_drift_detected"]))
    col4.metric("Retraining", str(monitoring_summary["retraining_recommended"]))

    if monitoring_summary["alert_level"] == "OK":
        st.success(monitoring_summary["recommended_action"])
    else:
        st.error(monitoring_summary["recommended_action"])


def main():
    st.title("ChurnOps: Telecom Customer Churn MLOps Dashboard")

    st.markdown(
        """
        This dashboard demonstrates the business-facing layer of the ChurnOps MLOps system.
        The goal is to identify customers likely to churn so the retention team can prioritize
        interventions and reduce revenue loss.
        """
    )

    model = load_model()

    if model is None:
        st.error("Model not found. Run `python src/models/train_model.py` first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Prediction",
            "Model Metrics",
            "Explainability",
            "Monitoring",
        ]
    )

    with tab1:
        customer = build_customer_input()

        if st.button("Predict Churn Risk"):
            input_df = pd.DataFrame([customer])
            churn_probability = float(model.predict_proba(input_df)[:, 1][0])
            prediction = int(churn_probability >= 0.50)
            prediction_label = "Churn" if prediction == 1 else "No Churn"
            risk_level = get_risk_level(churn_probability)
            business_action = get_business_action(risk_level)

            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            col1.metric("Prediction", prediction_label)
            col2.metric("Churn Probability", f"{churn_probability:.4f}")
            col3.metric("Risk Level", risk_level)

            if risk_level == "High":
                st.error(business_action)
            elif risk_level == "Medium":
                st.warning(business_action)
            else:
                st.success(business_action)

    with tab2:
        show_metrics()

        if THRESHOLD_PLOT_PATH.exists():
            st.subheader("Threshold Analysis")
            st.image(str(THRESHOLD_PLOT_PATH))

        if THRESHOLD_ANALYSIS_PATH.exists():
            threshold_df = pd.read_csv(THRESHOLD_ANALYSIS_PATH)
            st.dataframe(threshold_df)

    with tab3:
        st.subheader("Feature Importance")

        if FEATURE_IMPORTANCE_PATH.exists():
            st.image(str(FEATURE_IMPORTANCE_PATH))
            st.info(
                "Feature importance shows which variables have the strongest influence "
                "on churn prediction. This is model explainability, not causal proof."
            )
        else:
            st.warning("Feature importance plot not found. Run feature_importance.py first.")

    with tab4:
        show_monitoring()


if __name__ == "__main__":
    main()