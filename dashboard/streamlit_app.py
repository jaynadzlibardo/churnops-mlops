import streamlit as st

st.set_page_config(
    page_title="ChurnOps Dashboard",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("ChurnOps")
st.sidebar.caption("End-to-End MLOps Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Model Performance",
        "Confusion Matrix",
        "Threshold Analysis",
        "Feature Importance",
        "Monitoring & Drift",
        "Batch Predictions",
        "MLOps System Health",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Champion Model")
st.sidebar.write("**Model Registry Name:** `churnops_model`")
st.sidebar.write("**Alias:** `champion`")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Serving:** FastAPI + Docker")


# ------------------------------------------------------------
# Executive Overview
# ------------------------------------------------------------

if page == "Executive Overview":
    st.title("ChurnOps Executive Overview")
    st.caption(
        "Production-style churn prediction system with model registry, API deployment, monitoring, and business-ready scoring."
    )

    st.divider()

    st.subheader("Model Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "0.7474")
    col2.metric("ROC-AUC", "0.8446")
    col3.metric("Precision", "0.5165")
    col4.metric("Recall", "0.7794")
    col5.metric("F1-score", "0.6213")

    st.divider()

    st.subheader("Monitoring Health")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Features Monitored", "19")
    col2.metric("Drifted Features", "0")
    col3.metric("Data Drift", "Not Detected")
    col4.metric("Prediction Drift", "Not Detected")
    col5.metric("Retraining", "Not Needed")

    st.success(
        "Monitoring result: No data drift or prediction drift detected. The champion model remains suitable for current scoring."
    )

    st.divider()

    st.subheader("Business Interpretation")

    st.markdown(
        """
        The current champion model is optimized for **churn prioritization**, not just raw accuracy.

        The model's **recall of 0.7794** means it captures around **78% of actual churners**.
        This is important because missed churners represent lost retention opportunities.

        The **precision of 0.5165** means some customers flagged as high risk may not actually churn.
        This creates possible retention campaign cost, but it may be acceptable if the cost of losing a customer is higher than the cost of offering retention incentives.

        Current monitoring shows **0 drifted features**, so retraining is **not recommended** at this time.
        """
    )

    st.divider()

    st.subheader("Batch Prediction Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk", "2")
    col2.metric("Medium Risk", "0")
    col3.metric("Low Risk", "2")

    st.divider()

    st.subheader("Project Health Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tests", "8 passed")
    col2.metric("CI/CD", "Passed")
    col3.metric("API Health", "Model Loaded")
    col4.metric("Docker", "Working")

    st.markdown(
        """
        This confirms that ChurnOps is not only a notebook-based model. 
        It includes automated testing, CI validation, API serving, containerization, 
        model registry, monitoring, and batch scoring.
        """
    )

# ------------------------------------------------------------
# Placeholder pages
# ------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")
    st.info("Next phase: metric charts, ROC curve, precision-recall curve, and classification report.")

elif page == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.info("Next phase: confusion matrix heatmap and business error analysis.")

elif page == "Threshold Analysis":
    st.title("Threshold Analysis")
    st.info("Later phase: threshold slider, precision-recall-F1 trade-off charts, and retention policy analysis.")

elif page == "Feature Importance":
    st.title("Feature Importance")
    st.info("Later phase: top churn drivers and business action mapping.")

elif page == "Monitoring & Drift":
    st.title("Monitoring & Drift")
    st.info("Later phase: drift status cards, drift tables, and Evidently report integration.")

elif page == "Batch Predictions":
    st.title("Batch Predictions")
    st.info("Later phase: batch scoring table, risk bands, churn probability charts, and CSV download.")

elif page == "MLOps System Health":
    st.title("MLOps System Health")
    st.info("Later phase: model registry status, API status, Docker status, test status, and deployment checklist.")