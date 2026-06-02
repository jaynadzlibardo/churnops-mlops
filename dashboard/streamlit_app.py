import matplotlib.pyplot as plt
import pandas as pd
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
    st.caption(
        "Final test-set performance of the champion Logistic Regression churn model."
    )

    st.divider()

    accuracy = 0.7474
    roc_auc = 0.8446
    precision = 0.5165
    recall = 0.7794
    f1_score = 0.6213

    st.subheader("Performance KPI Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("ROC-AUC", f"{roc_auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1-score", f"{f1_score:.4f}")

    st.divider()

    st.subheader("Metric Comparison Chart")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["ROC-AUC", "Recall", "Accuracy", "F1-score", "Precision"],
            "Score": [roc_auc, recall, accuracy, f1_score, precision],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.barh(metrics_df["Metric"], metrics_df["Score"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Champion Model Test Metrics")

    for index, value in enumerate(metrics_df["Score"]):
        ax.text(
            value + 0.01,
            index,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.divider()

    st.subheader("Business Meaning of Each Metric")

    metric_meaning = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": accuracy,
                "Business Meaning": "Overall percentage of correct churn and non-churn predictions.",
                "Interpretation": "Useful as a general metric, but not enough by itself for churn because churn classes are usually imbalanced.",
            },
            {
                "Metric": "ROC-AUC",
                "Value": roc_auc,
                "Business Meaning": "Measures how well the model separates churners from non-churners.",
                "Interpretation": "Strong result. A ROC-AUC of 0.8446 means the model has good ranking ability.",
            },
            {
                "Metric": "Precision",
                "Value": precision,
                "Business Meaning": "Of customers flagged as churn risks, how many actually churned.",
                "Interpretation": "Moderate. Some retention offers may be spent on customers who would not churn.",
            },
            {
                "Metric": "Recall",
                "Value": recall,
                "Business Meaning": "Of all actual churners, how many the model successfully captured.",
                "Interpretation": "Strong for retention use case. The model catches about 78% of churners.",
            },
            {
                "Metric": "F1-score",
                "Value": f1_score,
                "Business Meaning": "Balances precision and recall.",
                "Interpretation": "Good summary metric when both missed churners and wasted offers matter.",
            },
        ]
    )

    st.dataframe(metric_meaning, use_container_width=True)

    st.divider()

    st.subheader("Model Recommendation")

    st.success(
        "Recommendation: Use the Logistic Regression champion model for churn prioritization. "
        "The model has strong ROC-AUC and recall, making it useful for identifying customers who need retention action."
    )

    st.warning(
        "Main trade-off: Precision is moderate. This means the business may spend some retention budget on customers who would not actually churn. "
        "Threshold tuning should be used to control this campaign cost."
    )

    st.info(
        "Business decision: If the company wants to reduce missed churners, prioritize recall. "
        "If the company wants to reduce wasted retention offers, increase the decision threshold to improve precision."
    )

elif page == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.caption(
        "Final test-set confusion matrix for the champion churn prediction model."
    )

    st.divider()

    tn = 571
    fp = 205
    fn = 62
    tp = 219

    st.subheader("Confusion Matrix Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True Negatives", tn)
    col2.metric("False Positives", fp)
    col3.metric("False Negatives", fn)
    col4.metric("True Positives", tp)

    st.divider()

    st.subheader("Confusion Matrix Heatmap")

    matrix = [[tn, fp], [fn, tp]]

    fig, ax = plt.subplots(figsize=(4, 3))

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred. No Churn", "Pred. Churn"], fontsize=8)
    ax.set_yticklabels(["Actual No Churn", "Actual Churn"], fontsize=8)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.divider()

    st.subheader("Business Error Analysis")

    error_table = [
        {
            "Outcome": "True Negative",
            "Count": tn,
            "Meaning": "Correctly identified non-churners.",
            "Business Impact": "No unnecessary retention action.",
        },
        {
            "Outcome": "False Positive",
            "Count": fp,
            "Meaning": "Non-churners incorrectly flagged as churn risks.",
            "Business Impact": "Possible wasted retention offer cost.",
        },
        {
            "Outcome": "False Negative",
            "Count": fn,
            "Meaning": "Actual churners missed by the model.",
            "Business Impact": "Lost retention opportunity and revenue risk.",
        },
        {
            "Outcome": "True Positive",
            "Count": tp,
            "Meaning": "Correctly identified actual churners.",
            "Business Impact": "Good target group for retention campaign.",
        },
    ]

    st.dataframe(error_table, use_container_width=True)

    st.warning(
        "Key business risk: 62 churners were missed. These false negatives represent customers who may leave without receiving a retention intervention."
    )

    st.success(
        "Key business value: 219 churners were correctly identified. These customers can be prioritized for retention outreach."
    )

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