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
    st.caption(
        "Decision threshold analysis for converting churn probabilities into retention actions."
    )

    st.divider()

    best_threshold = 0.60
    threshold_precision = 0.5714
    threshold_recall = 0.6975
    threshold_f1 = 0.6282
    flagged_customers = 343
    missed_churners = 85

    st.subheader("Recommended Threshold Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Best Threshold", f"{best_threshold:.2f}")
    col2.metric("Precision", f"{threshold_precision:.4f}")
    col3.metric("Recall", f"{threshold_recall:.4f}")
    col4.metric("F1-score", f"{threshold_f1:.4f}")
    col5.metric("Flagged Customers", flagged_customers)

    st.divider()

    st.subheader("Retention Trade-off")

    col1, col2 = st.columns(2)

    with col1:
        st.warning(
            f"At threshold {best_threshold:.2f}, the model misses {missed_churners} actual churners. "
            "These customers may leave without receiving a retention intervention."
        )

    with col2:
        st.success(
            f"At threshold {best_threshold:.2f}, the model flags {flagged_customers} customers for retention action. "
            "This gives the business a focused customer list for outreach."
        )

    st.divider()
    st.subheader("Threshold Trade-off Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        threshold_metrics_df = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall", "F1-score"],
                "Score": [threshold_precision, threshold_recall, threshold_f1],
            }
        )

        fig, ax = plt.subplots(figsize=(4, 3))

        ax.barh(
            threshold_metrics_df["Metric"],
            threshold_metrics_df["Score"],
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("Performance at Threshold 0.60")

        for index, value in enumerate(threshold_metrics_df["Score"]):
            ax.text(
                value + 0.01,
                index,
                f"{value:.4f}",
                va="center",
                fontsize=8,
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with chart_col2:
        operational_df = pd.DataFrame(
            {
                "Metric": ["Flagged Customers", "Missed Churners"],
                "Count": [flagged_customers, missed_churners],
            }
        )

        fig, ax = plt.subplots(figsize=(4, 3))

        ax.barh(
            operational_df["Metric"],
            operational_df["Count"],
        )
        ax.set_xlabel("Customer Count")
        ax.set_title("Operational Impact at Threshold 0.60")

        for index, value in enumerate(operational_df["Count"]):
            ax.text(
                value + 5,
                index,
                str(value),
                va="center",
                fontsize=8,
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    st.divider()
    st.subheader("Business Decision Logic")

    threshold_logic = pd.DataFrame(
        [
            {
                "Decision Option": "Lower threshold",
                "Expected Effect": "More customers flagged",
                "Benefit": "Catches more churners",
                "Risk": "Higher retention campaign cost and more false positives",
            },
            {
                "Decision Option": "Higher threshold",
                "Expected Effect": "Fewer customers flagged",
                "Benefit": "More efficient campaign spend",
                "Risk": "More missed churners and lost revenue opportunity",
            },
            {
                "Decision Option": "Use 0.60 threshold",
                "Expected Effect": "Balanced precision and recall",
                "Benefit": "Best F1-score in current analysis",
                "Risk": "Still misses some churners",
            },
        ]
    )

    st.dataframe(threshold_logic, use_container_width=True)

    st.divider()

    st.subheader("Recommended Retention Policy")

    st.info(
        "Recommended policy: Use the 0.60 threshold for the initial retention campaign. "
        "This threshold gives the best F1-score and balances campaign efficiency with churn capture."
    )

    st.markdown(
        """
        **Business use:**

        - Customers with churn probability **>= 0.60** should be tagged as high-risk.
        - High-risk customers should be prioritized for retention outreach.
        - If the retention budget is limited, increase the threshold.
        - If lost customers are more expensive than retention offers, lower the threshold.
        """
    )

elif page == "Feature Importance":
    st.title("Feature Importance")
    st.caption(
        "Top churn drivers from the champion Logistic Regression model."
    )

    st.divider()

    feature_importance_df = pd.DataFrame(
        {
            "Feature": [
                "tenure",
                "Contract_Two year",
                "Contract_Month-to-month",
                "TotalCharges",
                "InternetService_Fiber optic",
                "InternetService_DSL",
                "MonthlyCharges",
            ],
            "Importance Rank": [1, 2, 3, 4, 5, 6, 7],
            "Importance Score": [7, 6, 5, 4, 3, 2, 1],
        }
    )

    st.subheader("Top Churn Drivers")

    col1, col2, col3 = st.columns(3)

    col1.metric("Top Driver", "tenure")
    col2.metric("Drivers Shown", "7")
    col3.metric("Model Type", "Logistic Regression")

    st.divider()

    st.subheader("Feature Importance Chart")

    chart_df = feature_importance_df.sort_values(
        "Importance Score",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(
        chart_df["Feature"],
        chart_df["Importance Score"],
    )

    ax.set_xlabel("Relative Importance Score")
    ax.set_title("Top Churn Drivers")

    for index, value in enumerate(chart_df["Importance Score"]):
        ax.text(
            value + 0.1,
            index,
            str(value),
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.divider()

    st.subheader("Business Interpretation")

    business_actions = pd.DataFrame(
        [
            {
                "Feature": "tenure",
                "Business Meaning": "Customer lifetime is a major churn signal. Short-tenure customers are usually less loyal and more likely to leave.",
                "Recommended Action": "Improve onboarding, early-life customer support, and first 90-day engagement campaigns.",
            },
            {
                "Feature": "Contract_Two year",
                "Business Meaning": "Longer contracts usually reduce churn risk because customers have stronger commitment.",
                "Recommended Action": "Promote longer-term contracts to stable customers with targeted upgrade offers.",
            },
            {
                "Feature": "Contract_Month-to-month",
                "Business Meaning": "Month-to-month customers can leave more easily, increasing churn risk.",
                "Recommended Action": "Target month-to-month customers with loyalty discounts or contract conversion offers.",
            },
            {
                "Feature": "TotalCharges",
                "Business Meaning": "Total charges are related to customer lifetime value and account maturity.",
                "Recommended Action": "Prioritize high-value customers for proactive retention outreach.",
            },
            {
                "Feature": "InternetService_Fiber optic",
                "Business Meaning": "Fiber optic customers may represent a segment with higher price sensitivity or service expectations.",
                "Recommended Action": "Investigate fiber customer complaints, pricing, speed issues, and service quality experience.",
            },
            {
                "Feature": "InternetService_DSL",
                "Business Meaning": "DSL customers may behave differently from fiber customers in churn patterns.",
                "Recommended Action": "Compare churn rates by internet service type and tailor offers by segment.",
            },
            {
                "Feature": "MonthlyCharges",
                "Business Meaning": "Higher monthly bills may increase churn risk if customers perceive low value for money.",
                "Recommended Action": "Review pricing, bundles, discounts, and value communication for high-bill customers.",
            },
        ]
    )

    st.dataframe(business_actions, use_container_width=True)

    st.divider()

    st.subheader("Retention Strategy Recommendation")

    st.success(
        "Recommendation: Focus retention efforts on customers with short tenure, month-to-month contracts, high monthly charges, and risky internet service segments."
    )

    st.info(
        "Business decision: Use feature importance to design targeted campaigns instead of giving the same retention offer to every customer."
    )

    st.warning(
        "Model risk: This page currently uses rank-based importance. For stronger explainability, use actual Logistic Regression coefficients or SHAP values in the next iteration."
    )

elif page == "Monitoring & Drift":
    st.title("Monitoring & Drift")
    st.info("Later phase: drift status cards, drift tables, and Evidently report integration.")

elif page == "Batch Predictions":
    st.title("Batch Predictions")
    st.info("Later phase: batch scoring table, risk bands, churn probability charts, and CSV download.")

elif page == "MLOps System Health":
    st.title("MLOps System Health")
    st.info("Later phase: model registry status, API status, Docker status, test status, and deployment checklist.")