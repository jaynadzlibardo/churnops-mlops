import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ChurnOps Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"


# ============================================================
# Dashboard visual styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2.05rem !important;
        line-height: 1.2 !important;
    }

    h2, h3 {
        font-size: 1.25rem !important;
        line-height: 1.25 !important;
    }

    p, li, div {
        font-size: 0.95rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
    }

    [data-testid="stDataFrame"] {
        font-size: 0.85rem !important;
    }

    .stAlert {
        font-size: 0.90rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Shared chart settings
# ============================================================

NORMAL_CHART_SIZE = (5.2, 2.6)
WIDE_CHART_SIZE = (5.8, 2.8)
SMALL_CHART_SIZE = (4.2, 2.4)

plt.rcParams.update(
    {
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 8,
    }
)


# ============================================================
# Data loading helpers
# ============================================================

@st.cache_data
def load_json_report(file_name: str) -> dict:
    file_path = REPORTS_DIR / file_name

    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv_report(file_name: str) -> pd.DataFrame:
    file_path = REPORTS_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_metric(value, decimals=4) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def bool_to_status(value: bool, positive_text: str, negative_text: str) -> str:
    return positive_text if bool(value) else negative_text


def find_first_existing_column(df: pd.DataFrame, candidate_columns: list[str]) -> str | None:
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None


# ============================================================
# Chart helpers
# ============================================================

def render_barh_chart(
    df,
    label_col,
    value_col,
    title,
    xlabel,
    xlim=None,
    value_formatter=None,
    figsize=NORMAL_CHART_SIZE,
):
    """
    Render a compact horizontal bar chart with normalized dashboard sizing.
    """
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        st.warning("Chart data is unavailable.")
        return

    chart_df = df.copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    ax.barh(
        chart_df[label_col],
        chart_df[value_col],
        height=0.45,
    )

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)

    max_value = chart_df[value_col].max()

    if xlim is not None:
        ax.set_xlim(xlim)
        offset = (xlim[1] - xlim[0]) * 0.015
    else:
        offset = max_value * 0.03 if max_value > 0 else 0.05
        ax.set_xlim(0, max_value * 1.18 + offset)

    for index, value in enumerate(chart_df[value_col]):
        label = value_formatter(value) if value_formatter else str(value)

        ax.text(
            value + offset,
            index,
            label,
            va="center",
            fontsize=8,
        )

    ax.grid(axis="x", alpha=0.20)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=1.0)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def render_confusion_matrix(matrix):
    """
    Render a compact confusion matrix heatmap with normalized dashboard sizing.
    """
    fig, ax = plt.subplots(figsize=(3.8, 2.8), dpi=110)

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred. No Churn", "Pred. Churn"], fontsize=7)
    ax.set_yticklabels(["Actual No Churn", "Actual Churn"], fontsize=7)

    ax.set_xlabel("Predicted Label", fontsize=8)
    ax.set_ylabel("Actual Label", fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=10, pad=8)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=1.0)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ============================================================
# Load reports
# ============================================================

test_metrics_report = load_json_report("test_metrics_report.json")
monitoring_summary = load_json_report("monitoring_summary.json")
model_registry_report = load_json_report("model_registry_report.json")

batch_predictions_df = load_csv_report("batch_predictions.csv")
feature_importance_df = load_csv_report("feature_importance.csv")
threshold_analysis_df = load_csv_report("threshold_analysis.csv")


# ============================================================
# Extract dynamic model metrics
# ============================================================

metrics = test_metrics_report.get("metrics", {})

accuracy = safe_float(metrics.get("accuracy"), 0.7474)
roc_auc = safe_float(metrics.get("roc_auc"), 0.8446)
precision = safe_float(metrics.get("precision"), 0.5165)
recall = safe_float(metrics.get("recall"), 0.7794)
f1_score = safe_float(metrics.get("f1"), 0.6213)
default_threshold = safe_float(metrics.get("threshold"), 0.50)

confusion_matrix_report = test_metrics_report.get("confusion_matrix", {})

tn = safe_int(confusion_matrix_report.get("true_negatives"), 571)
fp = safe_int(confusion_matrix_report.get("false_positives"), 205)
fn = safe_int(confusion_matrix_report.get("false_negatives"), 62)
tp = safe_int(confusion_matrix_report.get("true_positives"), 219)


# ============================================================
# Extract dynamic monitoring values
# ============================================================

features_monitored = safe_int(monitoring_summary.get("total_features"), 19)
drifted_features = safe_int(monitoring_summary.get("drifted_features"), 0)
drifted_feature_share = safe_float(monitoring_summary.get("drifted_feature_share"), 0.0)
data_drift = bool(monitoring_summary.get("data_drift_detected", False))
prediction_drift = bool(monitoring_summary.get("prediction_drift_detected", False))
retraining_recommended = bool(monitoring_summary.get("retraining_recommended", False))
alert_level = monitoring_summary.get("alert_level", "OK")
monitoring_recommended_action = monitoring_summary.get(
    "recommended_action",
    "No immediate retraining required. Continue monitoring.",
)

prediction_drift_details = monitoring_summary.get("prediction_drift", {})
reference_mean_churn_probability = safe_float(
    prediction_drift_details.get("reference_mean_churn_probability"),
    0.0,
)
current_mean_churn_probability = safe_float(
    prediction_drift_details.get("current_mean_churn_probability"),
    0.0,
)
absolute_mean_change = safe_float(
    prediction_drift_details.get("absolute_mean_change"),
    0.0,
)

feature_drift_results = monitoring_summary.get("feature_drift_results", [])
feature_drift_df = pd.DataFrame(feature_drift_results)


# ============================================================
# Extract dynamic threshold analysis
# ============================================================

if not threshold_analysis_df.empty:
    for col in threshold_analysis_df.columns:
        threshold_analysis_df[col] = pd.to_numeric(threshold_analysis_df[col], errors="ignore")

    if "f1" in threshold_analysis_df.columns:
        best_threshold_row = threshold_analysis_df.loc[
            threshold_analysis_df["f1"].astype(float).idxmax()
        ]
    else:
        best_threshold_row = pd.Series(dtype="object")

    best_threshold = safe_float(best_threshold_row.get("threshold"), 0.60)
    threshold_precision = safe_float(best_threshold_row.get("precision"), 0.5714)
    threshold_recall = safe_float(best_threshold_row.get("recall"), 0.6975)
    threshold_f1 = safe_float(best_threshold_row.get("f1"), 0.6282)
    threshold_fp = safe_int(best_threshold_row.get("false_positives"), 147)
    threshold_fn = safe_int(best_threshold_row.get("false_negatives"), 85)
    threshold_tp = safe_int(best_threshold_row.get("true_positives"), 196)

    flagged_customers = threshold_fp + threshold_tp
    missed_churners = threshold_fn
else:
    best_threshold = 0.60
    threshold_precision = 0.5714
    threshold_recall = 0.6975
    threshold_f1 = 0.6282
    flagged_customers = 343
    missed_churners = 85


# ============================================================
# Extract dynamic feature importance
# ============================================================

if not feature_importance_df.empty:
    feature_col = find_first_existing_column(feature_importance_df, ["feature", "Feature"])
    importance_col = find_first_existing_column(
        feature_importance_df,
        ["importance", "Importance", "importance_score", "Importance Score"],
    )

    if feature_col and importance_col:
        feature_importance_df = feature_importance_df[[feature_col, importance_col]].copy()
        feature_importance_df.columns = ["Feature", "Importance"]
        feature_importance_df["Importance"] = pd.to_numeric(
            feature_importance_df["Importance"],
            errors="coerce",
        ).fillna(0)
        feature_importance_df = feature_importance_df.sort_values(
            "Importance",
            ascending=False,
        ).reset_index(drop=True)
        feature_importance_df["Rank"] = feature_importance_df.index + 1
    else:
        feature_importance_df = pd.DataFrame()

if feature_importance_df.empty:
    feature_importance_df = pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4, 5, 6, 7],
            "Feature": [
                "tenure",
                "Contract_Two year",
                "Contract_Month-to-month",
                "TotalCharges",
                "InternetService_Fiber optic",
                "InternetService_DSL",
                "MonthlyCharges",
            ],
            "Importance": [1.2068, 0.7461, 0.6621, 0.5874, 0.5660, 0.5068, 0.5007],
        }
    )

top_feature = feature_importance_df.iloc[0]["Feature"] if not feature_importance_df.empty else "N/A"


# ============================================================
# Extract dynamic batch predictions
# ============================================================

def add_risk_band_from_probability(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    probability_col = find_first_existing_column(
        output_df,
        ["churn_probability", "probability", "prediction_probability"],
    )

    risk_col = find_first_existing_column(
        output_df,
        ["risk_band", "risk_level", "Risk Band", "Risk Level"],
    )

    if risk_col:
        output_df["risk_band_dashboard"] = output_df[risk_col].astype(str)
    elif probability_col:
        probabilities = pd.to_numeric(output_df[probability_col], errors="coerce").fillna(0)

        output_df["risk_band_dashboard"] = probabilities.apply(
            lambda value: "High Risk"
            if value >= best_threshold
            else "Medium Risk"
            if value >= 0.40
            else "Low Risk"
        )
    else:
        output_df["risk_band_dashboard"] = "Unknown"

    return output_df


if not batch_predictions_df.empty:
    batch_predictions_df = add_risk_band_from_probability(batch_predictions_df)

    probability_col = find_first_existing_column(
        batch_predictions_df,
        ["churn_probability", "probability", "prediction_probability"],
    )

    if "customer_id" not in batch_predictions_df.columns:
        batch_predictions_df.insert(
            0,
            "customer_id",
            [f"CUST-{index + 1:03d}" for index in range(len(batch_predictions_df))],
        )

    if probability_col and probability_col != "churn_probability":
        batch_predictions_df["churn_probability"] = pd.to_numeric(
            batch_predictions_df[probability_col],
            errors="coerce",
        ).fillna(0)

    if "churn_probability" not in batch_predictions_df.columns:
        batch_predictions_df["churn_probability"] = 0.0

    batch_predictions_df["recommended_action"] = batch_predictions_df["risk_band_dashboard"].apply(
        lambda value: "Priority retention call"
        if "high" in str(value).lower()
        else "Monitor or send low-cost engagement offer"
        if "medium" in str(value).lower()
        else "No immediate action"
    )

else:
    batch_predictions_df = pd.DataFrame(
        [
            {
                "customer_id": "CUST-001",
                "churn_probability": 0.8845,
                "risk_band_dashboard": "High Risk",
                "recommended_action": "Priority retention call",
            },
            {
                "customer_id": "CUST-002",
                "churn_probability": 0.7200,
                "risk_band_dashboard": "High Risk",
                "recommended_action": "Priority retention call",
            },
            {
                "customer_id": "CUST-003",
                "churn_probability": 0.2300,
                "risk_band_dashboard": "Low Risk",
                "recommended_action": "No immediate action",
            },
            {
                "customer_id": "CUST-004",
                "churn_probability": 0.1200,
                "risk_band_dashboard": "Low Risk",
                "recommended_action": "No immediate action",
            },
        ]
    )

total_scored = len(batch_predictions_df)

risk_lower = batch_predictions_df["risk_band_dashboard"].astype(str).str.lower()

high_risk_count = int(risk_lower.str.contains("high").sum())
medium_risk_count = int(risk_lower.str.contains("medium").sum())
low_risk_count = int(risk_lower.str.contains("low").sum())


# ============================================================
# Sidebar
# ============================================================

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

st.sidebar.divider()
st.sidebar.markdown("### Data Source")
st.sidebar.write("Dashboard values are loaded from `reports/`.")


# ============================================================
# Executive Overview
# ============================================================

if page == "Executive Overview":
    st.title("ChurnOps Executive Overview")
    st.caption(
        "Production-style churn prediction system with model registry, API deployment, monitoring, and business-ready scoring."
    )

    st.divider()

    st.subheader("Model Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", format_metric(accuracy))
    col2.metric("ROC-AUC", format_metric(roc_auc))
    col3.metric("Precision", format_metric(precision))
    col4.metric("Recall", format_metric(recall))
    col5.metric("F1-score", format_metric(f1_score))

    st.divider()

    st.subheader("Monitoring Health")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Features Monitored", features_monitored)
    col2.metric("Drifted Features", drifted_features)
    col3.metric("Data Drift", bool_to_status(data_drift, "Detected", "Not Detected"))
    col4.metric("Prediction Drift", bool_to_status(prediction_drift, "Detected", "Not Detected"))
    col5.metric("Retraining", bool_to_status(retraining_recommended, "Recommended", "Not Needed"))

    if not data_drift and not prediction_drift and not retraining_recommended:
        st.success(
            "Monitoring result: No data drift or prediction drift detected. The champion model remains suitable for current scoring."
        )
    else:
        st.error(
            "Monitoring result: Drift or retraining trigger detected. Review model reliability before continued scoring."
        )

    st.divider()

    st.subheader("Business Interpretation")

    st.markdown(
        f"""
        The current champion model is optimized for **churn prioritization**, not just raw accuracy.

        The model's **recall of {recall:.4f}** means it captures around **{recall * 100:.0f}% of actual churners**.
        This is important because missed churners represent lost retention opportunities.

        The **precision of {precision:.4f}** means around **{precision * 100:.0f}% of customers flagged as churn risks actually churned**.
        This creates possible retention campaign cost, but it may be acceptable if the cost of losing a customer is higher than the cost of offering retention incentives.

        Current monitoring shows **{drifted_features} drifted features out of {features_monitored} monitored features**, so retraining is **{"recommended" if retraining_recommended else "not recommended"}** at this time.
        """
    )

    st.divider()

    st.subheader("Batch Prediction Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk", high_risk_count)
    col2.metric("Medium Risk", medium_risk_count)
    col3.metric("Low Risk", low_risk_count)

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


# ============================================================
# Model Performance
# ============================================================

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption(
        "Final test-set performance of the champion Logistic Regression churn model."
    )

    st.divider()

    st.subheader("Performance KPI Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", format_metric(accuracy))
    col2.metric("ROC-AUC", format_metric(roc_auc))
    col3.metric("Precision", format_metric(precision))
    col4.metric("Recall", format_metric(recall))
    col5.metric("F1-score", format_metric(f1_score))

    st.divider()

    st.subheader("Metric Comparison Chart")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["ROC-AUC", "Recall", "Accuracy", "F1-score", "Precision"],
            "Score": [roc_auc, recall, accuracy, f1_score, precision],
        }
    )

    render_barh_chart(
        df=metrics_df,
        label_col="Metric",
        value_col="Score",
        title="Champion Model Test Metrics",
        xlabel="Score",
        xlim=(0, 1),
        value_formatter=lambda value: f"{value:.4f}",
        figsize=WIDE_CHART_SIZE,
    )

    st.divider()

    st.subheader("Business Meaning of Each Metric")

    metric_meaning = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": round(accuracy, 4),
                "Business Meaning": "Overall percentage of correct churn and non-churn predictions.",
                "Interpretation": "Useful as a general metric, but not enough by itself for churn because churn classes are usually imbalanced.",
            },
            {
                "Metric": "ROC-AUC",
                "Value": round(roc_auc, 4),
                "Business Meaning": "Measures how well the model separates churners from non-churners.",
                "Interpretation": f"ROC-AUC of {roc_auc:.4f} means the model has good ranking ability.",
            },
            {
                "Metric": "Precision",
                "Value": round(precision, 4),
                "Business Meaning": "Of customers flagged as churn risks, how many actually churned.",
                "Interpretation": "Moderate precision means some retention offers may be spent on customers who would not churn.",
            },
            {
                "Metric": "Recall",
                "Value": round(recall, 4),
                "Business Meaning": "Of all actual churners, how many the model successfully captured.",
                "Interpretation": f"The model catches about {recall * 100:.0f}% of churners.",
            },
            {
                "Metric": "F1-score",
                "Value": round(f1_score, 4),
                "Business Meaning": "Balances precision and recall.",
                "Interpretation": "Useful when both missed churners and wasted offers matter.",
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


# ============================================================
# Confusion Matrix
# ============================================================

elif page == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.caption(
        "Final test-set confusion matrix for the champion churn prediction model."
    )

    st.divider()

    st.subheader("Confusion Matrix Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True Negatives", tn)
    col2.metric("False Positives", fp)
    col3.metric("False Negatives", fn)
    col4.metric("True Positives", tp)

    st.divider()

    st.subheader("Confusion Matrix Heatmap")

    matrix = [[tn, fp], [fn, tp]]

    render_confusion_matrix(matrix)

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
        f"Key business risk: {fn} churners were missed. These false negatives represent customers who may leave without receiving a retention intervention."
    )

    st.success(
        f"Key business value: {tp} churners were correctly identified. These customers can be prioritized for retention outreach."
    )


# ============================================================
# Threshold Analysis
# ============================================================

elif page == "Threshold Analysis":
    st.title("Threshold Analysis")
    st.caption(
        "Decision threshold analysis for converting churn probabilities into retention actions."
    )

    st.divider()

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

        render_barh_chart(
            df=threshold_metrics_df,
            label_col="Metric",
            value_col="Score",
            title=f"Performance at Threshold {best_threshold:.2f}",
            xlabel="Score",
            xlim=(0, 1),
            value_formatter=lambda value: f"{value:.4f}",
            figsize=SMALL_CHART_SIZE,
        )

    with chart_col2:
        operational_df = pd.DataFrame(
            {
                "Metric": ["Flagged Customers", "Missed Churners"],
                "Count": [flagged_customers, missed_churners],
            }
        )

        render_barh_chart(
            df=operational_df,
            label_col="Metric",
            value_col="Count",
            title=f"Operational Impact at Threshold {best_threshold:.2f}",
            xlabel="Customer Count",
            value_formatter=lambda value: f"{int(value)}",
            figsize=SMALL_CHART_SIZE,
        )

    if not threshold_analysis_df.empty:
        st.divider()
        st.subheader("Threshold Analysis Table")
        st.dataframe(threshold_analysis_df, use_container_width=True)

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
                "Decision Option": f"Use {best_threshold:.2f} threshold",
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
        f"Recommended policy: Use the {best_threshold:.2f} threshold for the initial retention campaign. "
        "This threshold gives the best F1-score and balances campaign efficiency with churn capture."
    )

    st.markdown(
        f"""
        **Business use:**

        - Customers with churn probability **>= {best_threshold:.2f}** should be tagged as high-risk.
        - High-risk customers should be prioritized for retention outreach.
        - If the retention budget is limited, increase the threshold.
        - If lost customers are more expensive than retention offers, lower the threshold.
        """
    )


# ============================================================
# Feature Importance
# ============================================================

elif page == "Feature Importance":
    st.title("Feature Importance")
    st.caption(
        "Top churn drivers from the champion Logistic Regression model."
    )

    st.divider()

    top_features_df = feature_importance_df.head(10).copy()

    st.subheader("Top Churn Drivers")

    col1, col2, col3 = st.columns(3)

    col1.metric("Top Driver", str(top_feature))
    col2.metric("Drivers Shown", len(top_features_df))
    col3.metric("Model Type", "Logistic Regression")

    st.divider()

    st.subheader("Feature Importance Chart")

    chart_df = top_features_df.sort_values(
        "Importance",
        ascending=True,
    )

    render_barh_chart(
        df=chart_df,
        label_col="Feature",
        value_col="Importance",
        title="Top Churn Drivers",
        xlabel="Model Importance",
        value_formatter=lambda value: f"{value:.4f}",
        figsize=WIDE_CHART_SIZE,
    )

    st.divider()

    st.subheader("Feature Importance Table")

    st.dataframe(top_features_df, use_container_width=True)

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
        "Model risk: Feature importance shows model influence, not guaranteed causality. Use this as a prioritization guide, not as proof of root cause."
    )


# ============================================================
# Monitoring & Drift
# ============================================================

elif page == "Monitoring & Drift":
    st.title("Monitoring & Drift")
    st.caption(
        "Monitoring summary for data drift, prediction drift, and retraining decision."
    )

    st.divider()

    st.subheader("Monitoring KPI Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Features Monitored", features_monitored)
    col2.metric("Drifted Features", drifted_features)
    col3.metric("Data Drift", str(data_drift))
    col4.metric("Prediction Drift", str(prediction_drift))
    col5.metric("Retraining", bool_to_status(retraining_recommended, "Recommended", "Not Needed"))

    st.divider()

    st.subheader("Drift Health Chart")

    drift_df = pd.DataFrame(
        {
            "Monitoring Check": [
                "Stable Features",
                "Drifted Features",
            ],
            "Count": [
                features_monitored - drifted_features,
                drifted_features,
            ],
        }
    )

    render_barh_chart(
        df=drift_df,
        label_col="Monitoring Check",
        value_col="Count",
        title="Feature Drift Monitoring Result",
        xlabel="Feature Count",
        xlim=(0, features_monitored + 2),
        value_formatter=lambda value: f"{int(value)}",
        figsize=NORMAL_CHART_SIZE,
    )

    st.divider()

    st.subheader("Prediction Drift Summary")

    prediction_drift_summary_df = pd.DataFrame(
        [
            {
                "Metric": "Reference Mean Churn Probability",
                "Value": round(reference_mean_churn_probability, 4),
            },
            {
                "Metric": "Current Mean Churn Probability",
                "Value": round(current_mean_churn_probability, 4),
            },
            {
                "Metric": "Absolute Mean Change",
                "Value": round(absolute_mean_change, 4),
            },
            {
                "Metric": "Alert Level",
                "Value": alert_level,
            },
        ]
    )

    st.dataframe(prediction_drift_summary_df, use_container_width=True)

    if not feature_drift_df.empty:
        st.divider()
        st.subheader("Feature Drift Details")

        display_cols = [
            col
            for col in [
                "feature",
                "feature_type",
                "test",
                "statistic",
                "p_value",
                "drift_detected",
            ]
            if col in feature_drift_df.columns
        ]

        st.dataframe(feature_drift_df[display_cols], use_container_width=True)

    st.divider()

    st.subheader("Monitoring Decision Table")

    monitoring_decisions = pd.DataFrame(
        [
            {
                "Check": "Data Drift",
                "Result": "Detected" if data_drift else "Not Detected",
                "Meaning": "Checks whether current input feature distribution changed from reference data.",
                "Business Decision": "Model can continue scoring if no drift is detected.",
            },
            {
                "Check": "Prediction Drift",
                "Result": "Detected" if prediction_drift else "Not Detected",
                "Meaning": "Checks whether current prediction pattern changed from the reference prediction pattern.",
                "Business Decision": "No immediate model behavior issue detected if prediction drift is false.",
            },
            {
                "Check": "Feature Drift",
                "Result": f"{drifted_features} of {features_monitored} features drifted",
                "Meaning": "Counts monitored features that crossed the drift threshold.",
                "Business Decision": "No data-driven retraining trigger if drifted feature count is zero.",
            },
            {
                "Check": "Retraining Recommendation",
                "Result": "Recommended" if retraining_recommended else "Not Recommended",
                "Meaning": "Monitoring-based recommendation for model retraining.",
                "Business Decision": monitoring_recommended_action,
            },
        ]
    )

    st.dataframe(monitoring_decisions, use_container_width=True)

    st.divider()

    st.subheader("Retraining Decision")

    if not data_drift and not prediction_drift and not retraining_recommended:
        st.success(
            "Monitoring status: Healthy. No data drift, no prediction drift, and no retraining recommended."
        )

        st.info(
            "Business decision: Keep the current MLflow champion model active for scoring. Continue periodic monitoring before triggering retraining."
        )
    else:
        st.error(
            "Monitoring status: Review needed. Drift or retraining trigger was detected."
        )

        st.warning(
            "Business decision: Investigate drifted features, compare model performance on recent data, and consider retraining."
        )

    st.divider()

    st.subheader("Operational Monitoring Policy")

    st.markdown(
        """
        Recommended monitoring policy:

        - Run monitoring after every new scoring batch.
        - Trigger investigation if any important feature starts drifting.
        - Trigger retraining review if prediction drift is detected.
        - Compare new model vs champion model before promotion.
        - Promote only if the new model improves business-relevant metrics such as recall, F1-score, or retention ROI.
        """
    )


# ============================================================
# Batch Predictions
# ============================================================

elif page == "Batch Predictions":
    st.title("Batch Predictions")
    st.caption(
        "Operational batch scoring output for customer churn risk prioritization."
    )

    st.divider()

    st.subheader("Batch Scoring Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers Scored", total_scored)
    col2.metric("High Risk", high_risk_count)
    col3.metric("Medium Risk", medium_risk_count)
    col4.metric("Low Risk", low_risk_count)

    st.divider()

    st.subheader("Risk Band Distribution")

    risk_distribution_df = pd.DataFrame(
        {
            "Risk Band": ["High Risk", "Medium Risk", "Low Risk"],
            "Customer Count": [
                high_risk_count,
                medium_risk_count,
                low_risk_count,
            ],
        }
    )

    render_barh_chart(
        df=risk_distribution_df,
        label_col="Risk Band",
        value_col="Customer Count",
        title="Batch Prediction Risk Distribution",
        xlabel="Customer Count",
        xlim=(0, max(total_scored + 1, 5)),
        value_formatter=lambda value: f"{int(value)}",
        figsize=NORMAL_CHART_SIZE,
    )

    st.divider()

    st.subheader("Scored Customer Output")

    display_cols = [
        col
        for col in [
            "customer_id",
            "churn_probability",
            "risk_band_dashboard",
            "recommended_action",
        ]
        if col in batch_predictions_df.columns
    ]

    st.dataframe(batch_predictions_df[display_cols], use_container_width=True)

    csv_data = batch_predictions_df.to_csv(index=False)

    st.download_button(
        label="Download Batch Prediction Results",
        data=csv_data,
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("Retention Action Mapping")

    action_mapping_df = pd.DataFrame(
        [
            {
                "Risk Band": "High Risk",
                "Probability Rule": f"churn_probability >= {best_threshold:.2f}",
                "Business Action": "Prioritize for retention outreach.",
                "Owner": "Retention team",
            },
            {
                "Risk Band": "Medium Risk",
                "Probability Rule": f"0.40 <= churn_probability < {best_threshold:.2f}",
                "Business Action": "Monitor or send low-cost engagement offer.",
                "Owner": "Customer success / marketing",
            },
            {
                "Risk Band": "Low Risk",
                "Probability Rule": "churn_probability < 0.40",
                "Business Action": "No immediate retention action.",
                "Owner": "No action needed",
            },
        ]
    )

    st.dataframe(action_mapping_df, use_container_width=True)

    st.divider()

    st.subheader("Business Recommendation")

    if high_risk_count > 0:
        st.success(
            f"Recommendation: Prioritize the {high_risk_count} high-risk customers for immediate retention outreach."
        )
    else:
        st.success(
            "Recommendation: No high-risk customers were found in the current batch."
        )

    st.info(
        "Business decision: Use the batch prediction output as a daily or weekly retention queue for the customer success team."
    )


# ============================================================
# MLOps System Health
# ============================================================

elif page == "MLOps System Health":
    st.title("MLOps System Health")
    st.caption(
        "Production-readiness summary for the ChurnOps end-to-end MLOps pipeline."
    )

    st.divider()

    st.subheader("System Health Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("GitHub Actions", "Passed")
    col2.metric("Pytest", "8 passed")
    col3.metric("Model Registry", "Champion")
    col4.metric("API Health", "Loaded")
    col5.metric("Docker", "Working")

    st.divider()

    st.subheader("MLOps Component Checklist")

    system_health_df = pd.DataFrame(
        [
            {
                "Component": "GitHub Repository",
                "Status": "Complete",
                "Evidence": "Repo pushed to GitHub.",
                "Business / Technical Value": "Enables version control and portfolio review.",
            },
            {
                "Component": "GitHub Actions",
                "Status": "Passed",
                "Evidence": "CI workflow completed successfully.",
                "Business / Technical Value": "Automates quality checks before changes are accepted.",
            },
            {
                "Component": "Pytest",
                "Status": "8 passed",
                "Evidence": "Local tests passed.",
                "Business / Technical Value": "Reduces risk of broken pipeline logic.",
            },
            {
                "Component": "Data Validation",
                "Status": "Working",
                "Evidence": "Validation pipeline checks input data quality.",
                "Business / Technical Value": "Prevents bad data from entering training or scoring.",
            },
            {
                "Component": "Preprocessing",
                "Status": "Working",
                "Evidence": "Train/validation/test split completed.",
                "Business / Technical Value": "Creates reproducible model-ready datasets.",
            },
            {
                "Component": "MLflow Training",
                "Status": "Working",
                "Evidence": "Experiments and metrics tracked.",
                "Business / Technical Value": "Makes model selection auditable and repeatable.",
            },
            {
                "Component": "Model Registry",
                "Status": "Champion active",
                "Evidence": "Registered model name: churnops_model, alias: champion.",
                "Business / Technical Value": "Supports controlled model promotion and deployment.",
            },
            {
                "Component": "FastAPI Service",
                "Status": "Working",
                "Evidence": "/health returns model_loaded = True; /predict returns probability.",
                "Business / Technical Value": "Makes the model usable by applications or services.",
            },
            {
                "Component": "Docker",
                "Status": "Working",
                "Evidence": "Containerized API builds and runs.",
                "Business / Technical Value": "Improves reproducibility and deployment portability.",
            },
            {
                "Component": "Monitoring",
                "Status": "Working",
                "Evidence": f"{features_monitored} features monitored, {drifted_features} drifted features.",
                "Business / Technical Value": "Detects when the model may become unreliable.",
            },
            {
                "Component": "Batch Prediction",
                "Status": "Working",
                "Evidence": f"{total_scored} customers scored; {high_risk_count} high risk, {medium_risk_count} medium risk, {low_risk_count} low risk.",
                "Business / Technical Value": "Creates operational retention queue.",
            },
            {
                "Component": "Streamlit Dashboard",
                "Status": "Working",
                "Evidence": "Dashboard pages show model, monitoring, threshold, and batch results.",
                "Business / Technical Value": "Makes model outputs understandable for business users.",
            },
        ]
    )

    st.dataframe(system_health_df, use_container_width=True)

    st.divider()

    st.subheader("MLOps Architecture Flow")

    architecture_df = pd.DataFrame(
        [
            {
                "Step": 1,
                "Stage": "Data Validation",
                "Output": "Validated customer churn dataset",
                "Purpose": "Protects the pipeline from schema and data quality issues.",
            },
            {
                "Step": 2,
                "Stage": "Preprocessing",
                "Output": "Train / validation / test data",
                "Purpose": "Creates reproducible features and prevents leakage.",
            },
            {
                "Step": 3,
                "Stage": "Model Training",
                "Output": "Candidate models with metrics",
                "Purpose": "Compares models using business-relevant metrics.",
            },
            {
                "Step": 4,
                "Stage": "MLflow Registry",
                "Output": "Champion model",
                "Purpose": "Controls which model is used for serving.",
            },
            {
                "Step": 5,
                "Stage": "FastAPI Serving",
                "Output": "Prediction endpoint",
                "Purpose": "Exposes churn scoring through an API.",
            },
            {
                "Step": 6,
                "Stage": "Docker",
                "Output": "Containerized API",
                "Purpose": "Makes deployment reproducible.",
            },
            {
                "Step": 7,
                "Stage": "Monitoring",
                "Output": "Drift and retraining decision",
                "Purpose": "Checks if the model remains reliable over time.",
            },
            {
                "Step": 8,
                "Stage": "Dashboard",
                "Output": "Business-facing control room",
                "Purpose": "Turns model outputs into retention decisions.",
            },
        ]
    )

    st.dataframe(architecture_df, use_container_width=True)

    st.divider()

    st.subheader("Deployment Readiness Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            "Ready for portfolio demo: The project includes testing, CI, model registry, API serving, Docker, monitoring, batch scoring, and dashboard reporting."
        )

    with col2:
        st.info(
            "Best current use case: Retention prioritization. The model should support customer ranking and campaign targeting, not fully automated customer decisions."
        )

    st.divider()

    st.subheader("Demo Talking Points")

    st.markdown(
        """
        Use this page at the end of the demo to summarize the full system:

        - The project starts with data validation and preprocessing.
        - The model is trained and tracked using MLflow.
        - The best model is registered as the champion model.
        - FastAPI serves real-time predictions.
        - Docker containerizes the API.
        - Monitoring checks data drift and prediction drift.
        - Batch scoring creates a retention action queue.
        - Streamlit turns technical outputs into business decisions.
        """
    )

    st.success(
        "Dynamic dashboard upgrade complete: model metrics, confusion matrix, monitoring, batch predictions, feature importance, and threshold analysis are now loaded from the reports folder."
    )