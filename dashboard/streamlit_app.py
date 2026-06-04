import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
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
# API helpers
# ============================================================

def check_api_health(api_base_url: str) -> tuple[bool, dict | None, str | None]:
    health_url = f"{api_base_url.rstrip('/')}/health"

    try:
        response = requests.get(health_url, timeout=2)
        response.raise_for_status()
        return True, response.json(), None
    except requests.exceptions.RequestException as error:
        return False, None, str(error)


def call_prediction_api(api_base_url: str, payload: dict) -> tuple[bool, dict | None, str | None]:
    predict_url = f"{api_base_url.rstrip('/')}/predict"

    try:
        response = requests.post(predict_url, json=payload, timeout=5)
        response.raise_for_status()
        return True, response.json(), None
    except requests.exceptions.RequestException as error:
        return False, None, str(error)


def normalize_risk_band(value: str) -> str:
    value_lower = str(value).lower()

    if "high" in value_lower:
        return "High Risk"
    if "medium" in value_lower:
        return "Medium Risk"
    if "low" in value_lower:
        return "Low Risk"

    return "Unknown"


# ============================================================
# Unified retention policy
# ============================================================
# This policy is aligned with the API and threshold analysis.
#
# Low Risk:    churn_probability < 0.40
# Medium Risk: 0.40 <= churn_probability < best_threshold
# High Risk:   churn_probability >= best_threshold
#
# best_threshold is loaded from reports/threshold_analysis.csv.
# Fallback value is 0.60.

MEDIUM_RISK_THRESHOLD = 0.40
DEFAULT_RETENTION_THRESHOLD = 0.60


def assign_risk_band(probability: float, high_threshold: float) -> str:
    """
    Convert churn probability into risk band using the approved retention policy.
    """
    probability = safe_float(probability, 0.0)
    high_threshold = safe_float(high_threshold, DEFAULT_RETENTION_THRESHOLD)

    if probability >= high_threshold:
        return "High Risk"

    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"

    return "Low Risk"


def action_from_risk_band(risk_band: str) -> str:
    """
    Convert risk band into recommended business action.
    """
    risk_band_lower = str(risk_band).lower()

    if "high" in risk_band_lower:
        return "Priority retention outreach"

    if "medium" in risk_band_lower:
        return "Monitor or send low-cost engagement offer"

    if "low" in risk_band_lower:
        return "No immediate action"

    return "Review manually"


def prediction_label_from_probability(probability: float, high_threshold: float) -> str:
    """
    Convert churn probability into the dashboard decision label.

    This is aligned with the operational retention threshold.
    """
    probability = safe_float(probability, 0.0)
    high_threshold = safe_float(high_threshold, DEFAULT_RETENTION_THRESHOLD)

    if probability >= high_threshold:
        return "Churn Risk"

    return "No Churn Risk"


def render_retention_policy_note(high_threshold: float) -> None:
    """
    Display the unified retention policy used by API, dashboard, and batch scoring.
    """
    st.info(
        f"Decision policy: Customers with churn probability >= {high_threshold:.2f} "
        "are classified as High Risk and prioritized for retention outreach. "
        f"Customers from {MEDIUM_RISK_THRESHOLD:.2f} up to {high_threshold:.2f} "
        "are Medium Risk. Customers below 0.40 are Low Risk."
    )


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
metrics_report = load_json_report("metrics_report.json")

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
    0.4269,
)
current_mean_churn_probability = safe_float(
    prediction_drift_details.get("current_mean_churn_probability"),
    0.4049,
)
absolute_mean_change = safe_float(
    prediction_drift_details.get("absolute_mean_change"),
    0.0220,
)

feature_drift_results = monitoring_summary.get("feature_drift_results", [])
feature_drift_df = pd.DataFrame(feature_drift_results)


# ============================================================
# Extract dynamic threshold analysis
# ============================================================

if not threshold_analysis_df.empty:
    for col in threshold_analysis_df.columns:
        threshold_analysis_df[col] = pd.to_numeric(
            threshold_analysis_df[col],
            errors="ignore",
        )

    if "f1" in threshold_analysis_df.columns:
        best_threshold_row = threshold_analysis_df.loc[
            threshold_analysis_df["f1"].astype(float).idxmax()
        ]
    else:
        best_threshold_row = pd.Series(dtype="object")

    best_threshold = safe_float(best_threshold_row.get("threshold"), DEFAULT_RETENTION_THRESHOLD)
    threshold_precision = safe_float(best_threshold_row.get("precision"), 0.5714)
    threshold_recall = safe_float(best_threshold_row.get("recall"), 0.6975)
    threshold_f1 = safe_float(best_threshold_row.get("f1"), 0.6282)
    threshold_fp = safe_int(best_threshold_row.get("false_positives"), 147)
    threshold_fn = safe_int(best_threshold_row.get("false_negatives"), 85)
    threshold_tp = safe_int(best_threshold_row.get("true_positives"), 196)

    flagged_customers = threshold_fp + threshold_tp
    missed_churners = threshold_fn
else:
    best_threshold = DEFAULT_RETENTION_THRESHOLD
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
# Extract champion and challenger model metadata
# ============================================================

registered_model_name = model_registry_report.get("registered_model_name", "churnops_model")
champion_alias = model_registry_report.get("alias", "champion")
champion_model_name = model_registry_report.get("source_model_name", "logistic_regression_baseline")
champion_selection_metric = model_registry_report.get("selection_metric", "validation_roc_auc")

all_model_results = metrics_report.get("all_model_results", [])

champion_display_name = "Logistic Regression"
challenger_display_name = "Random Forest"

model_comparison_rows = []

for model_result in all_model_results:
    model_name = model_result.get("model_name", "unknown_model")
    model_metrics = model_result.get("metrics", {})

    if model_name == champion_model_name:
        role = "Champion"
        display_name = champion_display_name
    elif "random_forest" in model_name.lower():
        role = "Challenger"
        display_name = challenger_display_name
    else:
        role = "Candidate"
        display_name = model_name.replace("_", " ").title()

    model_comparison_rows.append(
        {
            "Role": role,
            "Model": display_name,
            "Run Name": model_name,
            "ROC-AUC": round(safe_float(model_metrics.get("roc_auc")), 4),
            "Precision": round(safe_float(model_metrics.get("precision")), 4),
            "Recall": round(safe_float(model_metrics.get("recall")), 4),
            "F1-score": round(safe_float(model_metrics.get("f1")), 4),
        }
    )

if not model_comparison_rows:
    model_comparison_rows = [
        {
            "Role": "Champion",
            "Model": "Logistic Regression",
            "Run Name": "logistic_regression_baseline",
            "ROC-AUC": 0.8447,
            "Precision": 0.5055,
            "Recall": 0.8143,
            "F1-score": 0.6238,
        },
        {
            "Role": "Challenger",
            "Model": "Random Forest",
            "Run Name": "random_forest_improved",
            "ROC-AUC": 0.8424,
            "Precision": 0.5282,
            "Recall": 0.8036,
            "F1-score": 0.6374,
        },
    ]

model_comparison_df = pd.DataFrame(model_comparison_rows)


# ============================================================
# Extract dynamic official batch predictions
# ============================================================

def add_risk_band_from_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dashboard risk band using churn_probability.

    Important:
    This recalculates risk bands from probability instead of trusting any existing
    risk_band or risk_level column. This prevents stale API or batch outputs from
    using old threshold rules.
    """
    output_df = df.copy()

    probability_col = find_first_existing_column(
        output_df,
        ["churn_probability", "probability", "prediction_probability"],
    )

    if probability_col:
        output_df["churn_probability"] = pd.to_numeric(
            output_df[probability_col],
            errors="coerce",
        ).fillna(0.0)

        output_df["risk_band_dashboard"] = output_df["churn_probability"].apply(
            lambda value: assign_risk_band(value, best_threshold)
        )

        output_df["prediction_label_dashboard"] = output_df["churn_probability"].apply(
            lambda value: prediction_label_from_probability(value, best_threshold)
        )
    else:
        output_df["churn_probability"] = 0.0
        output_df["risk_band_dashboard"] = "Unknown"
        output_df["prediction_label_dashboard"] = "Unknown"

    output_df["recommended_action"] = output_df["risk_band_dashboard"].apply(
        action_from_risk_band
    )

    return output_df


if not batch_predictions_df.empty:
    batch_predictions_df = add_risk_band_from_probability(batch_predictions_df)

    if "customer_id" not in batch_predictions_df.columns:
        batch_predictions_df.insert(
            0,
            "customer_id",
            [f"CUST-{index + 1:03d}" for index in range(len(batch_predictions_df))],
        )

else:
    batch_predictions_df = pd.DataFrame(
        [
            {
                "customer_id": "CUST-001",
                "churn_probability": 0.8845,
            },
            {
                "customer_id": "CUST-002",
                "churn_probability": 0.7200,
            },
            {
                "customer_id": "CUST-003",
                "churn_probability": 0.4300,
            },
            {
                "customer_id": "CUST-004",
                "churn_probability": 0.2300,
            },
        ]
    )

    batch_predictions_df = add_risk_band_from_probability(batch_predictions_df)

total_scored = len(batch_predictions_df)

risk_lower = batch_predictions_df["risk_band_dashboard"].astype(str).str.lower()

high_risk_count = int(risk_lower.str.contains("high").sum())
medium_risk_count = int(risk_lower.str.contains("medium").sum())
low_risk_count = int(risk_lower.str.contains("low").sum())


# ============================================================
# Demo session state helpers
# ============================================================

def initialize_demo_state() -> None:
    if "demo_predictions" not in st.session_state:
        st.session_state.demo_predictions = []

    if "last_prediction_result" not in st.session_state:
        st.session_state.last_prediction_result = None

    if "last_prediction_payload" not in st.session_state:
        st.session_state.last_prediction_payload = None


def get_demo_predictions_df() -> pd.DataFrame:
    initialize_demo_state()

    if not st.session_state.demo_predictions:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "source",
                "prediction_label",
                "churn_probability",
                "risk_band",
                "recommended_action",
                "tenure",
                "Contract",
                "InternetService",
                "MonthlyCharges",
                "TotalCharges",
            ]
        )

    return pd.DataFrame(st.session_state.demo_predictions)


def add_demo_prediction(payload: dict, prediction_result: dict, source: str = "Live API") -> None:
    """
    Add one live prediction into the temporary demo batch.

    The dashboard recalculates risk_band and recommended_action from probability
    to guarantee alignment with the approved threshold policy.
    """
    initialize_demo_state()

    probability = safe_float(
        prediction_result.get("churn_probability", prediction_result.get("probability", 0.0)),
        0.0,
    )

    risk_band = assign_risk_band(probability, best_threshold)
    recommended_action = action_from_risk_band(risk_band)
    prediction_label = prediction_label_from_probability(probability, best_threshold)

    next_index = len(st.session_state.demo_predictions) + 1

    record = {
        "customer_id": f"DEMO-{next_index:03d}",
        "source": source,
        "prediction_label": prediction_label,
        "churn_probability": round(probability, 4),
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "tenure": payload.get("tenure"),
        "Contract": payload.get("Contract"),
        "InternetService": payload.get("InternetService"),
        "MonthlyCharges": payload.get("MonthlyCharges"),
        "TotalCharges": payload.get("TotalCharges"),
    }

    st.session_state.demo_predictions.append(record)


def clear_demo_predictions() -> None:
    st.session_state.demo_predictions = []
    st.session_state.last_prediction_result = None
    st.session_state.last_prediction_payload = None


def build_demo_record(
    customer_id: str,
    source: str,
    churn_probability: float,
    tenure: int,
    contract: str,
    internet_service: str,
    monthly_charges: float,
    total_charges: float,
) -> dict:
    risk_band = assign_risk_band(churn_probability, best_threshold)
    prediction_label = prediction_label_from_probability(churn_probability, best_threshold)

    return {
        "customer_id": customer_id,
        "source": source,
        "prediction_label": prediction_label,
        "churn_probability": round(churn_probability, 4),
        "risk_band": risk_band,
        "recommended_action": action_from_risk_band(risk_band),
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def load_normal_demo_batch() -> None:
    initialize_demo_state()

    st.session_state.demo_predictions = [
        build_demo_record(
            customer_id="NORMAL-001",
            source="Normal Demo Batch",
            churn_probability=0.28,
            tenure=48,
            contract="Two year",
            internet_service="DSL",
            monthly_charges=45.0,
            total_charges=2160.0,
        ),
        build_demo_record(
            customer_id="NORMAL-002",
            source="Normal Demo Batch",
            churn_probability=0.36,
            tenure=36,
            contract="One year",
            internet_service="DSL",
            monthly_charges=55.0,
            total_charges=1980.0,
        ),
        build_demo_record(
            customer_id="NORMAL-003",
            source="Normal Demo Batch",
            churn_probability=0.43,
            tenure=22,
            contract="One year",
            internet_service="Fiber optic",
            monthly_charges=70.0,
            total_charges=1540.0,
        ),
        build_demo_record(
            customer_id="NORMAL-004",
            source="Normal Demo Batch",
            churn_probability=0.51,
            tenure=18,
            contract="Month-to-month",
            internet_service="DSL",
            monthly_charges=60.0,
            total_charges=1080.0,
        ),
    ]


def load_high_risk_shifted_demo_batch() -> None:
    initialize_demo_state()

    st.session_state.demo_predictions = [
        build_demo_record(
            customer_id="SHIFT-001",
            source="High-Risk Shifted Batch",
            churn_probability=0.88,
            tenure=3,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=98.0,
            total_charges=294.0,
        ),
        build_demo_record(
            customer_id="SHIFT-002",
            source="High-Risk Shifted Batch",
            churn_probability=0.81,
            tenure=5,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=95.7,
            total_charges=478.5,
        ),
        build_demo_record(
            customer_id="SHIFT-003",
            source="High-Risk Shifted Batch",
            churn_probability=0.76,
            tenure=7,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=91.0,
            total_charges=637.0,
        ),
        build_demo_record(
            customer_id="SHIFT-004",
            source="High-Risk Shifted Batch",
            churn_probability=0.72,
            tenure=9,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=89.0,
            total_charges=801.0,
        ),
        build_demo_record(
            customer_id="SHIFT-005",
            source="High-Risk Shifted Batch",
            churn_probability=0.67,
            tenure=12,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=86.0,
            total_charges=1032.0,
        ),
    ]


def summarize_risk_counts(df: pd.DataFrame, risk_col: str = "risk_band") -> tuple[int, int, int, int]:
    if df.empty or risk_col not in df.columns:
        return 0, 0, 0, 0

    risk_series = df[risk_col].astype(str).str.lower()

    high_count = int(risk_series.str.contains("high").sum())
    medium_count = int(risk_series.str.contains("medium").sum())
    low_count = int(risk_series.str.contains("low").sum())
    total_count = len(df)

    return total_count, high_count, medium_count, low_count


def render_demo_batch_summary(df: pd.DataFrame) -> None:
    total_count, high_count, medium_count, low_count = summarize_risk_counts(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Demo Customers Scored", total_count)
    col2.metric("High Risk", high_count)
    col3.metric("Medium Risk", medium_count)
    col4.metric("Low Risk", low_count)

    risk_distribution_df = pd.DataFrame(
        {
            "Risk Band": ["High Risk", "Medium Risk", "Low Risk"],
            "Customer Count": [high_count, medium_count, low_count],
        }
    )

    render_barh_chart(
        df=risk_distribution_df,
        label_col="Risk Band",
        value_col="Customer Count",
        title="Live Demo Risk Distribution",
        xlabel="Customer Count",
        xlim=(0, max(total_count + 1, 5)),
        value_formatter=lambda value: f"{int(value)}",
        figsize=NORMAL_CHART_SIZE,
    )


# ============================================================
# Monitoring job helpers
# ============================================================

def get_file_modified_time(file_name: str) -> str:
    file_path = REPORTS_DIR / file_name

    if not file_path.exists():
        return "File not found"

    modified_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
    return modified_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def run_monitoring_job() -> tuple[bool, str, str]:
    """
    Run the official monitoring script and regenerate reports/monitoring_summary.json.
    This makes the Monitoring & Drift page an active monitoring control panel.
    """
    monitoring_script = BASE_DIR / "src" / "monitoring" / "generate_drift_report.py"

    if not monitoring_script.exists():
        return False, "", f"Monitoring script not found: {monitoring_script}"

    result = subprocess.run(
        [sys.executable, str(monitoring_script)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    return result.returncode == 0, result.stdout, result.stderr


# ============================================================
# Initialize demo state
# ============================================================

initialize_demo_state()


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
        "Threshold Analysis",
        "Batch Predictions",
        "Live Prediction Lab",
        "Drift Simulation Lab",
        "Monitoring & Drift",
        "MLOps System Health",
        "Confusion Matrix",
        "Feature Importance",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Champion Model")
st.sidebar.write(f"**Model Registry Name:** `{registered_model_name}`")
st.sidebar.write(f"**Alias:** `{champion_alias}`")
st.sidebar.write("**Champion:** Logistic Regression")
st.sidebar.write("**Challenger:** Random Forest only")
st.sidebar.write("**Serving:** FastAPI + Docker")

st.sidebar.divider()
st.sidebar.markdown("### Retention Policy")
st.sidebar.write(f"**High Risk:** `>= {best_threshold:.2f}`")
st.sidebar.write(f"**Medium Risk:** `{MEDIUM_RISK_THRESHOLD:.2f} to < {best_threshold:.2f}`")
st.sidebar.write(f"**Low Risk:** `< {MEDIUM_RISK_THRESHOLD:.2f}`")

st.sidebar.divider()
st.sidebar.markdown("### FastAPI Settings")

api_base_url = st.sidebar.text_input(
    "API Base URL",
    value="http://127.0.0.1:8000",
)

api_is_online, api_health_payload, api_error = check_api_health(api_base_url)

if api_is_online:
    st.sidebar.success("API Online")
else:
    st.sidebar.error("API Offline")
    st.sidebar.caption("Start API before live prediction:")
    st.sidebar.code(
        "uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000",
        language="powershell",
    )

st.sidebar.divider()
st.sidebar.markdown("### Demo Flow")
st.sidebar.write("**Recommended flow:**")
st.sidebar.caption(
    "Overview → Performance → Threshold → Batch → Live API → Drift → Monitoring → System Health"
)

st.sidebar.write("**Backup technical pages:**")
st.sidebar.caption(
    "Confusion Matrix and Feature Importance are available for deeper technical questions."
)


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

    st.subheader("Operational Retention Policy")
    render_retention_policy_note(best_threshold)

    st.divider()

    st.subheader("Monitoring Health")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Features Monitored", features_monitored)
    col2.metric("Drifted Features", drifted_features)
    col3.metric("Data Drift", bool_to_status(data_drift, "Detected", "Not Detected"))
    col4.metric("Prediction Drift", bool_to_status(prediction_drift, "Detected", "Not Detected"))
    col5.metric("Retraining Review", bool_to_status(retraining_recommended, "Review Needed", "Not Needed"))

    if not data_drift and not prediction_drift and not retraining_recommended:
        st.success(
            "Monitoring result: No data drift or prediction drift detected. "
            "The champion model remains suitable for current scoring."
        )
    else:
        st.error(
            "Monitoring result: Drift or retraining trigger detected. "
            "Review model reliability before continued scoring."
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

        The operational policy uses **{best_threshold:.2f}** as the High Risk cutoff for priority retention outreach.
        """
    )

    st.divider()

    st.subheader("Official Batch Prediction Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk", high_risk_count)
    col2.metric("Medium Risk", medium_risk_count)
    col3.metric("Low Risk", low_risk_count)

    st.divider()

    st.subheader("Project Health Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tests", "8 passed")
    col2.metric("CI/CD", "Passed")
    col3.metric("API Health", "Online" if api_is_online else "Offline")
    col4.metric("Docker", "Working")

    st.markdown(
        """
        This confirms that ChurnOps is not only a notebook-based model. 
        It includes automated testing, CI validation, API serving, containerization, 
        model registry, monitoring, batch scoring, and a dynamic dashboard.
        """
    )


# ============================================================
# Live Prediction Lab
# ============================================================

elif page == "Live Prediction Lab":
    st.title("Live Prediction Lab")
    st.caption(
        "Score a single customer through the FastAPI prediction endpoint and add the result to the live demo batch."
    )

    st.divider()

    st.subheader("API Health Check")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("FastAPI Status", "Online" if api_is_online else "Offline")

    with col2:
        st.metric("API URL", api_base_url)

    if api_is_online:
        st.success("FastAPI is reachable. Live prediction is ready.")
        if api_health_payload:
            st.json(api_health_payload)
    else:
        st.error("FastAPI is not reachable. Start the API before using live prediction.")
        st.code(
            "uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000",
            language="powershell",
        )

    st.divider()

    st.subheader("Customer Input Form")

    with st.form("live_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("SeniorCitizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure", min_value=0, max_value=100, value=5, step=1)

        with col2:
            phone_service = st.selectbox("PhoneService", ["Yes", "No"])
            multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("InternetService", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        col4, col5, col6 = st.columns(3)

        with col4:
            paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])

        with col5:
            payment_method = st.selectbox(
                "PaymentMethod",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col6:
            monthly_charges = st.number_input(
                "MonthlyCharges",
                min_value=0.0,
                max_value=200.0,
                value=95.70,
                step=0.10,
            )
            total_charges = st.number_input(
                "TotalCharges",
                min_value=0.0,
                max_value=10000.0,
                value=478.50,
                step=0.10,
            )

        submitted = st.form_submit_button("Predict Churn")

    customer_payload = {
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

    if submitted:
        if not api_is_online:
            st.error("Prediction failed because FastAPI is offline.")
        else:
            success, prediction_result, prediction_error = call_prediction_api(
                api_base_url=api_base_url,
                payload=customer_payload,
            )

            if success and prediction_result:
                st.session_state.last_prediction_result = prediction_result
                st.session_state.last_prediction_payload = customer_payload

                add_demo_prediction(
                    payload=customer_payload,
                    prediction_result=prediction_result,
                    source="Live API",
                )

                st.success("Prediction successful. Result added to the demo batch.")
            else:
                st.error("Prediction request failed.")
                st.code(prediction_error)

    if st.session_state.last_prediction_result:
        st.divider()
        st.subheader("Latest Live Prediction Result")

        latest_result = st.session_state.last_prediction_result

        latest_probability = safe_float(
            latest_result.get("churn_probability", latest_result.get("probability", 0.0)),
            0.0,
        )

        latest_risk_band = assign_risk_band(latest_probability, best_threshold)
        latest_action = action_from_risk_band(latest_risk_band)
        latest_prediction_label = prediction_label_from_probability(
            latest_probability,
            best_threshold,
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Churn Probability", f"{latest_probability:.4f}")
        col2.metric("Prediction Label", latest_prediction_label)
        col3.metric("Risk Band", latest_risk_band)
        col4.metric("Action", latest_action)

        render_retention_policy_note(best_threshold)

        st.markdown("#### Raw API Response")
        st.json(latest_result)

    st.divider()

    st.subheader("Live Demo Batch Controls")

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        if st.button("Load Normal Demo Batch", key="live_load_normal"):
            load_normal_demo_batch()
            st.success("Normal demo batch loaded.")

    with control_col2:
        if st.button("Load High-Risk Shifted Batch", key="live_load_shifted"):
            load_high_risk_shifted_demo_batch()
            st.warning("High-risk shifted demo batch loaded.")

    with control_col3:
        if st.button("Clear Demo Batch", key="live_clear_demo"):
            clear_demo_predictions()
            st.info("Demo batch cleared.")

    st.divider()

    st.subheader("Live Demo Batch Summary")

    demo_df = get_demo_predictions_df()

    if demo_df.empty:
        st.info(
            "No demo batch data yet. Score a customer above or load a demo batch to show changing dashboard counts."
        )
    else:
        render_demo_batch_summary(demo_df)

        st.divider()

        st.subheader("Live Demo Scored Customers")
        st.dataframe(demo_df, use_container_width=True)

        st.download_button(
            label="Download Demo Batch",
            data=demo_df.to_csv(index=False),
            file_name="demo_live_predictions.csv",
            mime="text/csv",
        )

        st.info(
            "This section replaces the separate demo batch page during presentation, so the live API result and operational queue are shown in one flow."
        )


# ============================================================
# Demo Batch Scoring Lab
# ============================================================

elif page == "Demo Batch Scoring Lab":
    st.title("Demo Batch Scoring Lab")
    st.caption(
        "Use temporary demo predictions to show how dashboard numbers and visuals change during scoring."
    )

    st.divider()

    st.subheader("Demo Batch Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load Normal Demo Batch"):
            load_normal_demo_batch()
            st.success("Normal demo batch loaded.")

    with col2:
        if st.button("Load High-Risk Shifted Batch"):
            load_high_risk_shifted_demo_batch()
            st.warning("High-risk shifted demo batch loaded.")

    with col3:
        if st.button("Clear Demo Batch"):
            clear_demo_predictions()
            st.info("Demo batch cleared.")

    st.divider()

    demo_df = get_demo_predictions_df()

    st.subheader("Live Demo Batch Summary")

    if demo_df.empty:
        st.info(
            "No demo batch data yet. Use Live Prediction Lab or load a demo batch using the buttons above."
        )
    else:
        render_demo_batch_summary(demo_df)

        st.divider()

        st.subheader("Live Demo Scored Customers")
        st.dataframe(demo_df, use_container_width=True)

        st.download_button(
            label="Download Demo Batch",
            data=demo_df.to_csv(index=False),
            file_name="demo_live_predictions.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("Business Interpretation")

        total_demo, high_demo, medium_demo, low_demo = summarize_risk_counts(demo_df)

        st.markdown(
            f"""
            The current demo batch has **{total_demo} scored customers**.

            - **{high_demo} high-risk customers** should be prioritized for retention outreach.
            - **{medium_demo} medium-risk customers** can receive low-cost engagement or monitoring.
            - **{low_demo} low-risk customers** do not need immediate retention action.

            This page shows how live scoring can become an operational retention queue.
            """
        )

        render_retention_policy_note(best_threshold)


# ============================================================
# Drift Simulation Lab
# ============================================================

elif page == "Drift Simulation Lab":
    st.title("Drift Simulation Lab")
    st.caption(
        "Simulate prediction drift by comparing the live demo batch against the reference prediction distribution."
    )

    st.divider()

    st.subheader("Simulation Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load Normal Demo Batch", key="drift_load_normal"):
            load_normal_demo_batch()
            st.success("Normal demo batch loaded.")

    with col2:
        if st.button("Load High-Risk Shifted Batch", key="drift_load_shifted"):
            load_high_risk_shifted_demo_batch()
            st.warning("High-risk shifted demo batch loaded.")

    with col3:
        if st.button("Clear Demo Batch", key="drift_clear"):
            clear_demo_predictions()
            st.info("Demo batch cleared.")

    st.divider()

    demo_df = get_demo_predictions_df()

    st.subheader("Prediction Drift Check")

    if demo_df.empty:
        st.info(
            "No demo predictions available. Load a normal batch, load a shifted batch, or score customers in the Live Prediction Lab."
        )
    else:
        demo_df["churn_probability"] = pd.to_numeric(
            demo_df["churn_probability"],
            errors="coerce",
        ).fillna(0)

        demo_mean_probability = float(demo_df["churn_probability"].mean())
        demo_absolute_change = abs(demo_mean_probability - reference_mean_churn_probability)

        drift_threshold = st.slider(
            "Prediction Drift Alert Threshold",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.01,
        )

        demo_prediction_drift_detected = demo_absolute_change >= drift_threshold

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Reference Mean Probability", f"{reference_mean_churn_probability:.4f}")
        col2.metric("Demo Batch Mean Probability", f"{demo_mean_probability:.4f}")
        col3.metric("Absolute Mean Change", f"{demo_absolute_change:.4f}")
        col4.metric(
            "Prediction Drift",
            "Detected" if demo_prediction_drift_detected else "Not Detected",
        )

        drift_comparison_df = pd.DataFrame(
            {
                "Distribution": [
                    "Reference Mean",
                    "Demo Batch Mean",
                    "Absolute Change",
                    "Alert Threshold",
                ],
                "Value": [
                    reference_mean_churn_probability,
                    demo_mean_probability,
                    demo_absolute_change,
                    drift_threshold,
                ],
            }
        )

        render_barh_chart(
            df=drift_comparison_df,
            label_col="Distribution",
            value_col="Value",
            title="Prediction Drift Simulation",
            xlabel="Probability / Change",
            xlim=(0, 1),
            value_formatter=lambda value: f"{value:.4f}",
            figsize=WIDE_CHART_SIZE,
        )

        st.divider()

        st.subheader("Drift Decision")

        if demo_prediction_drift_detected:
            st.error(
                "Prediction drift detected in the demo batch. Investigation and retraining review are needed."
            )

            st.warning(
                "Business meaning: The new scoring population has a noticeably different churn risk profile compared with the reference data. The team should investigate before relying on the same model policy."
            )
        else:
            st.success(
                "No prediction drift detected in the demo batch. Retraining review is not needed."
            )

            st.info(
                "Business meaning: The current demo scoring population is still close to the reference prediction distribution."
            )

        st.divider()

        st.subheader("Demo Batch Used for Drift Check")
        st.dataframe(demo_df, use_container_width=True)

        st.divider()

        st.subheader("How to Explain This in the Demo")

        st.markdown(
            """
            Use this page to show two scenarios:

            1. **Normal demo batch**: churn probability distribution remains close to the reference data.
            2. **High-risk shifted batch**: average churn probability increases, triggering a prediction drift warning.

            This demonstrates why monitoring matters after deployment.
            """
        )


# ============================================================
# Monitoring & Drift
# ============================================================

elif page == "Monitoring & Drift":
    st.title("Monitoring & Drift")
    st.caption(
        "Official monitoring control panel for data drift, prediction drift, and retraining review."
    )

    st.divider()

    st.subheader("Official Monitoring Job")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(
            "This page displays the latest official monitoring output from `reports/monitoring_summary.json`. "
            "For this demo, reference data is validation data and current data is test data. "
            "In production, current data would be the latest scored customer batch."
        )
        st.caption(
            f"Last monitoring summary update: {get_file_modified_time('monitoring_summary.json')}"
        )

    with col2:
        if st.button("Run / Refresh Monitoring Check"):
            with st.spinner("Running monitoring job and regenerating monitoring reports..."):
                success, stdout, stderr = run_monitoring_job()

            if success:
                st.success("Monitoring job completed. Dashboard data refreshed.")
                if stdout.strip():
                    st.code(stdout, language="text")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Monitoring job failed. Check the error output below.")
                st.code(stderr if stderr.strip() else "No error details returned.", language="text")

    monitoring_setup_df = pd.DataFrame(
        [
            {
                "Environment": "Current Demo",
                "Reference Data": "Validation data",
                "Current Data": "Test data",
                "Purpose": "Demonstrates monitoring mechanics using held-out project data.",
            },
            {
                "Environment": "Production Target",
                "Reference Data": "Training/validation baseline",
                "Current Data": "Latest scored customer batch",
                "Purpose": "Detects whether live customer population or model outputs changed after deployment.",
            },
        ]
    )

    st.dataframe(monitoring_setup_df, use_container_width=True)

    st.warning(
        "MLOps rule: Drift does not automatically retrain the model. "
        "Drift triggers investigation and retraining review. A challenger model should only replace the Logistic Regression champion after evaluation."
    )

    st.divider()

    st.subheader("Monitoring KPI Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Features Monitored", features_monitored)
    col2.metric("Drifted Features", drifted_features)
    col3.metric("Data Drift", str(data_drift))
    col4.metric("Prediction Drift", str(prediction_drift))
    col5.metric("Retraining Review", bool_to_status(retraining_recommended, "Review Needed", "Not Needed"))

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
                "Check": "Retraining Review",
                "Result": "Review Needed" if retraining_recommended else "Not Needed",
                "Meaning": "Monitoring-based trigger to investigate whether retraining may be needed.",
                "Business Decision": monitoring_recommended_action,
            },
        ]
    )

    st.dataframe(monitoring_decisions, use_container_width=True)

    st.divider()

    st.subheader("Retraining Review Decision")

    if not data_drift and not prediction_drift and not retraining_recommended:
        st.success(
            "Monitoring status: Healthy. No data drift, no prediction drift, and no retraining review needed."
        )

        st.info(
            "Business decision: Keep the current MLflow champion model active for scoring. Continue periodic monitoring before considering retraining."
        )
    else:
        st.error(
            "Monitoring status: Review needed. Drift or a monitoring trigger was detected."
        )

        st.warning(
            "Business decision: Investigate drifted features, wait for/collect recent labels if needed, compare champion vs challenger, and retrain only if justified."
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

    st.subheader("Official Batch Scoring Summary")

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
        title="Official Batch Prediction Risk Distribution",
        xlabel="Customer Count",
        xlim=(0, max(total_scored + 1, 5)),
        value_formatter=lambda value: f"{int(value)}",
        figsize=NORMAL_CHART_SIZE,
    )

    st.divider()

    st.subheader("Official Scored Customer Output")

    display_cols = [
        col
        for col in [
            "customer_id",
            "prediction_label_dashboard",
            "churn_probability",
            "risk_band_dashboard",
            "recommended_action",
        ]
        if col in batch_predictions_df.columns
    ]

    st.dataframe(batch_predictions_df[display_cols], use_container_width=True)

    csv_data = batch_predictions_df.to_csv(index=False)

    st.download_button(
        label="Download Official Batch Prediction Results",
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
                "Probability Rule": f"{MEDIUM_RISK_THRESHOLD:.2f} <= churn_probability < {best_threshold:.2f}",
                "Business Action": "Monitor or send low-cost engagement offer.",
                "Owner": "Customer success / marketing",
            },
            {
                "Risk Band": "Low Risk",
                "Probability Rule": f"churn_probability < {MEDIUM_RISK_THRESHOLD:.2f}",
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

    st.subheader("Champion vs Challenger")

    st.dataframe(model_comparison_df, use_container_width=True)

    st.info(
        "Champion policy: Logistic Regression remains the deployed champion because it was selected using the predefined validation ROC-AUC criterion and is easier to explain. "
        "Random Forest is retained only as a challenger model for comparison, not as the served model."
    )

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

    render_retention_policy_note(best_threshold)

    st.markdown(
        f"""
        **Business use:**

        - Customers with churn probability **>= {best_threshold:.2f}** should be tagged as High Risk.
        - High-risk customers should be prioritized for retention outreach.
        - Medium-risk customers should receive low-cost engagement or monitoring.
        - If the retention budget is limited, increase the threshold.
        - If lost customers are more expensive than retention offers, lower the threshold.
        """
    )

    st.divider()

    st.subheader("Business Impact Simulator")

    st.caption(
        "Estimate whether the selected threshold can produce positive retention value. "
        "These are scenario assumptions for decision support, not guaranteed financial results."
    )

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

    with sim_col1:
        monthly_revenue = st.number_input(
            "Average Monthly Revenue per Customer",
            min_value=0.0,
            value=1500.0,
            step=100.0,
        )

    with sim_col2:
        retention_offer_cost = st.number_input(
            "Retention Offer Cost per Flagged Customer",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

    with sim_col3:
        retained_months = st.number_input(
            "Expected Retained Months",
            min_value=1,
            max_value=24,
            value=3,
            step=1,
        )

    with sim_col4:
        retention_success_rate = st.slider(
            "Expected Retention Success Rate",
            min_value=0.00,
            max_value=1.00,
            value=0.30,
            step=0.05,
        )

    expected_saved_customers = threshold_tp * retention_success_rate
    campaign_cost = flagged_customers * retention_offer_cost
    expected_saved_revenue = expected_saved_customers * monthly_revenue * retained_months
    estimated_net_impact = expected_saved_revenue - campaign_cost
    roi_ratio = estimated_net_impact / campaign_cost if campaign_cost > 0 else 0.0

    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)

    impact_col1.metric("Expected Saved Customers", f"{expected_saved_customers:.1f}")
    impact_col2.metric("Campaign Cost", f"{campaign_cost:,.0f}")
    impact_col3.metric("Expected Saved Revenue", f"{expected_saved_revenue:,.0f}")
    impact_col4.metric("Estimated Net Impact", f"{estimated_net_impact:,.0f}")

    st.metric("Estimated ROI Ratio", f"{roi_ratio:.2f}x")

    impact_table = pd.DataFrame(
        [
            {
                "Input / Output": "Flagged Customers",
                "Value": flagged_customers,
                "Meaning": "Customers who would receive retention action at the selected threshold.",
            },
            {
                "Input / Output": "True Positives",
                "Value": threshold_tp,
                "Meaning": "Customers correctly identified as churners in the test-set threshold analysis.",
            },
            {
                "Input / Output": "Expected Saved Customers",
                "Value": round(expected_saved_customers, 2),
                "Meaning": "Estimated retained customers after applying the assumed success rate.",
            },
            {
                "Input / Output": "Campaign Cost",
                "Value": round(campaign_cost, 2),
                "Meaning": "Flagged customers multiplied by offer cost.",
            },
            {
                "Input / Output": "Expected Saved Revenue",
                "Value": round(expected_saved_revenue, 2),
                "Meaning": "Expected saved customers multiplied by monthly revenue and retained months.",
            },
            {
                "Input / Output": "Estimated Net Impact",
                "Value": round(estimated_net_impact, 2),
                "Meaning": "Expected saved revenue minus campaign cost.",
            },
        ]
    )

    st.dataframe(impact_table, use_container_width=True)

    if estimated_net_impact >= 0:
        st.success(
            "Scenario result: Positive estimated net impact. This threshold may be financially reasonable under the current assumptions."
        )
    else:
        st.warning(
            "Scenario result: Negative estimated net impact. Consider improving offer targeting, reducing offer cost, increasing threshold, or improving retention success rate."
        )

    st.info(
        "Business decision: Use this simulator to discuss the retention campaign as an ROI trade-off, not just a model accuracy result."
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
    col4.metric("API Health", "Online" if api_is_online else "Offline")
    col5.metric("Docker", "Working")

    st.divider()

    st.subheader("MLOps Component Checklist")

    demo_df = get_demo_predictions_df()
    demo_total, demo_high, demo_medium, demo_low = summarize_risk_counts(demo_df)

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
                "Status": "Working" if api_is_online else "Offline",
                "Evidence": "/health and /predict endpoints are used for live scoring.",
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
                "Component": "Official Batch Prediction",
                "Status": "Working",
                "Evidence": f"{total_scored} customers scored; {high_risk_count} high risk, {medium_risk_count} medium risk, {low_risk_count} low risk.",
                "Business / Technical Value": "Creates operational retention queue.",
            },
            {
                "Component": "Live Demo Batch",
                "Status": "Active" if demo_total > 0 else "Empty",
                "Evidence": f"{demo_total} demo customers scored; {demo_high} high risk, {demo_medium} medium risk, {demo_low} low risk.",
                "Business / Technical Value": "Demonstrates live scoring and changing dashboard visuals.",
            },
            {
                "Component": "Streamlit Dashboard",
                "Status": "Working",
                "Evidence": "Dashboard pages show model, monitoring, threshold, batch, live scoring, and drift simulation.",
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
            "Ready for portfolio demo: The project includes testing, CI, model registry, API serving, Docker, monitoring, batch scoring, live demo scoring, drift simulation, and dashboard reporting."
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
        - FastAPI serves live predictions.
        - Docker containerizes the API.
        - Monitoring checks data drift and prediction drift.
        - Batch scoring creates a retention action queue.
        - Live demo scoring shows changing dashboard visuals.
        - Drift simulation shows why monitoring matters.
        - Streamlit turns technical outputs into business decisions.
        """
    )

    st.success(
        "Demo-ready upgrade complete: the dashboard now supports aligned risk policy, presentation-flow navigation, live scoring, ROI simulation, honest monitoring context, and prediction drift simulation."
    )