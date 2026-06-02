"""
Monitoring script for ChurnOps.

This script compares reference data and current data to detect:
1. Numeric feature drift
2. Categorical feature drift
3. Prediction drift
4. Retraining recommendation

Outputs:
- reports/monitoring_report.html
- reports/monitoring_summary.json

This is intentionally lightweight and demo-friendly.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency


REFERENCE_DATA_PATH = Path("data/processed/val.csv")
CURRENT_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/churn_model.pkl")

MONITORING_REPORT_PATH = Path("reports/monitoring_report.html")
MONITORING_SUMMARY_PATH = Path("reports/monitoring_summary.json")

TARGET_COLUMN = "Churn"
NUMERIC_DRIFT_PVALUE_THRESHOLD = 0.05
CATEGORICAL_DRIFT_PVALUE_THRESHOLD = 0.05
DRIFTED_FEATURE_SHARE_THRESHOLD = 0.30
PREDICTION_MEAN_CHANGE_THRESHOLD = 0.10


def load_inputs():
    """Load reference data, current data, and trained model."""
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Current data not found: {CURRENT_DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    current_df = pd.read_csv(CURRENT_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return reference_df, current_df, model


def get_feature_columns(df: pd.DataFrame):
    """Separate numeric and categorical feature columns."""
    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    numeric_features = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


def detect_numeric_drift(reference_df, current_df, numeric_features):
    """Detect numeric drift using Kolmogorov-Smirnov test."""
    results = []

    for feature in numeric_features:
        ref_values = reference_df[feature].dropna()
        cur_values = current_df[feature].dropna()

        if len(ref_values) == 0 or len(cur_values) == 0:
            p_value = None
            drift_detected = False
            statistic = None
        else:
            statistic, p_value = ks_2samp(ref_values, cur_values)
            drift_detected = p_value < NUMERIC_DRIFT_PVALUE_THRESHOLD

        results.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "reference_mean": float(ref_values.mean()) if len(ref_values) else None,
                "current_mean": float(cur_values.mean()) if len(cur_values) else None,
                "test": "Kolmogorov-Smirnov",
                "statistic": float(statistic) if statistic is not None else None,
                "p_value": float(p_value) if p_value is not None else None,
                "drift_detected": bool(drift_detected),
            }
        )

    return results


def detect_categorical_drift(reference_df, current_df, categorical_features):
    """Detect categorical drift using chi-square test."""
    results = []

    for feature in categorical_features:
        ref_counts = reference_df[feature].fillna("MISSING").value_counts()
        cur_counts = current_df[feature].fillna("MISSING").value_counts()

        categories = sorted(set(ref_counts.index).union(set(cur_counts.index)))

        ref_aligned = [ref_counts.get(category, 0) for category in categories]
        cur_aligned = [cur_counts.get(category, 0) for category in categories]

        contingency_table = np.array([ref_aligned, cur_aligned])

        try:
            statistic, p_value, _, _ = chi2_contingency(contingency_table)
            drift_detected = p_value < CATEGORICAL_DRIFT_PVALUE_THRESHOLD
        except ValueError:
            statistic = None
            p_value = None
            drift_detected = False

        results.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "reference_top_value": str(ref_counts.idxmax()) if len(ref_counts) else None,
                "current_top_value": str(cur_counts.idxmax()) if len(cur_counts) else None,
                "test": "Chi-square",
                "statistic": float(statistic) if statistic is not None else None,
                "p_value": float(p_value) if p_value is not None else None,
                "drift_detected": bool(drift_detected),
            }
        )

    return results


def generate_predictions(reference_df, current_df, model):
    """Generate churn probabilities for reference and current data."""
    X_reference = reference_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    X_current = current_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    reference_proba = model.predict_proba(X_reference)[:, 1]
    current_proba = model.predict_proba(X_current)[:, 1]

    return reference_proba, current_proba


def detect_prediction_drift(reference_proba, current_proba):
    """Detect prediction drift using probability distribution and mean change."""
    statistic, p_value = ks_2samp(reference_proba, current_proba)

    reference_mean = float(np.mean(reference_proba))
    current_mean = float(np.mean(current_proba))
    mean_change = float(abs(current_mean - reference_mean))

    drift_detected = (
        p_value < NUMERIC_DRIFT_PVALUE_THRESHOLD
        or mean_change > PREDICTION_MEAN_CHANGE_THRESHOLD
    )

    return {
        "reference_mean_churn_probability": reference_mean,
        "current_mean_churn_probability": current_mean,
        "absolute_mean_change": mean_change,
        "test": "Kolmogorov-Smirnov",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "prediction_drift_detected": bool(drift_detected),
    }


def create_html_report(summary, feature_results, prediction_drift):
    """Create a simple HTML monitoring report."""
    drift_rows = ""

    for result in feature_results:
        drift_status = "DRIFT" if result["drift_detected"] else "OK"
        p_value = result["p_value"]
        p_value_text = f"{p_value:.5f}" if p_value is not None else "N/A"

        drift_rows += f"""
        <tr>
            <td>{result["feature"]}</td>
            <td>{result["feature_type"]}</td>
            <td>{result["test"]}</td>
            <td>{p_value_text}</td>
            <td>{drift_status}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChurnOps Monitoring Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                color: #222;
            }}
            h1, h2 {{
                color: #1f4e79;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 30px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .card {{
                border: 1px solid #ddd;
                padding: 16px;
                margin-bottom: 20px;
                border-radius: 8px;
                background-color: #fafafa;
            }}
        </style>
    </head>
    <body>
        <h1>ChurnOps Monitoring Report</h1>

        <div class="card">
            <h2>Monitoring Summary</h2>
            <p><strong>Reference dataset:</strong> Validation data</p>
            <p><strong>Current dataset:</strong> Test data</p>
            <p><strong>Total features monitored:</strong> {summary["total_features"]}</p>
            <p><strong>Drifted features:</strong> {summary["drifted_features"]}</p>
            <p><strong>Drifted feature share:</strong> {summary["drifted_feature_share"]:.2%}</p>
            <p><strong>Data drift detected:</strong> {summary["data_drift_detected"]}</p>
            <p><strong>Prediction drift detected:</strong> {summary["prediction_drift_detected"]}</p>
            <p><strong>Retraining recommended:</strong> {summary["retraining_recommended"]}</p>
            <p><strong>Alert level:</strong> {summary["alert_level"]}</p>
            <p><strong>Recommended action:</strong> {summary["recommended_action"]}</p>
        </div>

        <div class="card">
            <h2>Prediction Drift</h2>
            <p><strong>Reference mean churn probability:</strong> {prediction_drift["reference_mean_churn_probability"]:.4f}</p>
            <p><strong>Current mean churn probability:</strong> {prediction_drift["current_mean_churn_probability"]:.4f}</p>
            <p><strong>Absolute mean change:</strong> {prediction_drift["absolute_mean_change"]:.4f}</p>
            <p><strong>P-value:</strong> {prediction_drift["p_value"]:.5f}</p>
            <p><strong>Prediction drift detected:</strong> {prediction_drift["prediction_drift_detected"]}</p>
        </div>

        <h2>Feature Drift Details</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Type</th>
                <th>Test</th>
                <th>P-value</th>
                <th>Status</th>
            </tr>
            {drift_rows}
        </table>

        <div class="card">
            <h2>Business Interpretation</h2>
            <p>
                Data drift means the customer population may be changing compared with the data used during model validation.
                Prediction drift means the model's churn risk outputs are changing. If drift persists, the business should review
                data quality, check recent customer behavior, and consider retraining the model.
            </p>
        </div>
    </body>
    </html>
    """

    MONITORING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MONITORING_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(html)


def main():
    print("=" * 80)
    print("CHURNOPS MONITORING AND DRIFT REPORT")
    print("=" * 80)

    print("[INFO] Loading reference data, current data, and model...")
    reference_df, current_df, model = load_inputs()

    print(f"[INFO] Reference rows: {reference_df.shape[0]}")
    print(f"[INFO] Current rows: {current_df.shape[0]}")

    numeric_features, categorical_features = get_feature_columns(reference_df)

    print(f"[INFO] Numeric features monitored: {numeric_features}")
    print(f"[INFO] Categorical features monitored: {categorical_features}")

    print("[INFO] Detecting numeric feature drift...")
    numeric_results = detect_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        numeric_features=numeric_features,
    )

    print("[INFO] Detecting categorical feature drift...")
    categorical_results = detect_categorical_drift(
        reference_df=reference_df,
        current_df=current_df,
        categorical_features=categorical_features,
    )

    feature_results = numeric_results + categorical_results

    print("[INFO] Generating prediction probabilities...")
    reference_proba, current_proba = generate_predictions(
        reference_df=reference_df,
        current_df=current_df,
        model=model,
    )

    print("[INFO] Detecting prediction drift...")
    prediction_drift = detect_prediction_drift(
        reference_proba=reference_proba,
        current_proba=current_proba,
    )

    total_features = len(feature_results)
    drifted_features = sum(result["drift_detected"] for result in feature_results)
    drifted_feature_share = drifted_features / total_features if total_features else 0

    data_drift_detected = drifted_feature_share > DRIFTED_FEATURE_SHARE_THRESHOLD
    prediction_drift_detected = prediction_drift["prediction_drift_detected"]

    retraining_recommended = data_drift_detected or prediction_drift_detected

    if retraining_recommended:
        alert_level = "REVIEW_REQUIRED"
        recommended_action = (
            "Review data quality, investigate feature/prediction drift, "
            "and consider retraining if drift persists."
        )
    else:
        alert_level = "OK"
        recommended_action = "No immediate retraining required. Continue monitoring."

    summary = {
        "reference_data_path": str(REFERENCE_DATA_PATH),
        "current_data_path": str(CURRENT_DATA_PATH),
        "total_features": int(total_features),
        "drifted_features": int(drifted_features),
        "drifted_feature_share": float(drifted_feature_share),
        "data_drift_detected": bool(data_drift_detected),
        "prediction_drift_detected": bool(prediction_drift_detected),
        "retraining_recommended": bool(retraining_recommended),
        "alert_level": alert_level,
        "recommended_action": recommended_action,
        "prediction_drift": prediction_drift,
        "feature_drift_results": feature_results,
    }

    MONITORING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MONITORING_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("[INFO] Creating HTML monitoring report...")
    create_html_report(
        summary=summary,
        feature_results=feature_results,
        prediction_drift=prediction_drift,
    )

    print("-" * 80)
    print(f"[RESULT] Total features monitored: {total_features}")
    print(f"[RESULT] Drifted features: {drifted_features}")
    print(f"[RESULT] Drifted feature share: {drifted_feature_share:.2%}")
    print(f"[RESULT] Data drift detected: {data_drift_detected}")
    print(f"[RESULT] Prediction drift detected: {prediction_drift_detected}")
    print(f"[RESULT] Retraining recommended: {retraining_recommended}")
    print(f"[RESULT] Alert level: {alert_level}")
    print(f"[RESULT] Recommended action: {recommended_action}")
    print(f"[RESULT] Monitoring summary saved to: {MONITORING_SUMMARY_PATH}")
    print(f"[RESULT] Monitoring HTML report saved to: {MONITORING_REPORT_PATH}")
    print("-" * 80)
    print("[SUCCESS] Monitoring completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()