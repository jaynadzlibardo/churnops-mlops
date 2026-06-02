"""
Threshold analysis script for ChurnOps.

This script:
1. Loads the selected churn model.
2. Loads the test dataset.
3. Evaluates multiple classification thresholds.
4. Calculates precision, recall, F1, false positives, and false negatives.
5. Saves threshold analysis CSV and plot.
6. Creates business interpretation for retention decision-making.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/churn_model.pkl")

THRESHOLD_CSV_PATH = Path("reports/threshold_analysis.csv")
THRESHOLD_PLOT_PATH = Path("reports/threshold_analysis.png")
THRESHOLD_REPORT_PATH = Path("reports/threshold_analysis_report.json")

TARGET_COLUMN = "Churn"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def load_inputs():
    """Load test data and trained model."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_test, y_test, model


def evaluate_thresholds(y_true, y_proba):
    """Evaluate model performance across thresholds."""
    rows = []

    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "customers_flagged_for_retention": int(fp + tp),
                "missed_churners": int(fn),
            }
        )

    return pd.DataFrame(rows)


def save_threshold_plot(threshold_df):
    """Save threshold comparison plot."""
    plt.figure(figsize=(9, 6))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1-score",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Analysis: Precision vs Recall Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    THRESHOLD_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(THRESHOLD_PLOT_PATH, dpi=150)
    plt.close()


def choose_recommended_threshold(threshold_df):
    """
    Recommend a threshold.

    For churn, recall is important because missed churners mean possible revenue loss.
    But we do not want precision to collapse too much.

    Rule:
    - Choose the threshold with highest F1.
    - Include business note that lower threshold can be used if campaign budget is large.
    """
    best_row = threshold_df.sort_values(
        by=["f1", "recall"],
        ascending=[False, False],
    ).iloc[0]

    return best_row.to_dict()


def create_business_report(threshold_df, recommended_threshold):
    """Create business interpretation report."""
    low_threshold_row = threshold_df[threshold_df["threshold"] == 0.30].iloc[0].to_dict()
    default_threshold_row = threshold_df[threshold_df["threshold"] == 0.50].iloc[0].to_dict()
    high_threshold_row = threshold_df[threshold_df["threshold"] == 0.70].iloc[0].to_dict()

    report = {
        "model_path": str(MODEL_PATH),
        "test_data_path": str(TEST_PATH),
        "thresholds_tested": THRESHOLDS,
        "recommended_threshold": recommended_threshold,
        "business_interpretation": {
            "low_threshold_0_30": {
                "meaning": (
                    "More customers are flagged for retention. This catches more churners "
                    "but increases false positives and campaign cost."
                ),
                "results": low_threshold_row,
            },
            "default_threshold_0_50": {
                "meaning": (
                    "Balanced default threshold. Useful when the business wants a middle ground "
                    "between catching churners and controlling retention cost."
                ),
                "results": default_threshold_row,
            },
            "high_threshold_0_70": {
                "meaning": (
                    "Fewer customers are flagged. This reduces wasted offers but may miss more churners."
                ),
                "results": high_threshold_row,
            },
            "decision_rule": (
                "If retention budget is large, use a lower threshold to catch more churners. "
                "If retention budget is limited, use a higher threshold to focus on the highest-risk customers."
            ),
        },
    }

    return report


def main():
    print("=" * 80)
    print("CHURNOPS THRESHOLD ANALYSIS")
    print("=" * 80)

    print("[INFO] Loading test data and selected model...")
    X_test, y_test, model = load_inputs()

    print(f"[INFO] Test rows: {X_test.shape[0]}")
    print(f"[INFO] Feature columns: {X_test.shape[1]}")

    print("[INFO] Generating churn probabilities...")
    y_proba = model.predict_proba(X_test)[:, 1]

    print("[INFO] Evaluating thresholds...")
    threshold_df = evaluate_thresholds(y_test, y_proba)

    THRESHOLD_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(THRESHOLD_CSV_PATH, index=False)

    print("[INFO] Saving threshold analysis plot...")
    save_threshold_plot(threshold_df)

    recommended_threshold = choose_recommended_threshold(threshold_df)
    report = create_business_report(threshold_df, recommended_threshold)

    with open(THRESHOLD_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print("[RESULT] Threshold Analysis")
    print(threshold_df.to_string(index=False))
    print("-" * 80)
    print(f"[RESULT] Recommended threshold by F1: {recommended_threshold['threshold']}")
    print(f"[RESULT] Precision: {recommended_threshold['precision']}")
    print(f"[RESULT] Recall: {recommended_threshold['recall']}")
    print(f"[RESULT] F1-score: {recommended_threshold['f1']}")
    print(f"[RESULT] Missed churners: {recommended_threshold['missed_churners']}")
    print(
        "[RESULT] Customers flagged for retention: "
        f"{recommended_threshold['customers_flagged_for_retention']}"
    )
    print("-" * 80)
    print(f"[RESULT] Threshold CSV saved to: {THRESHOLD_CSV_PATH}")
    print(f"[RESULT] Threshold plot saved to: {THRESHOLD_PLOT_PATH}")
    print(f"[RESULT] Threshold report saved to: {THRESHOLD_REPORT_PATH}")
    print("-" * 80)
    print("[SUCCESS] Threshold analysis completed successfully.")
    print("[BUSINESS NOTE] Threshold choice depends on retention budget and churn risk tolerance.")
    print("=" * 80)


if __name__ == "__main__":
    main()