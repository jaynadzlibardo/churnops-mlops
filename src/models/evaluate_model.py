"""
Final model evaluation script for ChurnOps.

This script:
1. Loads the held-out test dataset.
2. Loads the selected best model.
3. Evaluates final model performance.
4. Saves metrics report.
5. Saves confusion matrix.
6. Adds business interpretation.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/churn_model.pkl")
METRICS_REPORT_PATH = Path("reports/test_metrics_report.json")
CONFUSION_MATRIX_PATH = Path("reports/test_confusion_matrix.png")

TARGET_COLUMN = "Churn"
THRESHOLD = 0.50


def load_test_data():
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


def assign_business_error_explanation(cm):
    """
    Extract confusion matrix values and explain them.

    Confusion matrix layout:
    [[true_negative, false_positive],
     [false_negative, true_positive]]
    """
    tn, fp, fn, tp = cm.ravel()

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_business_cost": (
            "Customer is predicted as churn risk but would not actually churn. "
            "Business may waste a retention offer."
        ),
        "false_negative_business_cost": (
            "Customer is predicted as safe but actually churns. "
            "Business may lose customer revenue."
        ),
    }


def save_confusion_matrix(y_true, y_pred):
    """Save test confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title("Test Confusion Matrix - Selected Churn Model")
    plt.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    return cm


def main():
    print("=" * 80)
    print("CHURNOPS FINAL MODEL EVALUATION")
    print("=" * 80)

    print("[INFO] Loading test data and selected model...")
    X_test, y_test, model = load_test_data()

    print(f"[INFO] Test rows: {X_test.shape[0]}")
    print(f"[INFO] Feature columns: {X_test.shape[1]}")
    print(f"[INFO] Classification threshold: {THRESHOLD}")

    print("[INFO] Generating predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("[INFO] Saving confusion matrix...")
    cm = save_confusion_matrix(y_test, y_pred)
    error_explanation = assign_business_error_explanation(cm)

    report = {
        "model_path": str(MODEL_PATH),
        "test_data_path": str(TEST_PATH),
        "test_rows": int(X_test.shape[0]),
        "metrics": metrics,
        "confusion_matrix": error_explanation,
        "business_interpretation": {
            "roc_auc": (
                "ROC-AUC measures how well the model ranks churners above non-churners."
            ),
            "recall": (
                "Recall measures how many actual churners were successfully detected. "
                "This is important because missed churners represent potential lost revenue."
            ),
            "precision": (
                "Precision measures how many predicted churners were actually churners. "
                "This matters because false positives may waste retention budget."
            ),
            "threshold": (
                "The 0.50 threshold is the default decision cutoff. "
                "The business can lower it to catch more churners or raise it to reduce wasted offers."
            ),
        },
        "recommended_business_use": (
            "Use the churn probability to rank customers by risk. "
            "Prioritize high-risk customers for retention campaigns, especially when campaign budget is limited."
        ),
    }

    METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print("[RESULT] Final Test Metrics")
    print(f"         Accuracy:  {metrics['accuracy']:.4f}")
    print(f"         ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"         Precision: {metrics['precision']:.4f}")
    print(f"         Recall:    {metrics['recall']:.4f}")
    print(f"         F1-score:  {metrics['f1']:.4f}")
    print("-" * 80)
    print("[RESULT] Confusion Matrix Values")
    print(f"         True Negatives:  {error_explanation['true_negatives']}")
    print(f"         False Positives: {error_explanation['false_positives']}")
    print(f"         False Negatives: {error_explanation['false_negatives']}")
    print(f"         True Positives:  {error_explanation['true_positives']}")
    print("-" * 80)
    print(f"[RESULT] Saved test metrics report to: {METRICS_REPORT_PATH}")
    print(f"[RESULT] Saved test confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print("[SUCCESS] Final model evaluation completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()