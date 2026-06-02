"""
Model training script for ChurnOps.

This script:
1. Loads processed train and validation data.
2. Loads the fitted preprocessor.
3. Trains a baseline Logistic Regression model.
4. Trains an improved Random Forest model.
5. Tracks experiments using MLflow.
6. Logs parameters, metrics, plots, and model artifacts.
7. Saves the best model based on validation ROC-AUC.
"""

from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
BEST_MODEL_PATH = Path("models/churn_model.pkl")
METRICS_REPORT_PATH = Path("reports/metrics_report.json")
CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix.png")

TARGET_COLUMN = "Churn"
EXPERIMENT_NAME = "churnops-telco-churn"


def load_data():
    """Load processed train and validation datasets."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation data not found: {VAL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    return X_train, y_train, X_val, y_val, preprocessor


def evaluate_model(model, X_val, y_val, threshold=0.50):
    """Evaluate model using validation data."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    return metrics, y_pred, y_proba


def save_confusion_matrix(y_true, y_pred, model_name):
    """Save confusion matrix plot."""
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    return CONFUSION_MATRIX_PATH


def train_and_log_model(model_name, classifier, preprocessor, X_train, y_train, X_val, y_val):
    """Train model, evaluate it, and log outputs to MLflow."""
    print(f"[INFO] Training model: {model_name}")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:
        model_pipeline.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(
            model=model_pipeline,
            X_val=X_val,
            y_val=y_val,
        )

        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("threshold", 0.50)

        if model_name == "logistic_regression_baseline":
            mlflow.log_param("classifier", "LogisticRegression")
            mlflow.log_param("max_iter", classifier.max_iter)
            mlflow.log_param("class_weight", classifier.class_weight)

        if model_name == "random_forest_improved":
            mlflow.log_param("classifier", "RandomForestClassifier")
            mlflow.log_param("n_estimators", classifier.n_estimators)
            mlflow.log_param("max_depth", classifier.max_depth)
            mlflow.log_param("min_samples_split", classifier.min_samples_split)
            mlflow.log_param("class_weight", classifier.class_weight)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Save and log confusion matrix for this run
        cm_path = save_confusion_matrix(y_val, y_pred, model_name)
        mlflow.log_artifact(str(cm_path))

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            input_example=X_val.head(3),
        )

        run_id = run.info.run_id

        print(f"[RESULT] {model_name}")
        print(f"         Run ID: {run_id}")
        print(f"         ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"         Precision: {metrics['precision']:.4f}")
        print(f"         Recall:    {metrics['recall']:.4f}")
        print(f"         F1-score:  {metrics['f1']:.4f}")

        return {
            "model_name": model_name,
            "run_id": run_id,
            "model": model_pipeline,
            "metrics": metrics,
        }


def save_best_model(best_result, all_results):
    """Save best model and metrics report."""
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_result["model"], BEST_MODEL_PATH)

    report = {
        "selection_metric": "validation_roc_auc",
        "best_model": best_result["model_name"],
        "best_run_id": best_result["run_id"],
        "best_metrics": best_result["metrics"],
        "all_model_results": [
            {
                "model_name": result["model_name"],
                "run_id": result["run_id"],
                "metrics": result["metrics"],
            }
            for result in all_results
        ],
        "business_interpretation": {
            "roc_auc": "Measures how well the model ranks churners above non-churners.",
            "recall": "Higher recall means fewer missed churners.",
            "precision": "Higher precision means fewer wasted retention offers.",
            "f1": "Balances precision and recall.",
        },
    }

    with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print(f"[RESULT] Best model selected: {best_result['model_name']}")
    print(f"[RESULT] Best validation ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
    print(f"[RESULT] Saved best model to: {BEST_MODEL_PATH}")
    print(f"[RESULT] Saved metrics report to: {METRICS_REPORT_PATH}")


def main():
    print("=" * 80)
    print("CHURNOPS MODEL TRAINING WITH MLFLOW")
    print("=" * 80)

    print("[INFO] Loading data and preprocessor...")
    X_train, y_train, X_val, y_val, preprocessor = load_data()

    print(f"[INFO] Training rows: {X_train.shape[0]}")
    print(f"[INFO] Validation rows: {X_val.shape[0]}")
    print(f"[INFO] Feature columns: {X_train.shape[1]}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    models_to_train = [
        (
            "logistic_regression_baseline",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "random_forest_improved",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for model_name, classifier in models_to_train:
        result = train_and_log_model(
            model_name=model_name,
            classifier=classifier,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        results.append(result)

    best_result = max(results, key=lambda item: item["metrics"]["roc_auc"])

    save_best_model(best_result, results)

    print("-" * 80)
    print("[SUCCESS] Model training completed successfully.")
    print("[MLOPS NOTE] Experiments were tracked in MLflow.")
    print("[MLOPS NOTE] Best model was selected using validation ROC-AUC.")
    print("[BUSINESS NOTE] Recall shows how many churners the model catches.")
    print("=" * 80)


if __name__ == "__main__":
    main()