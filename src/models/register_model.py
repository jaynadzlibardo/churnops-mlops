"""
MLflow model registry script for ChurnOps.

This script:
1. Finds the best MLflow run based on validation ROC-AUC.
2. Registers the best model artifact in MLflow Model Registry.
3. Assigns an alias to the registered model version.
4. Prints registry information for demo evidence.
"""

from pathlib import Path
import json
import time

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


EXPERIMENT_NAME = "churnops-telco-churn"
REGISTERED_MODEL_NAME = "churnops_model"
MODEL_ARTIFACT_PATH = "model"
REGISTRY_REPORT_PATH = Path("reports/model_registry_report.json")


def get_best_run(client: MlflowClient):
    """Find the best run by validation ROC-AUC."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found. "
            "Run src/models/train_model.py first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No runs found in experiment '{EXPERIMENT_NAME}'. "
            "Run model training first."
        )

    return runs[0]


def ensure_registered_model_exists(client: MlflowClient):
    """Create registered model if it does not already exist."""
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
        print(f"[INFO] Registered model already exists: {REGISTERED_MODEL_NAME}")
    except MlflowException:
        print(f"[INFO] Creating registered model: {REGISTERED_MODEL_NAME}")
        client.create_registered_model(REGISTERED_MODEL_NAME)


def wait_until_model_version_ready(client: MlflowClient, model_name: str, version: str):
    """Wait until registered model version is ready."""
    for _ in range(30):
        model_version = client.get_model_version(
            name=model_name,
            version=version,
        )

        status = model_version.status

        if status == "READY":
            return model_version

        print(f"[INFO] Waiting for model version to be READY. Current status: {status}")
        time.sleep(1)

    raise TimeoutError("Model version registration did not become READY in time.")


def main():
    print("=" * 80)
    print("CHURNOPS MLFLOW MODEL REGISTRY")
    print("=" * 80)

    client = MlflowClient()

    print("[INFO] Searching for best MLflow run...")
    best_run = get_best_run(client)

    run_id = best_run.info.run_id
    model_name = best_run.data.params.get("model_name", "unknown_model")
    roc_auc = best_run.data.metrics.get("roc_auc")
    recall = best_run.data.metrics.get("recall")
    precision = best_run.data.metrics.get("precision")
    f1 = best_run.data.metrics.get("f1")

    print(f"[RESULT] Best run ID: {run_id}")
    print(f"[RESULT] Best model name: {model_name}")
    print(f"[RESULT] Validation ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] Validation Recall: {recall:.4f}")
    print(f"[RESULT] Validation Precision: {precision:.4f}")
    print(f"[RESULT] Validation F1: {f1:.4f}")

    ensure_registered_model_exists(client)

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"

    print(f"[INFO] Registering model from URI: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    model_version = registered_model.version

    print(f"[INFO] Registered model version created: {model_version}")
    wait_until_model_version_ready(
        client=client,
        model_name=REGISTERED_MODEL_NAME,
        version=model_version,
    )

    print("[INFO] Adding model version tags...")
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="project",
        value="churnops-mlops",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="selection_metric",
        value="validation_roc_auc",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="business_metric",
        value="recall",
    )

    # MLflow 3 prefers aliases instead of old staging labels.
    print("[INFO] Setting model alias: champion")
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version,
    )

    report = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "alias": "champion",
        "source_run_id": run_id,
        "source_model_uri": model_uri,
        "source_model_name": model_name,
        "selection_metric": "validation_roc_auc",
        "metrics": {
            "roc_auc": float(roc_auc),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
        },
        "business_interpretation": (
            "The registered champion model is the selected model for deployment. "
            "It was chosen based on validation ROC-AUC, while recall is tracked as the "
            "business metric because it reflects how many churners are caught."
        ),
    }

    REGISTRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REGISTRY_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print(f"[RESULT] Registered model name: {REGISTERED_MODEL_NAME}")
    print(f"[RESULT] Registered version: {model_version}")
    print("[RESULT] Alias: champion")
    print(f"[RESULT] Registry report saved to: {REGISTRY_REPORT_PATH}")
    print("-" * 80)
    print("[SUCCESS] Model registration completed successfully.")
    print("[MLOPS NOTE] The selected model is now versioned and ready for deployment.")
    print("=" * 80)


if __name__ == "__main__":
    main()