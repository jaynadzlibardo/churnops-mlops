"""
Batch prediction script for ChurnOps.

This script:
1. Loads a CSV file of customer records.
2. Loads the trained churn model.
3. Generates churn predictions and probabilities.
4. Assigns risk levels and business actions.
5. Saves the output to reports/batch_predictions.csv.
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/batch/sample_customers.csv")
DEFAULT_OUTPUT_PATH = Path("reports/batch_predictions.csv")
MODEL_PATH = Path("models/churn_model.pkl")

THRESHOLD = 0.50


def get_risk_level(churn_probability: float) -> str:
    """Convert churn probability into business risk level."""
    if churn_probability < 0.30:
        return "Low"
    if churn_probability <= 0.60:
        return "Medium"
    return "High"


def get_business_action(risk_level: str) -> str:
    """Recommend business action based on risk level."""
    if risk_level == "Low":
        return "No immediate retention action needed."
    if risk_level == "Medium":
        return "Monitor customer and consider low-cost retention engagement."
    return "Prioritize for retention campaign or proactive customer support."


def validate_input_columns(input_df: pd.DataFrame, model) -> None:
    """
    Validate that the input CSV has the required model features.

    The trained pipeline expects the same raw feature names used during training.
    """
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)
    else:
        expected_columns = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]

    missing_columns = [col for col in expected_columns if col not in input_df.columns]

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{missing_columns}"
        )


def run_batch_prediction(input_path: Path, output_path: Path) -> None:
    """Run batch prediction and save output CSV."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/models/train_model.py first."
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Input batch file not found: {input_path}")

    print("=" * 80)
    print("CHURNOPS BATCH PREDICTION")
    print("=" * 80)

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"[INFO] Loading input customers from: {input_path}")
    input_df = pd.read_csv(input_path)

    print(f"[INFO] Input rows: {input_df.shape[0]}")
    print(f"[INFO] Input columns: {input_df.shape[1]}")

    print("[INFO] Validating input columns...")
    validate_input_columns(input_df, model)

    print("[INFO] Generating churn probabilities...")
    churn_probabilities = model.predict_proba(input_df)[:, 1]
    predictions = (churn_probabilities >= THRESHOLD).astype(int)

    output_df = input_df.copy()
    output_df["prediction"] = predictions
    output_df["prediction_label"] = output_df["prediction"].map(
        {1: "Churn", 0: "No Churn"}
    )
    output_df["churn_probability"] = churn_probabilities.round(4)
    output_df["risk_level"] = output_df["churn_probability"].apply(get_risk_level)
    output_df["business_action"] = output_df["risk_level"].apply(get_business_action)

    output_df = output_df.sort_values(
        by="churn_probability",
        ascending=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    high_risk_count = int((output_df["risk_level"] == "High").sum())
    medium_risk_count = int((output_df["risk_level"] == "Medium").sum())
    low_risk_count = int((output_df["risk_level"] == "Low").sum())

    print("-" * 80)
    print(f"[RESULT] Batch predictions saved to: {output_path}")
    print(f"[RESULT] Total customers scored: {output_df.shape[0]}")
    print(f"[RESULT] High risk customers: {high_risk_count}")
    print(f"[RESULT] Medium risk customers: {medium_risk_count}")
    print(f"[RESULT] Low risk customers: {low_risk_count}")
    print("-" * 80)
    print("[SUCCESS] Batch prediction completed successfully.")
    print("[BUSINESS NOTE] High-risk customers should be prioritized for retention.")
    print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ChurnOps batch prediction.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to input customer CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where predictions CSV will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_batch_prediction(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()