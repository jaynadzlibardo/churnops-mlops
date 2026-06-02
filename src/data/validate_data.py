"""
Data validation script for ChurnOps.

This script validates the raw Telco Customer Churn dataset before it enters
the preprocessing and model training pipeline.

Checks included:
1. File existence
2. Required columns
3. Row and column count
4. Missing values
5. Duplicate rows
6. Target distribution
7. Invalid TotalCharges values
8. Report generation
"""

from pathlib import Path
import json
import pandas as pd


RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
REPORT_PATH = Path("reports/data_validation_report.json")


REQUIRED_COLUMNS = [
    "customerID",
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
    "Churn",
]


def validate_file_exists(path: Path) -> None:
    """Check if the raw dataset exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            "Place the Telco churn CSV inside data/raw/ and rename it to telco_churn.csv"
        )


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """Return missing required columns."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def validate_total_charges(df: pd.DataFrame) -> dict:
    """
    Check TotalCharges quality.

    In the Telco churn dataset, TotalCharges sometimes contains blank strings.
    These need to be converted to numeric later during preprocessing.
    """
    total_charges_raw = df["TotalCharges"].astype(str)

    blank_count = total_charges_raw.str.strip().eq("").sum()
    numeric_converted = pd.to_numeric(total_charges_raw, errors="coerce")
    invalid_numeric_count = numeric_converted.isna().sum()

    return {
        "blank_values": int(blank_count),
        "invalid_numeric_values_after_conversion": int(invalid_numeric_count),
    }


def main() -> None:
    print("=" * 80)
    print("CHURNOPS DATA VALIDATION")
    print("=" * 80)

    validate_file_exists(RAW_DATA_PATH)

    print(f"[INFO] Loading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    print("[INFO] Checking required columns...")
    missing_columns = validate_required_columns(df)

    print("[INFO] Checking missing values...")
    missing_values = df.isna().sum().to_dict()

    print("[INFO] Checking duplicate rows...")
    duplicate_rows = int(df.duplicated().sum())

    print("[INFO] Checking target distribution...")
    target_distribution = df["Churn"].value_counts(dropna=False).to_dict()

    print("[INFO] Checking TotalCharges quality...")
    total_charges_quality = validate_total_charges(df)

    validation_status = "PASSED"

    if missing_columns:
        validation_status = "FAILED"

    if "Churn" not in df.columns:
        validation_status = "FAILED"

    report = {
        "validation_status": validation_status,
        "raw_data_path": str(RAW_DATA_PATH),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "required_column_count": len(REQUIRED_COLUMNS),
        "actual_columns": list(df.columns),
        "missing_required_columns": missing_columns,
        "missing_values": {k: int(v) for k, v in missing_values.items()},
        "duplicate_rows": duplicate_rows,
        "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
        "total_charges_quality": total_charges_quality,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print(f"[RESULT] Validation status: {validation_status}")
    print(f"[RESULT] Rows: {df.shape[0]}")
    print(f"[RESULT] Columns: {df.shape[1]}")
    print(f"[RESULT] Duplicate rows: {duplicate_rows}")
    print(f"[RESULT] Missing required columns: {missing_columns}")
    print(f"[RESULT] Target distribution: {target_distribution}")
    print(f"[RESULT] TotalCharges quality: {total_charges_quality}")
    print(f"[RESULT] Validation report saved to: {REPORT_PATH}")
    print("-" * 80)

    if validation_status == "FAILED":
        raise ValueError(
            "Data validation failed. Check missing required columns in the validation report."
        )

    print("[SUCCESS] Data validation completed successfully.")


if __name__ == "__main__":
    main()