"""
Feature engineering and preprocessing script for ChurnOps.

This script:
1. Loads the raw Telco Customer Churn dataset.
2. Cleans TotalCharges.
3. Drops customerID.
4. Converts Churn to binary target.
5. Splits into train, validation, and test sets.
6. Fits preprocessing only on training data to avoid leakage.
7. Saves processed splits and fitted preprocessor artifact.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
FEATURE_METADATA_PATH = Path("reports/feature_metadata.json")


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw Telco churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset before splitting."""
    df = df.copy()

    # Convert TotalCharges from object/string to numeric.
    # Blank values become NaN and will be imputed later.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert target to binary.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Drop ID column because it is not predictive and can cause overfitting.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    return df


def split_data(df: pd.DataFrame):
    """
    Split data into train, validation, and test sets.

    Final split:
    - Train: 70%
    - Validation: 15%
    - Test: 15%

    Stratification keeps the churn ratio similar across splits.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    val_df = X_val.copy()
    val_df[TARGET_COLUMN] = y_val

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    return train_df, val_df, test_df


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numeric features:
    - Missing values: median
    - Scaling: StandardScaler

    Categorical features:
    - Missing values: most frequent
    - Encoding: OneHotEncoder
    """
    X_train = train_df.drop(columns=[TARGET_COLUMN])

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    """Save processed splits, preprocessor, and feature metadata."""
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "target_column": TARGET_COLUMN,
        "dropped_columns": [ID_COLUMN],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(train_df.shape[0]),
        "validation_rows": int(val_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "train_churn_rate": float(train_df[TARGET_COLUMN].mean()),
        "validation_churn_rate": float(val_df[TARGET_COLUMN].mean()),
        "test_churn_rate": float(test_df[TARGET_COLUMN].mean()),
        "leakage_prevention": (
            "Data was split before fitting the preprocessor. "
            "The preprocessor was fitted only on the training data."
        ),
    }

    with open(FEATURE_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def main() -> None:
    print("=" * 80)
    print("CHURNOPS PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 80)

    print(f"[INFO] Loading raw data from: {RAW_DATA_PATH}")
    raw_df = load_raw_data(RAW_DATA_PATH)

    print("[INFO] Cleaning data...")
    clean_df = clean_data(raw_df)

    print("[INFO] Splitting data into train/validation/test...")
    train_df, val_df, test_df = split_data(clean_df)

    print("[INFO] Building preprocessing pipeline...")
    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df)

    print("[INFO] Fitting preprocessor on training data only...")
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    preprocessor.fit(X_train)

    print("[INFO] Saving processed datasets and preprocessor...")
    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("-" * 80)
    print(f"[RESULT] Train rows: {train_df.shape[0]}")
    print(f"[RESULT] Validation rows: {val_df.shape[0]}")
    print(f"[RESULT] Test rows: {test_df.shape[0]}")
    print(f"[RESULT] Numeric features: {numeric_features}")
    print(f"[RESULT] Categorical features: {categorical_features}")
    print(f"[RESULT] Train churn rate: {train_df[TARGET_COLUMN].mean():.4f}")
    print(f"[RESULT] Validation churn rate: {val_df[TARGET_COLUMN].mean():.4f}")
    print(f"[RESULT] Test churn rate: {test_df[TARGET_COLUMN].mean():.4f}")
    print(f"[RESULT] Saved train data to: {TRAIN_PATH}")
    print(f"[RESULT] Saved validation data to: {VAL_PATH}")
    print(f"[RESULT] Saved test data to: {TEST_PATH}")
    print(f"[RESULT] Saved preprocessor to: {PREPROCESSOR_PATH}")
    print(f"[RESULT] Saved feature metadata to: {FEATURE_METADATA_PATH}")
    print("-" * 80)

    print("[SUCCESS] Preprocessing completed successfully.")
    print("[MLOPS NOTE] Preprocessor was fitted only on training data to avoid leakage.")


if __name__ == "__main__":
    main()