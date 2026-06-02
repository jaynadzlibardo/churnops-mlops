"""
Feature importance script for ChurnOps.

This script:
1. Loads the trained model pipeline.
2. Extracts transformed feature names from the preprocessor.
3. Extracts feature importance from the classifier.
4. Saves a feature importance CSV.
5. Saves a feature importance plot.

For Logistic Regression:
- Uses absolute coefficient values.

For Random Forest:
- Uses feature_importances_.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = Path("models/churn_model.pkl")
FEATURE_IMPORTANCE_CSV_PATH = Path("reports/feature_importance.csv")
FEATURE_IMPORTANCE_PLOT_PATH = Path("reports/feature_importance.png")
FEATURE_IMPORTANCE_REPORT_PATH = Path("reports/feature_importance_report.json")

TOP_N = 15


def load_model():
    """Load trained model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/models/train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def get_feature_names(model_pipeline):
    """Extract transformed feature names from the preprocessing pipeline."""
    preprocessor = model_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    cleaned_feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]

    return cleaned_feature_names


def get_importance_values(model_pipeline):
    """Extract importance values depending on classifier type."""
    classifier = model_pipeline.named_steps["classifier"]
    classifier_name = classifier.__class__.__name__

    if hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])
        importance_type = "absolute_logistic_regression_coefficients"

    elif hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        importance_type = "tree_feature_importance"

    else:
        raise ValueError(
            f"Classifier {classifier_name} does not expose coefficients or feature importances."
        )

    return classifier_name, importance_type, importance_values


def create_feature_importance_table(feature_names, importance_values):
    """Create sorted feature importance dataframe."""
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def save_feature_importance_plot(importance_df):
    """Save top N feature importance plot."""
    top_df = importance_df.head(TOP_N).sort_values(
        by="importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {TOP_N} Churn Prediction Features")
    plt.tight_layout()

    FEATURE_IMPORTANCE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close()


def create_business_interpretation(top_features):
    """Create simple business interpretation for top features."""
    return {
        "summary": (
            "The most important features indicate which customer characteristics "
            "contribute most to churn risk prediction."
        ),
        "top_features": top_features,
        "business_use": (
            "The retention team can use these drivers to understand which customer "
            "segments need attention, such as customers with risky contract types, "
            "short tenure, high charges, or missing support services."
        ),
        "caution": (
            "Feature importance explains model influence, not guaranteed causality. "
            "Business actions should still be validated through retention experiments."
        ),
    }


def main():
    print("=" * 80)
    print("CHURNOPS FEATURE IMPORTANCE")
    print("=" * 80)

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model_pipeline = load_model()

    print("[INFO] Extracting transformed feature names...")
    feature_names = get_feature_names(model_pipeline)

    print("[INFO] Extracting importance values...")
    classifier_name, importance_type, importance_values = get_importance_values(
        model_pipeline
    )

    print(f"[INFO] Classifier: {classifier_name}")
    print(f"[INFO] Importance type: {importance_type}")
    print(f"[INFO] Number of transformed features: {len(feature_names)}")

    importance_df = create_feature_importance_table(
        feature_names=feature_names,
        importance_values=importance_values,
    )

    FEATURE_IMPORTANCE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV_PATH, index=False)

    print("[INFO] Saving feature importance plot...")
    save_feature_importance_plot(importance_df)

    top_features = importance_df.head(TOP_N).to_dict(orient="records")

    report = {
        "model_path": str(MODEL_PATH),
        "classifier": classifier_name,
        "importance_type": importance_type,
        "top_n": TOP_N,
        "top_features": top_features,
        "business_interpretation": create_business_interpretation(top_features),
    }

    with open(FEATURE_IMPORTANCE_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("-" * 80)
    print(f"[RESULT] Feature importance CSV saved to: {FEATURE_IMPORTANCE_CSV_PATH}")
    print(f"[RESULT] Feature importance plot saved to: {FEATURE_IMPORTANCE_PLOT_PATH}")
    print(f"[RESULT] Feature importance report saved to: {FEATURE_IMPORTANCE_REPORT_PATH}")
    print("-" * 80)

    print("[RESULT] Top churn drivers:")
    for index, row in importance_df.head(10).iterrows():
        print(f"         {index + 1}. {row['feature']} = {row['importance']:.4f}")

    print("-" * 80)
    print("[SUCCESS] Feature importance completed successfully.")
    print("[BUSINESS NOTE] Use this to explain major churn risk drivers.")
    print("=" * 80)


if __name__ == "__main__":
    main()