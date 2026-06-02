from pathlib import Path
import pandas as pd


def test_raw_data_exists():
    assert Path("data/raw/telco_churn.csv").exists()


def test_processed_data_exists():
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/val.csv").exists()
    assert Path("data/processed/test.csv").exists()


def test_required_target_column_exists():
    train_df = pd.read_csv("data/processed/train.csv")
    assert "Churn" in train_df.columns


def test_no_customer_id_in_processed_data():
    train_df = pd.read_csv("data/processed/train.csv")
    assert "customerID" not in train_df.columns