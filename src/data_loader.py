"""Loading and validating transaction data."""

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

PUBLIC_REQUIRED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_public_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError if the table is not a valid creditcard.csv-style table."""
    validate_columns(df, PUBLIC_REQUIRED_COLUMNS)
    if df[["Amount", "Class"]].isna().any().any():
        raise ValueError("Amount and Class must not contain missing values")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount contains negative values")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")


def load_public_dataset(path: Path = config.PUBLIC_DATASET_PATH,
                        drop_duplicates: bool = True) -> pd.DataFrame:
    """Load creditcard.csv, validate it, and optionally drop exact duplicates."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from Kaggle (see data/README.md)."
        )
    df = pd.read_csv(path)
    validate_public_dataset(df)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %d duplicate rows", before - len(df))
    return df