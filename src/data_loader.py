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
def read_transactions_csv(source, required_columns: list[str],
                          max_rows: int = 200_000) -> pd.DataFrame:
    """Read an uploaded CSV and raise a clear ValueError if it is unusable.

    `source` can be a path or a file-like object (Streamlit's uploader).
    """
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty") from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise ValueError("The file could not be read as a CSV") from None

    if df.empty:
        raise ValueError("The file has a header but no rows")
    if len(df) > max_rows:
        raise ValueError(f"File has {len(df):,} rows; the limit is {max_rows:,}")

    validate_columns(df, required_columns)

    non_numeric = []
    for col in required_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if (converted.isna() & df[col].notna()).any():
            non_numeric.append(col)
        df[col] = converted
    if non_numeric:
        raise ValueError(f"Non-numeric values in columns: {non_numeric}")

    if df[required_columns].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if "Amount" in df.columns and (df["Amount"] < 0).any():
        raise ValueError("Amount contains negative values")
    return df