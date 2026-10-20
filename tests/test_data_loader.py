import pandas as pd
import pytest

from src.data_loader import (
    PUBLIC_REQUIRED_COLUMNS,
    load_public_dataset,
    validate_columns,
    validate_public_dataset,
)


def make_df() -> pd.DataFrame:
    """A tiny valid table with the right columns."""
    data = {c: [0.0, 0.0, 0.0] for c in PUBLIC_REQUIRED_COLUMNS}
    data["Amount"] = [10.0, 20.0, 30.0]
    data["Class"] = [0, 1, 0]
    return pd.DataFrame(data)


def test_valid_table_passes():
    validate_public_dataset(make_df())


def test_missing_column_raises():
    df = make_df().drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, PUBLIC_REQUIRED_COLUMNS)


def test_negative_amount_raises():
    df = make_df()
    df.loc[0, "Amount"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate_public_dataset(df)


def test_bad_class_label_raises():
    df = make_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_public_dataset(df)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_public_dataset(tmp_path / "nope.csv")


def test_loader_drops_duplicates(tmp_path):
    df = make_df()
    with_dup = pd.concat([df, df.iloc[[0]]])  # row 0 repeated
    csv_path = tmp_path / "tiny.csv"
    with_dup.to_csv(csv_path, index=False)

    loaded = load_public_dataset(csv_path)
    assert len(with_dup) == 4
    assert len(loaded) == 3