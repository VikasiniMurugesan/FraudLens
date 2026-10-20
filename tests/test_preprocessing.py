import pandas as pd
import pytest

from src.preprocessing import time_based_split


def make_df(n: int = 100) -> pd.DataFrame:
    """Rows deliberately in REVERSE time order, to prove the function sorts."""
    return pd.DataFrame({"Time": list(range(n, 0, -1)), "Class": [0] * n})


def test_split_sizes():
    train, val, test = time_based_split(make_df(100))
    assert (len(train), len(val), len(test)) == (60, 20, 20)


def test_no_rows_lost():
    train, val, test = time_based_split(make_df(100))
    assert len(train) + len(val) + len(test) == 100


def test_chronological_order():
    train, val, test = time_based_split(make_df(100))
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        time_based_split(make_df(100), train_frac=0.9, val_frac=0.3)