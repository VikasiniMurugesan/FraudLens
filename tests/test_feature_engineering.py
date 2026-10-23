import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import DEMO_FEATURES, add_demo_features


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": ["2024-01-01 02:30:00", "2024-01-01 12:00:00"],
        "amount": [100.0, 10.0],
        "merchant_category": ["electronics", "grocery"],
        "distance_from_home_km": [50.0, 2.0],
        "is_online": [1, 0],
    })


def test_adds_all_feature_columns():
    out = add_demo_features(make_df())
    assert set(DEMO_FEATURES) <= set(out.columns)


def test_hour_and_night_flag():
    out = add_demo_features(make_df())
    assert out["hour"].tolist() == [2, 12]
    assert out["is_night"].tolist() == [1, 0]


def test_log_amount():
    out = add_demo_features(make_df())
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(100.0))


def test_one_hot_categories():
    out = add_demo_features(make_df())
    assert out["cat_electronics"].tolist() == [1, 0]
    assert out["cat_grocery"].tolist() == [0, 1]


def test_unknown_category_is_all_zero():
    df = make_df()
    df.loc[0, "merchant_category"] = "spaceship"
    out = add_demo_features(df)
    cat_cols = [c for c in DEMO_FEATURES if c.startswith("cat_")]
    assert out.loc[0, cat_cols].sum() == 0


def test_input_is_not_modified():
    df = make_df()
    before = df.copy()
    add_demo_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_demo_features(make_df().drop(columns=["timestamp"]))


def test_bad_timestamp_raises():
    df = make_df()
    df.loc[0, "timestamp"] = "not a date"
    with pytest.raises(ValueError, match="timestamp"):
        add_demo_features(df)


def test_negative_amount_raises():
    df = make_df()
    df.loc[0, "amount"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        add_demo_features(df)