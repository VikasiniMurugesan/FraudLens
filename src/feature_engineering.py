"""Turn readable (simulated) transaction columns into model features."""

import numpy as np
import pandas as pd

from src.data_loader import validate_columns
from src.synthetic_data import CATEGORIES

REQUIRED_RAW = ["timestamp", "amount", "merchant_category",
                "distance_from_home_km", "is_online"]

DEMO_FEATURES = (
    ["log_amount", "distance_from_home_km", "is_online", "hour", "is_night"]
    + [f"cat_{c}" for c in CATEGORIES]
)


def add_demo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the demo model's feature columns added."""
    validate_columns(df, REQUIRED_RAW)
    out = df.copy()

    try:
        ts = pd.to_datetime(out["timestamp"])
    except (ValueError, TypeError):
        raise ValueError(
            "Column 'timestamp' contains values that are not valid dates"
        ) from None
    if (out["amount"] < 0).any():
        raise ValueError("amount contains negative values")

    out["log_amount"] = np.log1p(out["amount"])
    out["hour"] = ts.dt.hour.astype(int)
    out["is_night"] = (out["hour"] < 5).astype(int)
    for category in CATEGORIES:
        out[f"cat_{category}"] = (out["merchant_category"] == category).astype(int)
    return out