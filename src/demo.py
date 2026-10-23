"""Helpers for the simulated demo mode."""

import pandas as pd

import config
from src.feature_engineering import add_demo_features
from src.predict import load_bundle


def load_demo() -> tuple[dict, pd.DataFrame]:
    """Load the demo model and the simulated transactions with features added."""
    bundle = load_bundle(config.DEMO_MODEL_PATH)
    if not config.SAMPLE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{config.SAMPLE_CSV_PATH} not found. "
            "Generate it: python -m src.synthetic_data"
        )
    raw = pd.read_csv(config.SAMPLE_CSV_PATH)
    return bundle, add_demo_features(raw)


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    """Return the first of `names` that exists in df, or None."""
    for name in names:
        if name in df.columns:
            return name
    return None