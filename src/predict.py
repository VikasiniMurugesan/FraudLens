"""Load a saved model bundle and score transactions."""

import logging
from pathlib import Path

import joblib
import pandas as pd

import config
from src.risk_scoring import probability_to_score, score_to_category

logger = logging.getLogger(__name__)


def load_bundle(path: Path = config.MODEL_PATH) -> dict:
    """Load the saved model, feature list and threshold."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train the model first: python -m src.train_model"
        )
    return joblib.load(path)


def score_transactions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a copy of df with risk_score, risk_category and flagged columns."""
    features = bundle["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[features]
    if X.isna().any().any():
        raise ValueError("Feature columns contain missing values")

    proba = bundle["pipeline"].predict_proba(X)[:, 1]

    out = df.copy()
    out["risk_score"] = probability_to_score(proba)
    out["risk_category"] = [score_to_category(s) for s in out["risk_score"]]
    out["flagged"] = out["risk_score"] >= bundle["threshold"] * 100
    logger.info("Scored %d transactions, %d flagged", len(out), out["flagged"].sum())
    return out