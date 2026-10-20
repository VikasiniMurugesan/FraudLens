"""Turn model probabilities into 0-100 risk scores and risk categories."""

import numpy as np

import config


def probability_to_score(proba):
    """Convert probabilities in [0, 1] to risk scores in [0, 100]."""
    arr = np.asarray(proba, dtype=float)
    if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any():
        raise ValueError("Probabilities must be between 0 and 1")
    return np.round(arr * 100, 1)


def score_to_category(score: float, thresholds: dict | None = None) -> str:
    """Map a 0-100 score to 'low', 'medium' or 'high'."""
    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100")
    t = thresholds or config.RISK_THRESHOLDS
    if score >= t["high"]:
        return "high"
    if score >= t["medium"]:
        return "medium"
    return "low"
