"""Explain why one transaction received its risk score."""

import pandas as pd


def explain_transaction(row: pd.Series, bundle: dict, top_n: int = 3,
                        min_points: float = 0.5) -> list[dict]:
    """Return the features that add the most to this transaction's score.

    For each feature we set it to its typical (training median) value,
    rescore, and measure how many score points disappear.
    """
    reference = bundle.get("reference")
    if reference is None:
        raise ValueError(
            "Model bundle has no reference values. "
            "Retrain: python -m src.train_model"
        )
    features = bundle["features"]
    missing = [c for c in features if c not in row.index]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pipeline = bundle["pipeline"]
    x = pd.DataFrame([row[features].astype(float)])
    base = pipeline.predict_proba(x)[0, 1]

    variants = pd.concat([x] * len(features), ignore_index=True)
    for i, feature in enumerate(features):
        variants.loc[i, feature] = reference[feature]
    drops = (base - pipeline.predict_proba(variants)[:, 1]) * 100

    reasons = [
        {
            "feature": feature,
            "value": float(x.iloc[0][feature]),
            "typical": float(reference[feature]),
            "points": round(float(drop), 1),
        }
        for feature, drop in zip(features, drops)
        if drop >= min_points
    ]
    reasons.sort(key=lambda r: r["points"], reverse=True)
    return reasons[:top_n]


def reason_to_text(reason: dict) -> str:
    """Turn one reason into a plain sentence for the app."""
    direction = "higher" if reason["value"] > reason["typical"] else "lower"
    return (
        f"{reason['feature']} is {direction} than typical "
        f"({reason['value']:.2f} vs {reason['typical']:.2f}), "
        f"adding about {reason['points']} points to the risk score"
    )