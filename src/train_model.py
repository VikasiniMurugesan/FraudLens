"""Train and evaluate fraud models."""

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data_loader import load_public_dataset
from src.preprocessing import time_based_split

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def build_baseline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                     random_state=config.RANDOM_SEED)),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.RANDOM_SEED)),
    ])


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Score a fitted model on (X, y) and return the key metrics."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def train_and_save(threshold: float = 0.3) -> dict:
    """Train the chosen model on the training set, report validation and
    test metrics, and save the model and metrics to disk."""
    df = load_public_dataset()
    train, val, test = time_based_split(df)

    model = build_random_forest().fit(train[FEATURE_COLUMNS], train[TARGET])

    results = {
        "model": "random_forest",
        "threshold": threshold,
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "frauds": {"train": int(train[TARGET].sum()),
                   "validation": int(val[TARGET].sum()),
                   "test": int(test[TARGET].sum())},
        "validation": evaluate(model, val[FEATURE_COLUMNS], val[TARGET], threshold),
        "test": evaluate(model, test[FEATURE_COLUMNS], test[TARGET], threshold),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    reference = train[FEATURE_COLUMNS].median().to_dict()
    joblib.dump({"pipeline": model, "features": FEATURE_COLUMNS,
                 "threshold": threshold, "reference": reference},
                config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    results = train_and_save()
    for split in ("validation", "test"):
        print("==", split)
        for name, value in results[split].items():
            print(f"  {name}: {round(value, 4) if isinstance(value, float) else value}")
    print("Saved model to", config.MODEL_PATH)