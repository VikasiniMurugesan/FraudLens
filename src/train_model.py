"""Train and evaluate fraud models."""

import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Score a fitted model on (X, y) and return the key metrics."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


if __name__ == "__main__":
    df = load_public_dataset()
    train, val, test = time_based_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET]

    print("Do-nothing accuracy on validation:",
          round(1 - y_val.mean(), 4))

    model = build_baseline()
    model.fit(X_train, y_train)

    for name, value in evaluate(model, X_val, y_val).items():
        print(f"{name}: {round(value, 4) if isinstance(value, float) else value}")