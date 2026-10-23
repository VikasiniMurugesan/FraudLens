"""Train a DEMO model on simulated transactions (not real fraud data)."""

import json

import joblib

import config
from src.feature_engineering import DEMO_FEATURES, add_demo_features
from src.synthetic_data import generate_transactions
from src.train_model import build_random_forest, evaluate

DEMO_THRESHOLD = 0.3  # same as the real model; NOT tuned for the demo


def train_demo_and_save() -> dict:
    # Training data: 20,000 simulated rows from one seed.
    train_raw = generate_transactions(n=20000, seed=7)
    # Held-out data: a different seed. This is the same data as sample_transactions.csv.
    holdout_raw = generate_transactions(n=2000, seed=config.RANDOM_SEED)

    X_train = add_demo_features(train_raw)[DEMO_FEATURES]
    y_train = train_raw["is_fraud"]
    X_hold = add_demo_features(holdout_raw)[DEMO_FEATURES]
    y_hold = holdout_raw["is_fraud"]

    model = build_random_forest().fit(X_train, y_train)

    results = {
        "note": "SIMULATED data. Fraud patterns were planted by the generator, "
                "so these scores do not show real-world fraud detection ability.",
        "train_rows": int(len(train_raw)),
        "holdout_rows": int(len(holdout_raw)),
        "holdout_frauds": int(y_hold.sum()),
        "threshold": DEMO_THRESHOLD,
        "holdout": evaluate(model, X_hold, y_hold, DEMO_THRESHOLD),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": model, "features": DEMO_FEATURES,
         "threshold": DEMO_THRESHOLD,
         "reference": X_train.median().to_dict(),
         "label": "Demo model (simulated data)"},
        config.DEMO_MODEL_PATH,
    )
    config.DEMO_METRICS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    results = train_demo_and_save()
    print(json.dumps(results, indent=2))
    print("Saved demo model to", config.DEMO_MODEL_PATH)