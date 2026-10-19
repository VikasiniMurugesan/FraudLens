"""Reproducible generator for SIMULATED transactions (not real data)."""

import numpy as np
import pandas as pd

import config

CATEGORIES = ["grocery", "fuel", "restaurant", "online_retail",
              "electronics", "travel", "entertainment", "utilities"]
GENUINE_P = [0.30, 0.15, 0.20, 0.10, 0.05, 0.05, 0.10, 0.05]
FRAUD_P = [0.05, 0.05, 0.05, 0.25, 0.30, 0.20, 0.05, 0.05]


def generate_transactions(n: int = 2000, fraud_rate: float = 0.03,
                          seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    is_fraud = rng.random(n) < fraud_rate

    amount = np.where(is_fraud,
                      rng.lognormal(5.0, 1.0, n),
                      rng.lognormal(3.5, 0.8, n)).round(2)

    distance = np.where(is_fraud,
                        rng.exponential(60, n),
                        rng.exponential(8, n)).round(1)

    is_online = np.where(is_fraud,
                         rng.random(n) < 0.70,
                         rng.random(n) < 0.25).astype(int)

    category = np.where(is_fraud,
                        rng.choice(CATEGORIES, size=n, p=FRAUD_P),
                        rng.choice(CATEGORIES, size=n, p=GENUINE_P))

        # Time: 30 days from 1 Jan 2024. 60% of fraud happens at night (0-4h).
    day = rng.integers(0, 30, n)
    night = is_fraud & (rng.random(n) < 0.60)
    hour = np.where(night, rng.integers(0, 5, n), rng.integers(7, 23, n))
    minute = rng.integers(0, 60, n)
    timestamp = (pd.Timestamp("2024-01-01")
                 + pd.to_timedelta(day, unit="D")
                 + pd.to_timedelta(hour, unit="h")
                 + pd.to_timedelta(minute, unit="min"))

    df = pd.DataFrame({
        "timestamp": timestamp,
        "amount": amount,
        "merchant_category": category,
        "distance_from_home_km": distance,
        "is_online": is_online,
        "is_fraud": is_fraud.astype(int),
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{i:06d}" for i in range(n)])
    return df


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv(config.SAMPLE_CSV_PATH, index=False)
    print(f"Wrote {len(df)} SIMULATED rows to {config.SAMPLE_CSV_PATH}")
    print(df.head(3).to_string())