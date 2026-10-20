"""Splitting data into train / validation / test sets."""

import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time order: earliest rows train, then validation, then test."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must be positive and sum to less than 1")

    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]