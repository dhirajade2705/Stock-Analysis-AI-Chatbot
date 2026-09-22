"""
services/preprocessing.py

Feature engineering pipeline for the ML prediction service.
Builds lag features and cleans data using ONLY chronological ordering
(no shuffling) to avoid time-series data leakage.
"""

import numpy as np
import pandas as pd

from services.indicators import add_all_indicators

LAG_PERIODS = [1, 2, 3, 5, 10]

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA20", "SMA50", "EMA20", "EMA50",
    "RSI14", "MACD", "BB_Upper", "BB_Lower",
    "Daily_Return", "Volatility", "Previous_Close",
] + [f"Lag_{p}" for p in LAG_PERIODS]

TARGET_COLUMN = "Target"


def build_feature_dataframe(raw_df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    Turns raw OHLCV data into a fully-engineered, model-ready DataFrame.
    Target = Close price `horizon_days` trading days into the future
    (i.e. the value the model learns to predict).
    """
    df = add_all_indicators(raw_df)

    df["Previous_Close"] = df["Close"].shift(1)
    for p in LAG_PERIODS:
        df[f"Lag_{p}"] = df["Close"].shift(p)

    # Target: future close price, shifted backward so each row's features
    # line up with the close price `horizon_days` ahead in time.
    df[TARGET_COLUMN] = df["Close"].shift(-horizon_days)

    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna()

    return model_df


def chronological_train_test_split(model_df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits data by TIME ORDER (no shuffling) so the test set is always
    later in time than the training set - this avoids look-ahead bias
    that a random shuffle would introduce into time-series data.
    """
    n = len(model_df)
    split_idx = int(n * (1 - test_size))
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test