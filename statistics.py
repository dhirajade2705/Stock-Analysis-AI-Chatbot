"""
services/statistics.py

Additional descriptive/statistical helpers shared by the API and chatbot,
kept separate from stock_data.py to avoid duplicating logic.
"""

import numpy as np
import pandas as pd


def price_n_days_ago(df: pd.DataFrame, n: int):
    """Return the closing price n trading days ago, or None if unavailable."""
    if len(df) <= n:
        return None
    return round(float(df["Close"].iloc[-1 - n]), 4)


def percentage_return(df: pd.DataFrame) -> float:
    """Total percentage return over the whole available period."""
    start = float(df["Close"].iloc[0])
    end = float(df["Close"].iloc[-1])
    if start == 0:
        return 0.0
    return round(((end - start) / start) * 100, 4)


def daily_return_series(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change() * 100


def historical_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """Annualised historical volatility (%) based on rolling daily returns."""
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    vol = returns.std() * np.sqrt(252) * 100
    return round(float(vol), 4)


def risk_level_from_volatility(volatility_pct: float) -> str:
    """Simple, transparent bucketing of annualised volatility into risk labels."""
    if volatility_pct < 20:
        return "LOW"
    if volatility_pct < 40:
        return "MODERATE"
    if volatility_pct < 60:
        return "HIGH"
    return "VERY HIGH"


def trend_direction(df: pd.DataFrame, lookback: int = 30) -> str:
    """Whether price has increased, decreased, or stayed flat over `lookback` days."""
    if len(df) < 2:
        return "UNKNOWN"
    lookback = min(lookback, len(df) - 1)
    start = float(df["Close"].iloc[-1 - lookback])
    end = float(df["Close"].iloc[-1])
    change_pct = ((end - start) / start) * 100 if start != 0 else 0
    if change_pct > 1:
        return "INCREASED"
    if change_pct < -1:
        return "DECREASED"
    return "FLAT"