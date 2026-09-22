"""
services/indicators.py

Calculates real technical indicators from OHLCV data using pandas/numpy
(with the `ta` library used for RSI/MACD/Bollinger where convenient).
No indicator value is ever fabricated - all are derived mathematically
from the historical Close/High/Low/Volume series that was downloaded.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

from config.config import Config


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a NEW DataFrame with technical indicator columns appended.
    The original df is not mutated.
    """
    data = df.copy()
    close = data["Close"]

    # --- SMA ---
    data["SMA20"] = SMAIndicator(close=close, window=Config.SMA_SHORT_WINDOW).sma_indicator()
    data["SMA50"] = SMAIndicator(close=close, window=Config.SMA_LONG_WINDOW).sma_indicator()

    # --- EMA ---
    data["EMA20"] = EMAIndicator(close=close, window=Config.EMA_SHORT_WINDOW).ema_indicator()
    data["EMA50"] = EMAIndicator(close=close, window=Config.EMA_LONG_WINDOW).ema_indicator()

    # --- RSI ---
    data["RSI14"] = RSIIndicator(close=close, window=Config.RSI_WINDOW).rsi()

    # --- MACD ---
    macd = MACD(
        close=close,
        window_slow=Config.MACD_SLOW,
        window_fast=Config.MACD_FAST,
        window_sign=Config.MACD_SIGNAL,
    )
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=close, window=Config.BOLLINGER_WINDOW, window_dev=Config.BOLLINGER_STD)
    data["BB_Upper"] = bb.bollinger_hband()
    data["BB_Middle"] = bb.bollinger_mavg()
    data["BB_Lower"] = bb.bollinger_lband()

    # --- Returns ---
    data["Daily_Return"] = close.pct_change() * 100
    data["Cumulative_Return"] = (1 + close.pct_change()).cumprod() - 1
    data["Cumulative_Return"] = data["Cumulative_Return"] * 100

    # --- Volatility (rolling 20-day std dev of daily returns, annualised) ---
    data["Volatility"] = close.pct_change().rolling(window=20).std() * np.sqrt(252) * 100

    return data


def get_latest_indicator_snapshot(df_with_indicators: pd.DataFrame) -> dict:
    """Return the most recent value of every indicator, handling NaNs safely."""
    last = df_with_indicators.iloc[-1]

    def safe(value):
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return None
        return round(float(value), 4)

    return {
        "sma20": safe(last.get("SMA20")),
        "sma50": safe(last.get("SMA50")),
        "ema20": safe(last.get("EMA20")),
        "ema50": safe(last.get("EMA50")),
        "rsi14": safe(last.get("RSI14")),
        "macd": safe(last.get("MACD")),
        "macd_signal": safe(last.get("MACD_Signal")),
        "macd_histogram": safe(last.get("MACD_Hist")),
        "bollinger_upper": safe(last.get("BB_Upper")),
        "bollinger_middle": safe(last.get("BB_Middle")),
        "bollinger_lower": safe(last.get("BB_Lower")),
        "daily_return": safe(last.get("Daily_Return")),
        "cumulative_return": safe(last.get("Cumulative_Return")),
        "volatility": safe(last.get("Volatility")),
    }


def indicator_series_for_charts(df_with_indicators: pd.DataFrame, max_points: int = 180) -> dict:
    """Return trimmed indicator time series (dates + values) for chart rendering."""
    d = df_with_indicators.tail(max_points)
    dates = [idx.strftime("%Y-%m-%d") for idx in d.index]

    def series(col):
        return [None if pd.isna(v) else round(float(v), 4) for v in d[col]]

    return {
        "dates": dates,
        "close": series("Close"),
        "sma20": series("SMA20"),
        "sma50": series("SMA50"),
        "ema20": series("EMA20"),
        "ema50": series("EMA50"),
        "rsi14": series("RSI14"),
        "macd": series("MACD"),
        "macd_signal": series("MACD_Signal"),
        "macd_histogram": series("MACD_Hist"),
        "bollinger_upper": series("BB_Upper"),
        "bollinger_middle": series("BB_Middle"),
        "bollinger_lower": series("BB_Lower"),
        "volume": [int(v) if not pd.isna(v) else 0 for v in d["Volume"]],
    }


def generate_technical_signal(snapshot: dict, current_price: float) -> dict:
    """
    Produce an educational BULLISH / BEARISH / NEUTRAL signal from real
    indicator values, with a plain-language explanation. This is NOT
    financial advice.
    """
    score = 0
    reasons = []

    rsi = snapshot.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            score += 1
            reasons.append(f"RSI is {rsi:.1f}, which is in oversold territory (<30), often a bullish signal.")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI is {rsi:.1f}, which is in overbought territory (>70), often a bearish signal.")
        else:
            reasons.append(f"RSI is {rsi:.1f}, which is in the neutral range (30-70).")

    macd = snapshot.get("macd")
    macd_signal = snapshot.get("macd_signal")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 1
            reasons.append("MACD line is above its signal line, a bullish momentum cue.")
        else:
            score -= 1
            reasons.append("MACD line is below its signal line, a bearish momentum cue.")

    sma20 = snapshot.get("sma20")
    sma50 = snapshot.get("sma50")
    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 1
            reasons.append("The 20-day SMA is above the 50-day SMA, indicating a short-term uptrend.")
        else:
            score -= 1
            reasons.append("The 20-day SMA is below the 50-day SMA, indicating a short-term downtrend.")

    if sma20 is not None and current_price is not None:
        if current_price > sma20:
            reasons.append("The current price is trading above its 20-day moving average.")
        else:
            reasons.append("The current price is trading below its 20-day moving average.")

    if score >= 2:
        label = "BULLISH"
    elif score <= -2:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "signal": label,
        "score": score,
        "reasons": reasons,
        "disclaimer": (
            "This signal is generated purely from historical technical indicators "
            "and is for educational purposes only. It is not financial advice."
        ),
    }