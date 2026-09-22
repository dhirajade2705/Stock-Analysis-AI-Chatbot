"""
services/stock_data.py

Handles retrieval, validation and light caching of historical stock market
data using yfinance. No prices are ever hard-coded - everything returned
here comes from the actual download.
"""

import os
import time
import json
import pandas as pd
import yfinance as yf

from config.config import Config


class StockDataError(Exception):
    """Raised when stock data cannot be retrieved or is invalid."""
    pass


def _cache_path(symbol: str, period: str) -> str:
    safe_symbol = symbol.upper().replace("/", "_").replace("\\", "_")
    filename = f"{safe_symbol}_{period}.json"
    return os.path.join(Config.CACHE_DIR, filename)


def _read_cache(symbol: str, period: str):
    path = _cache_path(symbol, period)
    if not os.path.exists(path):
        return None
    age_minutes = (time.time() - os.path.getmtime(path)) / 60.0
    if age_minutes > Config.CACHE_EXPIRY_MINUTES:
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        df = pd.read_json(payload["data"], orient="split")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _write_cache(symbol: str, period: str, df: pd.DataFrame):
    try:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        path = _cache_path(symbol, period)
        payload = {"data": df.to_json(orient="split", date_format="iso")}
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        # Caching is a best-effort optimisation; failures should not break the app.
        pass


def validate_symbol(symbol: str) -> str:
    """Basic validation/normalisation of a ticker symbol string."""
    if not symbol or not isinstance(symbol, str):
        raise StockDataError("Stock symbol cannot be empty.")
    symbol = symbol.strip().upper()
    if len(symbol) == 0:
        raise StockDataError("Stock symbol cannot be empty.")
    if len(symbol) > 15:
        raise StockDataError("Stock symbol looks invalid (too long).")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^")
    if not set(symbol).issubset(allowed):
        raise StockDataError("Stock symbol contains invalid characters.")
    return symbol


def validate_period(period: str) -> str:
    if period not in Config.VALID_PERIODS:
        raise StockDataError(
            f"Invalid period '{period}'. Valid options: {', '.join(Config.VALID_PERIODS)}"
        )
    return period


def get_historical_data(symbol: str, period: str = None) -> pd.DataFrame:
    """
    Download historical OHLCV data for a symbol using yfinance.
    Returns a pandas DataFrame indexed by date.
    Raises StockDataError on any failure.
    """
    symbol = validate_symbol(symbol)
    period = validate_period(period or Config.DEFAULT_PERIOD)

    cached = _read_cache(symbol, period)
    if cached is not None and not cached.empty:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:
        raise StockDataError(
            f"Network or data error while fetching '{symbol}': {exc}"
        )

    if df is None or df.empty:
        raise StockDataError(
            f"No data found for symbol '{symbol}'. It may be delisted, "
            f"invalid, or an unsupported exchange suffix was used."
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise StockDataError(f"Downloaded data for '{symbol}' contained no usable rows.")

    df.index.name = "Date"
    _write_cache(symbol, period, df)
    return df


def get_company_info(symbol: str) -> dict:
    """Fetch descriptive company/asset metadata. Falls back gracefully."""
    symbol = validate_symbol(symbol)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception:
        info = {}

    return {
        "symbol": symbol,
        "shortName": info.get("shortName") or info.get("longName") or symbol,
        "longName": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", "N/A"),
        "marketCap": info.get("marketCap", None),
    }


def dataframe_to_records(df: pd.DataFrame) -> list:
    """Convert an OHLCV DataFrame into a JSON-serialisable list of dicts."""
    records = []
    for date, row in df.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        })
    return records


def get_basic_stats(df: pd.DataFrame) -> dict:
    """Compute simple, real (non-fabricated) descriptive statistics."""
    if df is None or df.empty:
        raise StockDataError("Cannot compute statistics on empty data.")

    closes = df["Close"]
    current_price = float(closes.iloc[-1])
    previous_close = float(closes.iloc[-2]) if len(closes) > 1 else current_price
    change = current_price - previous_close
    pct_change = (change / previous_close * 100) if previous_close != 0 else 0.0

    return {
        "current_price": round(current_price, 4),
        "previous_close": round(previous_close, 4),
        "change": round(change, 4),
        "percent_change": round(pct_change, 4),
        "day_high": round(float(df["High"].iloc[-1]), 4),
        "day_low": round(float(df["Low"].iloc[-1]), 4),
        "period_high": round(float(df["High"].max()), 4),
        "period_low": round(float(df["Low"].min()), 4),
        "average_close": round(float(closes.mean()), 4),
        "current_volume": int(df["Volume"].iloc[-1]) if not pd.isna(df["Volume"].iloc[-1]) else 0,
        "average_volume": int(df["Volume"].mean()) if not pd.isna(df["Volume"].mean()) else 0,
        "records_count": int(len(df)),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
    }