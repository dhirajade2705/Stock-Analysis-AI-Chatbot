"""
services/chatbot.py

Intent-based chatbot for stock analysis questions. No external LLM is
required - every answer is generated from real data returned by
stock_data.py, indicators.py, statistics.py and prediction.py.

Architecture:
  User Question -> preprocess -> detect intent (keyword scoring)
  -> pull required data -> generate a natural-language answer.
"""

import re

from services import stock_data, indicators, statistics as stats
from services.prediction import train_and_predict, PredictionError
from services.stock_data import StockDataError

DISCLAIMER = (
    "This is educational information based on historical data and a machine-learning "
    "estimate. It is not financial advice, and no outcome is guaranteed."
)

# Each intent has an id, a set of trigger keywords/phrases, and a human label.
# Keyword matching is scored so the closest-matching intent wins.
INTENT_DEFINITIONS = [
    ("current_price", ["current price", "what is the price", "price now", "trading at", "how much is"]),
    ("stock_symbol", ["stock symbol", "ticker symbol", "what symbol"]),
    ("company_name", ["what company", "company does this", "which company"]),
    ("previous_close", ["previous closing", "previous close", "last close", "yesterday's close"]),
    ("percent_change_today", ["percentage change", "% change", "today's change", "change today"]),
    ("current_volume", ["trading volume", "current volume", "how many shares traded"]),
    ("period_high", ["highest price in", "period high", "highest price of the selected"]),
    ("period_low", ["lowest price in", "period low", "lowest price of the selected"]),
    ("average_close", ["average closing price", "average close", "mean closing price"]),
    ("recent_performance", ["how has this stock performed recently", "recent performance", "performed recently"]),

    ("price_7_days_ago", ["7 days ago", "seven days ago", "price a week ago"]),
    ("price_30_days_ago", ["30 days ago", "thirty days ago", "price a month ago"]),
    ("price_90_days_ago", ["90 days ago", "ninety days ago", "price three months ago"]),
    ("historical_high", ["highest historical price", "all time high", "historical high"]),
    ("historical_low", ["lowest historical price", "all time low", "historical low"]),
    ("historical_average", ["historical average price", "average historical price"]),
    ("increase_or_decrease", ["increased or decreased", "gone up or down", "risen or fallen"]),
    ("percentage_return", ["percentage return", "total return", "overall return"]),
    ("daily_return", ["daily return", "return today", "day's return"]),
    ("volatility_level", ["how volatile is this stock", "volatility of this stock"]),

    ("current_moving_average", ["current moving average", "moving average now"]),
    ("sma20", ["20-day moving average", "20 day moving average", "sma 20", "sma20"]),
    ("sma50", ["50-day moving average", "50 day moving average", "sma 50", "sma50"]),
    ("rsi_value", ["what is the rsi", "current rsi", "rsi value"]),
    ("rsi_meaning", ["what does rsi indicate", "explain rsi", "meaning of rsi"]),
    ("macd_value", ["what is macd", "current macd", "macd value"]),
    ("macd_meaning", ["what does macd indicate", "explain macd", "meaning of macd"]),
    ("bollinger_meaning", ["what are bollinger bands", "explain bollinger", "bollinger bands"]),
    ("above_below_ma", ["above or below its moving average", "above or below moving average"]),
    ("technical_trend", ["current technical trend", "technical trend", "what is the trend"]),

    ("predicted_future_price", ["predicted future price", "future price prediction", "what will the price be"]),
    ("predicted_next_day", ["predicted price for the next trading day", "predict tomorrow", "next day price"]),
    ("predicted_next_7_days", ["predicted price for the next 7 days", "next 7 days", "next seven days"]),
    ("model_used", ["what model is used", "which model", "ml model used"]),
    ("how_prediction_generated", ["how was the prediction generated", "how is the prediction made"]),
    ("model_accuracy", ["how accurate is the model", "model accuracy", "accuracy of the model"]),
    ("rmse_value", ["what is the rmse", "rmse value", "rmse score"]),
    ("mae_value", ["what is the mae", "mae value", "mae score"]),
    ("r2_value", ["r2 score", "r-squared", "r squared"]),
    ("prediction_guarantee", ["can the prediction be guaranteed", "is the prediction guaranteed"]),

    ("is_volatile", ["is this stock volatile", "is the stock volatile"]),
    ("risk_level", ["risk level", "how risky is this stock"]),
    ("trend_bullish_bearish", ["bullish, bearish, or neutral", "bullish or bearish", "is the trend bullish"]),
    ("prediction_factors", ["factors affect the prediction", "what affects the prediction"]),
    ("why_inaccurate", ["why can stock predictions be inaccurate", "why predictions can be wrong", "why is the prediction wrong"]),
    ("historical_data_used", ["what historical data was used", "which data was used"]),
    ("training_records_count", ["how many records were used", "training data size", "number of records"]),
    ("important_indicators", ["most important indicators", "which indicators matter most", "important features"]),
    ("buy_or_sell", ["should i buy", "should i sell", "buy or sell"]),
    ("full_summary", ["complete summary", "summarize this stock", "give me a summary", "overview of this stock"]),
]

# Precompute a flat lookup: keyword -> intent_id, sorted by keyword length (longest first)
_ALL_KEYWORDS = []
for intent_id, phrases in INTENT_DEFINITIONS:
    for phrase in phrases:
        _ALL_KEYWORDS.append((phrase, intent_id))
_ALL_KEYWORDS.sort(key=lambda x: len(x[0]), reverse=True)


def _preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s%.\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def detect_intent(question: str) -> str:
    """Score-based keyword matching. Returns the best-matching intent id, or 'unknown'."""
    processed = _preprocess(question)
    for phrase, intent_id in _ALL_KEYWORDS:
        if phrase in processed:
            return intent_id

    # Fallback: loose single-word scoring for paraphrased questions.
    scores = {}
    words = set(processed.split())
    for intent_id, phrases in INTENT_DEFINITIONS:
        score = 0
        for phrase in phrases:
            phrase_words = set(phrase.split())
            score += len(words & phrase_words)
        if score > 0:
            scores[intent_id] = score

    if scores:
        return max(scores, key=scores.get)
    return "unknown"


def _gather_context(symbol: str, period: str = "1y") -> dict:
    """Pulls all data needed to answer any of the 50 questions for a symbol."""
    df = stock_data.get_historical_data(symbol, period)
    info = stock_data.get_company_info(symbol)
    basic = stock_data.get_basic_stats(df)
    df_ind = indicators.add_all_indicators(df)
    snapshot = indicators.get_latest_indicator_snapshot(df_ind)
    signal = indicators.generate_technical_signal(snapshot, basic["current_price"])
    return {
        "df": df,
        "info": info,
        "basic": basic,
        "snapshot": snapshot,
        "signal": signal,
    }


def _try_prediction(df, symbol, horizon_days=1):
    try:
        return train_and_predict(df, symbol, horizon_days=horizon_days)
    except PredictionError as e:
        return {"error": str(e)}


def generate_answer(question: str, symbol: str, period: str = "1y") -> dict:
    """
    Main entry point. Detects intent, retrieves the relevant real data,
    and returns a natural-language answer plus the detected intent (for
    debugging/UI badges).
    """
    if not question or not question.strip():
        return {"intent": "unknown", "answer": "Please type a question about the selected stock."}

    try:
        symbol = stock_data.validate_symbol(symbol)
    except StockDataError as e:
        return {"intent": "error", "answer": str(e)}

    intent = detect_intent(question)

    try:
        ctx = _gather_context(symbol, period)
    except StockDataError as e:
        return {"intent": intent, "answer": f"I couldn't retrieve data for '{symbol}': {e}"}

    df = ctx["df"]
    info = ctx["info"]
    b = ctx["basic"]
    s = ctx["snapshot"]
    sig = ctx["signal"]
    name = info.get("shortName", symbol)

    def fmt(v):
        return f"{v:,.2f}" if isinstance(v, (int, float)) else "N/A"

    answer = None

    if intent == "current_price":
        answer = f"The current price of {name} ({symbol}) is {fmt(b['current_price'])}."
    elif intent == "stock_symbol":
        answer = f"The stock symbol is {symbol}."
    elif intent == "company_name":
        answer = f"This stock belongs to {info.get('longName', symbol)} ({symbol}), sector: {info.get('sector', 'N/A')}."
    elif intent == "previous_close":
        answer = f"The previous closing price was {fmt(b['previous_close'])}."
    elif intent == "percent_change_today":
        answer = f"Today's change is {fmt(b['change'])} ({fmt(b['percent_change'])}%)."
    elif intent == "current_volume":
        answer = f"The current trading volume is {b['current_volume']:,} shares."
    elif intent == "period_high":
        answer = f"The highest price in the selected period is {fmt(b['period_high'])}."
    elif intent == "period_low":
        answer = f"The lowest price in the selected period is {fmt(b['period_low'])}."
    elif intent == "average_close":
        answer = f"The average closing price over the selected period is {fmt(b['average_close'])}."
    elif intent == "recent_performance":
        trend = stats.trend_direction(df, lookback=30)
        answer = f"Over the last 30 trading days, {symbol} has {trend.lower()}, with a total period return of {stats.percentage_return(df)}%."

    elif intent == "price_7_days_ago":
        p = stats.price_n_days_ago(df, 7)
        answer = f"The price 7 trading days ago was {fmt(p)}." if p else "Not enough data to look back 7 days."
    elif intent == "price_30_days_ago":
        p = stats.price_n_days_ago(df, 30)
        answer = f"The price 30 trading days ago was {fmt(p)}." if p else "Not enough data to look back 30 days."
    elif intent == "price_90_days_ago":
        p = stats.price_n_days_ago(df, 90)
        answer = f"The price 90 trading days ago was {fmt(p)}." if p else "Not enough data to look back 90 days."
    elif intent == "historical_high":
        answer = f"The highest historical price in the loaded data is {fmt(b['period_high'])}."
    elif intent == "historical_low":
        answer = f"The lowest historical price in the loaded data is {fmt(b['period_low'])}."
    elif intent == "historical_average":
        answer = f"The historical average closing price is {fmt(b['average_close'])}."
    elif intent == "increase_or_decrease":
        trend = stats.trend_direction(df, lookback=30)
        answer = f"Over the last 30 trading days, the stock has {trend.lower()}."
    elif intent == "percentage_return":
        answer = f"The total percentage return over the loaded period is {stats.percentage_return(df)}%."
    elif intent == "daily_return":
        answer = f"The most recent daily return is {s['daily_return']}%." if s['daily_return'] is not None else "Not enough data to compute a daily return."
    elif intent == "volatility_level":
        vol = stats.historical_volatility(df)
        answer = f"The annualised historical volatility is {vol}%, which is considered {stats.risk_level_from_volatility(vol)} risk."

    elif intent == "current_moving_average":
        answer = f"The 20-day SMA is {fmt(s['sma20'])} and the 50-day SMA is {fmt(s['sma50'])}."
    elif intent == "sma20":
        answer = f"The 20-day moving average (SMA20) is {fmt(s['sma20'])}."
    elif intent == "sma50":
        answer = f"The 50-day moving average (SMA50) is {fmt(s['sma50'])}."
    elif intent == "rsi_value":
        answer = f"The current RSI(14) is {fmt(s['rsi14'])}."
    elif intent == "rsi_meaning":
        answer = ("RSI (Relative Strength Index) measures the speed and size of recent price "
                   "changes on a 0-100 scale. Above 70 often suggests overbought conditions, "
                   "below 30 often suggests oversold conditions.")
    elif intent == "macd_value":
        answer = f"MACD is {fmt(s['macd'])}, with a signal line of {fmt(s['macd_signal'])} and histogram of {fmt(s['macd_histogram'])}."
    elif intent == "macd_meaning":
        answer = ("MACD (Moving Average Convergence Divergence) shows the relationship between "
                   "two EMAs of price. When MACD crosses above its signal line it can indicate "
                   "bullish momentum, and below it can indicate bearish momentum.")
    elif intent == "bollinger_meaning":
        answer = (f"Bollinger Bands plot a moving average with upper/lower bands based on volatility. "
                   f"Current upper band: {fmt(s['bollinger_upper'])}, lower band: {fmt(s['bollinger_lower'])}. "
                   f"Price near the upper band can suggest overbought conditions, near the lower band oversold.")
    elif intent == "above_below_ma":
        pos = "above" if b['current_price'] > (s['sma20'] or 0) else "below"
        answer = f"The current price ({fmt(b['current_price'])}) is {pos} its 20-day moving average ({fmt(s['sma20'])})."
    elif intent == "technical_trend":
        answer = f"The current technical signal is {sig['signal']}. " + " ".join(sig['reasons'])

    elif intent in ("predicted_future_price", "predicted_next_day"):
        pred = _try_prediction(df, symbol, horizon_days=1)
        if "error" in pred:
            answer = pred["error"]
        else:
            answer = (f"The model predicts a next-trading-day price of {fmt(pred['predicted_price'])} "
                       f"(current: {fmt(pred['current_price'])}, change: {fmt(pred['predicted_change'])}, "
                       f"{fmt(pred['predicted_change_percent'])}%). {pred['disclaimer']}")
    elif intent == "predicted_next_7_days":
        pred = _try_prediction(df, symbol, horizon_days=7)
        if "error" in pred:
            answer = pred["error"]
        else:
            answer = (f"The model's 7-trading-day-ahead price estimate is {fmt(pred['predicted_price'])} "
                       f"(current: {fmt(pred['current_price'])}). {pred['disclaimer']}")
    elif intent == "model_used":
        answer = "The primary prediction model is a Random Forest Regressor (Linear Regression is available for comparison)."
    elif intent == "how_prediction_generated":
        answer = ("The prediction pipeline: historical data -> cleaning -> feature engineering "
                   "(technical indicators + lag features) -> chronological train/test split -> "
                   "model training -> evaluation -> future price estimate.")
    elif intent == "model_accuracy":
        pred = _try_prediction(df, symbol, horizon_days=1)
        if "error" in pred:
            answer = pred["error"]
        else:
            m = pred["metrics"]
            answer = (f"On held-out (chronologically later) test data, the model achieves "
                       f"R² = {m['r2_score']}, RMSE = {m['rmse']}, MAE = {m['mae']}. "
                       f"These are calculated from real test data, not fixed numbers.")
    elif intent == "rmse_value":
        pred = _try_prediction(df, symbol, horizon_days=1)
        answer = pred["error"] if "error" in pred else f"The RMSE on the test set is {pred['metrics']['rmse']}."
    elif intent == "mae_value":
        pred = _try_prediction(df, symbol, horizon_days=1)
        answer = pred["error"] if "error" in pred else f"The MAE on the test set is {pred['metrics']['mae']}."
    elif intent == "r2_value":
        pred = _try_prediction(df, symbol, horizon_days=1)
        answer = pred["error"] if "error" in pred else f"The R² score on the test set is {pred['metrics']['r2_score']}."
    elif intent == "prediction_guarantee":
        answer = ("No. The prediction is a statistical estimate based on historical patterns. "
                   "Markets are influenced by many unpredictable factors, so no prediction can be guaranteed.")

    elif intent == "is_volatile":
        vol = stats.historical_volatility(df)
        answer = f"Annualised volatility is {vol}%, which is {stats.risk_level_from_volatility(vol)} relative to typical equities."
    elif intent == "risk_level":
        vol = stats.historical_volatility(df)
        answer = f"Based on historical volatility ({vol}%), the risk level is classified as {stats.risk_level_from_volatility(vol)}."
    elif intent == "trend_bullish_bearish":
        answer = f"The current technical trend is {sig['signal']}. " + " ".join(sig['reasons'])
    elif intent == "prediction_factors":
        answer = ("The model uses OHLCV data, SMA/EMA, RSI, MACD, Bollinger Bands, daily returns, "
                   "volatility and lagged closing prices (1, 2, 3, 5, 10 days back) as input features.")
    elif intent == "why_inaccurate":
        answer = ("Stock predictions can be inaccurate because markets are affected by news, "
                   "macroeconomic events, investor sentiment, and randomness that historical "
                   "price data alone cannot capture. The model only learns statistical patterns.")
    elif intent == "historical_data_used":
        answer = f"Daily OHLCV (Open, High, Low, Close, Volume) data from {b['start_date']} to {b['end_date']} was used, sourced via yfinance."
    elif intent == "training_records_count":
        pred = _try_prediction(df, symbol, horizon_days=1)
        answer = pred["error"] if "error" in pred else f"The model was trained on {pred['training_rows']} rows and tested on {pred['testing_rows']} rows (chronological split)."
    elif intent == "important_indicators":
        pred = _try_prediction(df, symbol, horizon_days=1)
        if "error" in pred or not pred.get("feature_importance"):
            answer = "Feature importance is unavailable for this run, but the model uses SMA, EMA, RSI, MACD, Bollinger Bands and lagged prices."
        else:
            top = ", ".join(f"{f['feature']} ({f['importance']})" for f in pred["feature_importance"][:5])
            answer = f"The most important features for this stock's prediction are: {top}."
    elif intent == "buy_or_sell":
        vol = stats.historical_volatility(df)
        answer = (f"I can't tell you to buy or sell - that would be financial advice. Here's the educational picture: "
                   f"technical signal is {sig['signal']}, RSI is {fmt(s['rsi14'])}, MACD vs signal is "
                   f"{'bullish' if (s['macd'] or 0) > (s['macd_signal'] or 0) else 'bearish'}, "
                   f"and annualised volatility is {vol}% ({stats.risk_level_from_volatility(vol)} risk). "
                   f"Please do your own research or consult a licensed financial advisor before making decisions.")
    elif intent == "full_summary":
        pred = _try_prediction(df, symbol, horizon_days=1)
        pred_line = ""
        if "error" not in pred:
            pred_line = (f" The model estimates a next-day price of {fmt(pred['predicted_price'])} "
                         f"(R²={pred['metrics']['r2_score']}, RMSE={pred['metrics']['rmse']}).")
        vol = stats.historical_volatility(df)
        answer = (f"{name} ({symbol}) is currently trading at {fmt(b['current_price'])} "
                   f"({fmt(b['percent_change'])}% today). Over the loaded period it ranged from "
                   f"{fmt(b['period_low'])} to {fmt(b['period_high'])}, with a total return of "
                   f"{stats.percentage_return(df)}%. RSI is {fmt(s['rsi14'])} and the technical "
                   f"signal is {sig['signal']}. Annualised volatility is {vol}% "
                   f"({stats.risk_level_from_volatility(vol)} risk).{pred_line} {DISCLAIMER}")

    if answer is None:
        answer = (
            "I'm not sure I understood that. Try asking about the current price, RSI, MACD, "
            "moving averages, volatility, the ML prediction, or say 'summary' for a full overview."
        )

    if "financial advice" not in answer and intent in ("buy_or_sell",):
        pass  # disclaimer already embedded above

    return {"intent": intent, "answer": answer, "symbol": symbol}


SUGGESTED_QUESTIONS = [phrase.capitalize() + "?" for _id, phrases in INTENT_DEFINITIONS for phrase in phrases[:1]]