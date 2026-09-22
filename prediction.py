"""
services/prediction.py

Trains a RandomForestRegressor (with LinearRegression available for
comparison) on real, chronologically-split historical data and produces
a future price estimate with genuine evaluation metrics. No accuracy or
price is ever hard-coded.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.config import Config
from services.preprocessing import (
    build_feature_dataframe,
    chronological_train_test_split,
    FEATURE_COLUMNS,
)


class PredictionError(Exception):
    pass


def _model_path(symbol: str, model_name: str) -> str:
    return os.path.join(Config.TRAINED_MODELS_DIR, f"{symbol.upper()}_{model_name}.joblib")


def _build_model(model_name: str):
    if model_name == "linear_regression":
        return LinearRegression()
    # Default / primary model
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
    )


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_and_predict(raw_df: pd.DataFrame, symbol: str, model_name: str = "random_forest",
                       horizon_days: int = None) -> dict:
    """
    Full pipeline: feature engineering -> chronological split -> train ->
    evaluate -> predict the next `horizon_days`-ahead closing price.

    Returns a dict with the prediction, real evaluation metrics, and
    feature importance (for RandomForest) so the result is explainable.
    """
    horizon_days = horizon_days or Config.PREDICTION_HORIZON_DAYS
    model_df = build_feature_dataframe(raw_df, horizon_days=horizon_days)

    if len(model_df) < Config.MIN_ROWS_REQUIRED:
        raise PredictionError(
            f"Not enough historical data to train a reliable model. "
            f"Need at least {Config.MIN_ROWS_REQUIRED} usable rows after "
            f"feature engineering, but only {len(model_df)} are available. "
            f"Try selecting a longer historical period."
        )

    X_train, X_test, y_train, y_test = chronological_train_test_split(model_df, Config.TEST_SIZE)

    if len(X_test) == 0 or len(X_train) == 0:
        raise PredictionError("Not enough data to create a train/test split.")

    model = _build_model(model_name)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred_test)
    mape = _mape(y_test, y_pred_test)

    # Predict forward from the most recent available feature row.
    latest_features = model_df[FEATURE_COLUMNS].iloc[[-1]]
    future_price = float(model.predict(latest_features)[0])
    current_price = float(raw_df["Close"].iloc[-1])

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        pairs = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        feature_importance = [{"feature": f, "importance": round(float(v), 5)} for f, v in pairs[:8]]

    # Persist the trained model for potential reuse (not required for a fresh request,
    # but demonstrates a real, reusable training artifact for the demo/viva).
    try:
        os.makedirs(Config.TRAINED_MODELS_DIR, exist_ok=True)
        joblib.dump(model, _model_path(symbol, model_name))
    except Exception:
        pass

    return {
        "symbol": symbol.upper(),
        "model_used": "Random Forest Regressor" if model_name != "linear_regression" else "Linear Regression",
        "current_price": round(current_price, 4),
        "predicted_price": round(future_price, 4),
        "predicted_change": round(future_price - current_price, 4),
        "predicted_change_percent": round(((future_price - current_price) / current_price) * 100, 4) if current_price else 0,
        "prediction_horizon_days": horizon_days,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "metrics": {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 4) if not np.isnan(mape) else None,
        },
        "feature_importance": feature_importance,
        "disclaimer": (
            "This prediction is an estimate based on historical patterns learned "
            "by a machine-learning model. It is not guaranteed and should not be "
            "treated as financial advice."
        ),
    }