"""
api/stock_routes.py

Endpoints:
    GET /api/stock/<symbol>?period=1y
    GET /api/history/<symbol>?period=1y
    GET /api/indicators/<symbol>?period=1y
"""

from flask import Blueprint, jsonify, request

from services import stock_data, indicators
from services.stock_data import StockDataError

stock_bp = Blueprint("stock_bp", __name__)


def _period():
    return request.args.get("period", "1y")


@stock_bp.route("/api/stock/<symbol>", methods=["GET"])
def get_stock(symbol):
    try:
        period = _period()
        df = stock_data.get_historical_data(symbol, period)
        info = stock_data.get_company_info(symbol)
        basic = stock_data.get_basic_stats(df)
        return jsonify({"success": True, "info": info, "stats": basic, "period": period})
    except StockDataError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500


@stock_bp.route("/api/history/<symbol>", methods=["GET"])
def get_history(symbol):
    try:
        period = _period()
        df = stock_data.get_historical_data(symbol, period)
        records = stock_data.dataframe_to_records(df)
        return jsonify({"success": True, "symbol": symbol.upper(), "period": period, "history": records})
    except StockDataError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500


@stock_bp.route("/api/indicators/<symbol>", methods=["GET"])
def get_indicators(symbol):
    try:
        period = _period()
        df = stock_data.get_historical_data(symbol, period)
        df_ind = indicators.add_all_indicators(df)
        snapshot = indicators.get_latest_indicator_snapshot(df_ind)
        chart_series = indicators.indicator_series_for_charts(df_ind)
        basic = stock_data.get_basic_stats(df)
        signal = indicators.generate_technical_signal(snapshot, basic["current_price"])
        return jsonify({
            "success": True,
            "symbol": symbol.upper(),
            "period": period,
            "snapshot": snapshot,
            "series": chart_series,
            "signal": signal,
        })
    except StockDataError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500