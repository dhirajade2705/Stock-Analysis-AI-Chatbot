# Stock Analysis & Prediction AI Chatbot

## Abstract
This project is a final-year stock analysis and prediction dashboard that combines real historical stock data, technical indicators, machine-learning prediction, and an intent-based chatbot. It runs locally in Python using Flask and is designed for educational and research purposes.

## Features
- Search and analyze stock symbols such as AAPL, MSFT, GOOGL, NVDA and Indian tickers like RELIANCE.NS
- Fetch historical data using yfinance
- Display OHLCV history and chart visualizations
- Compute SMA, EMA, RSI, MACD, Bollinger Bands, returns and volatility
- Train a machine-learning model for price prediction
- Present prediction metrics and a technical signal
- Answer stock-related questions through a local chatbot
- REST API and responsive dashboard interface

## Technology Stack
- Python
- Flask
- Flask-CORS
- yfinance
- pandas
- NumPy
- scikit-learn
- ta
- joblib
- Chart.js
- Bootstrap 5
- pytest

## Folder Structure
```text
DBMS/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── models/
│   ├── trained/
│   └── scalers/
├── services/
│   ├── __init__.py
│   ├── stock_data.py
│   ├── indicators.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── chatbot.py
│   └── statistics.py
├── api/
│   ├── __init__.py
│   ├── stock_routes.py
│   ├── prediction_routes.py
│   └── chatbot_routes.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── tests/
│   └── ...
├── notebooks/
├── docs/
└── .venv/
```

## Requirements
Install the packages from `requirements.txt`.

## Installation
### Windows
```powershell
python --version
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Then open: http://127.0.0.1:5000

## API Endpoints
- GET `/api/health`
- GET `/api/stock/<symbol>`
- GET `/api/history/<symbol>`
- GET `/api/indicators/<symbol>`
- GET `/api/prediction/<symbol>`
- POST `/api/chat`

## ML Methodology
The model uses historical price and technical-indicator data, applies feature engineering with lag features, performs chronological train/test splitting, trains a Random Forest regressor, and evaluates using MAE, MSE, RMSE, and R².

## Chatbot
The chatbot uses keyword-based intent detection and real stock data to answer questions about price, trend, indicators, predictions, and risk.

## Technical Indicators
- SMA 20 and SMA 50
- EMA 20 and EMA 50
- RSI 14
- MACD and signal line
- Bollinger Bands
- Daily return and cumulative return
- Volatility

## Testing
```powershell
pytest
```

## Troubleshooting
- If yfinance fails, verify that internet access works.
- If a stock is invalid, use a valid symbol or exchange suffix.
- If packages are missing, reinstall from `requirements.txt`.

## GitHub Upload
```powershell
git init
git add .
git commit -m "Initial Stock Analysis AI Chatbot"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
Do not push `.env` files or secret data.

## Limitations
- Predictions are estimates based on past patterns.
- Market behavior can change suddenly.
- Model accuracy depends on the available history and data quality.

## Future Scope
- Add sentiment analysis
- Expand to forex and crypto
- Add login and portfolio tracking
- Improve model with XGBoost or time-series models

## Disclaimer
This application provides estimates based on historical market data and machine-learning models. Stock-price predictions are uncertain and should not be considered financial advice or guaranteed future prices.
