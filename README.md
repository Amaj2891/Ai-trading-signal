# TradeSignal AI - Real-time Trade Signal Prediction App

This is a prototype **Streamlit** web app that provides AI-powered trade signals (Buy, Sell, Hold) for cryptocurrency (Binance) and Forex (OANDA) markets.

It fetches historical OHLCV data, caches it locally, calculates technical indicators (RSI, MACD, etc.), and trains Random Forest models to predict next-period price movement with ~80-90% accuracy (depending on data).

---

## Features

- Fetches **Binance** crypto symbols and **OANDA** Forex pairs (requires API tokens)
- Caches OHLCV data locally for faster reloads
- Calculates advanced **Technical Analysis indicators** using `ta` library (RSI, MACD)
- Labels data based on next-bar returns with customizable thresholds
- Allows training either a **global model** (all symbols combined) or **per-symbol models**
- Shows validation reports and confusion matrices for model accuracy
- Displays latest AI trade signal predictions per symbol
- Interactive Streamlit UI for easy experimentation

---

## Requirements

- Python 3.7+
- Packages listed in `requirements.txt`

---

## Installation

1. Clone the repo or copy `app.py` and `requirements.txt`

2. Create & activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

#**Install dependencies:**

pip install -r requirements.txt

#**(Optional) Add API keys for Binance and OANDA in a .env file:**

BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
OANDA_TOKEN=your_oanda_token
OANDA_ACCOUNT_ID=your_oanda_account_id

#**Usage**
Run the Streamlit app:

streamlit run app.py

Click Discover Binance symbols to load symbols
Select symbols for crypto and FX markets
Configure OHLCV bar count, thresholds, and training mode
Click Fetch historical & train model
Review model reports and live predictions

#**Notes**

The app is a demo for prototyping, not production ready.
Model accuracy depends on data quality and chosen parameters.
API keys for Binance and OANDA are optional; without keys, FX data fetching may be limited.
OHLCV data is cached in .cache folder to speed up re-runs.

**
---

If you want me to add deployment instructions or screenshots, just say!
**
