"""
TradeSignal AI Single-file Demo with:
- Binance + OANDA symbol discovery
- OHLCV caching in .cache folder per symbol (CSV)
- Added TA indicators (RSI, MACD) using `ta` lib
- Option to train Global model or Per-symbol models
- Streamlit UI

Run:
  pip install streamlit ccxt pandas numpy scikit-learn requests python-dotenv ta
  streamlit run app.py
"""

import os
import time
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BINANCE_API_KEY = os.getenv("5ROkMdWucL9jk7d5FtBpVnq39AQH6goRW6glt21XRB9LoKh4s1m4TU4jn1sn6qtq", "")
BINANCE_SECRET = os.getenv("NiHnMivXGmPGGOvOjZp3aIrBmvfRNeYIrtglEQpcFJEkHnwexI6P5I6Kc4lE4Eg8", "")
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")

DEFAULT_OHLCV_LIMIT = 200
NEXT_RET_BUY_THRESH = 0.0008
NEXT_RET_SELL_THRESH = -0.0008

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="TradeSignal AI - Demo with TA and Cache", layout="wide")

st.title("TradeSignal AI — Demo with TA indicators, caching & model options")
st.markdown("""
This demo discovers Binance + OANDA FX symbols, caches OHLCV locally, engineers advanced TA features (RSI, MACD),
and lets you choose between training a global model (all symbols) or individual per-symbol models.

**Not production-ready; for prototyping only.**
""")

col1, col2 = st.columns([2,1])

with col2:
    st.header("Settings")
    api_info = st.expander("API keys (optional)")
    with api_info:
        bin_key = st.text_input("BINANCE_API_KEY", BINANCE_API_KEY, type="password")
        bin_secret = st.text_input("BINANCE_SECRET", BINANCE_SECRET, type="password")
        oanda_token = st.text_input("OANDA_TOKEN", OANDA_TOKEN, type="password")
        oanda_account = st.text_input("OANDA_ACCOUNT_ID", OANDA_ACCOUNT_ID)
        limit = st.number_input("OHLCV bars per symbol", min_value=50, max_value=1000, value=DEFAULT_OHLCV_LIMIT)
        sample_size = st.number_input("Max symbols to fetch/train on (small for demo)", min_value=1, max_value=100, value=10)
        st.markdown("**Label thresholds (next bar return)**")
        buy_t = st.number_input("Buy threshold (fraction)", value=NEXT_RET_BUY_THRESH, format="%.6f")
        sell_t = st.number_input("Sell threshold (fraction)", value=NEXT_RET_SELL_THRESH, format="%.6f")
        train_mode = st.selectbox("Training mode", options=["Global model (all symbols)", "Per-symbol models"])
        
with col1:
    st.header("Discover & select symbols")
    discover_btn = st.button("Discover Binance symbols")
    if "binance_symbols" not in st.session_state:
        st.session_state.binance_symbols = []
    if discover_btn or not st.session_state.binance_symbols:
        try:
            exch = ccxt.binance({"enableRateLimit": True})
            if bin_key:
                exch.apiKey = bin_key
                exch.secret = bin_secret
            info = exch.fetch_markets()
            symbols = [m['symbol'] for m in info if (m.get('active', m.get('status','TRADING')) in (True, 'TRADING'))]
            symbols = sorted(list(set(symbols)))
            st.session_state.binance_symbols = symbols
            st.success(f"Found {len(symbols)} Binance symbols (spot).")
        except Exception as e:
            st.error(f"Binance discovery error: {e}")
            st.session_state.binance_symbols = []

    oanda_available = bool(oanda_token and oanda_account)
    if oanda_available:
        st.success("OANDA credentials provided — FX discovery enabled.")
    else:
        st.warning("OANDA token/account not provided. FX discovery disabled; sample FX pairs available.")

    st.write("Select symbols to fetch & train (crypto and FX). Keep selection small for demo.")
    bin_samples = st.multiselect("Binance symbols", options=st.session_state.binance_symbols, default=st.session_state.binance_symbols[:min(20,len(st.session_state.binance_symbols))])
    fx_default = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    fx_samples = st.multiselect("FX symbols (OANDA instruments)", options=fx_default, default=fx_default[:3])
    fetch_button = st.button("Fetch historical & train model")

# --- Helper functions ---

def cache_path(symbol: str) -> str:
    safe_sym = symbol.replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe_sym}.csv")

def is_cache_fresh(symbol: str, max_age_minutes=10) -> bool:
    path = cache_path(symbol)
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(minutes=max_age_minutes):
        return False
    return True

def save_cache(df: pd.DataFrame, symbol: str):
    df.to_csv(cache_path(symbol), index=False)

def load_cache(symbol: str) -> pd.DataFrame:
    path = cache_path(symbol)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['t'])
    return None

def fetch_binance_ohlcv(symbol: str, limit: int = 200, timeframe: str = '1m', api_key=None, api_secret=None) -> pd.DataFrame:
    if is_cache_fresh(symbol):
        df = load_cache(symbol)
        if df is not None:
            return df
    exch = ccxt.binance({"enableRateLimit": True})
    if api_key:
        exch.apiKey = api_key
        exch.secret = api_secret
    bars = exch.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    save_cache(df, symbol)
    return df

def fetch_oanda_ohlcv(instrument: str, count: int = 200, granularity='M1', token: str = None) -> pd.DataFrame:
    if is_cache_fresh(instrument):
        df = load_cache(instrument)
        if df is not None:
            return df
    if not token:
        raise RuntimeError("OANDA token not provided")
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for c in data.get("candles", []):
        mid = c.get("mid")
        if not mid:
            continue
        t = pd.to_datetime(c['time'])
        o = float(mid['o']); h = float(mid['h']); l = float(mid['l']); c_ = float(mid['c'])
        volume = float(c.get('volume', 0))
        rows.append([t, o, h, l, c_, volume])
    df = pd.DataFrame(rows, columns=['t','o','h','l','c','v'])
    save_cache(df, instrument)
    return df

def featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df['ret1'] = df['c'].pct_change().fillna(0)
    df['ema5'] = df['c'].ewm(span=5, adjust=False).mean()
    df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
    df['ema_diff'] = df['ema5'] - df['ema12']
    df['vol_rolling'] = df['v'].rolling(10, min_periods=1).mean()
    df['std5'] = df['c'].rolling(5, min_periods=1).std().fillna(0)
    df['mom3'] = df['c'] - df['c'].shift(3)

    # TA indicators with ta library
    # RSI (14)
    df['rsi'] = ta.momentum.RSIIndicator(df['c'], window=14, fillna=True).rsi()
    # MACD line & Signal line
    macd = ta.trend.MACD(df['c'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    df = df.fillna(0)
    return df

def create_labels(df: pd.DataFrame, buy_thresh: float, sell_thresh: float):
    df = df.copy()
    df['next_ret'] = df['c'].shift(-1) / df['c'] - 1.0
    def label(r):
        if r > buy_thresh:
            return "buy"
        elif r < sell_thresh:
            return "sell"
        else:
            return "hold"
    df['label'] = df['next_ret'].apply(lambda x: label(x if not pd.isna(x) else 0.0))
    return df

def prepare_dataset(symbol_dfs: Dict[str, pd.DataFrame], buy_thresh: float, sell_thresh: float, use_per_symbol=False):
    """
    If use_per_symbol=True, returns dict[symbol] = (X, y)
    Else returns combined X,y,meta
    """
    if use_per_symbol:
        models_data = {}
        for sym, df in symbol_dfs.items():
            if df.shape[0] < 30:
                continue
            df = df.sort_values('t').reset_index(drop=True)
            df_feat = featurize_df(df)
            df_lab = create_labels(df_feat, buy_thresh, sell_thresh)
            feats = ['c', 'ret1', 'ema_diff', 'vol_rolling', 'std5', 'mom3', 'rsi', 'macd', 'macd_signal']
            df_used = df_lab.iloc[:-1]
            X = df_used[feats].values
            y = df_used['label'].values
            models_data[sym] = (X, y)
        return models_data
    else:
        X_list = []
        y_list = []
        meta = []
        for sym, df in symbol_dfs.items():
            if df.shape[0] < 30:
                continue
            df = df.sort_values('t').reset_index(drop=True)
            df_feat = featurize_df(df)
            df_lab = create_labels(df_feat, buy_thresh, sell_thresh)
            feats = ['c', 'ret1', 'ema_diff', 'vol_rolling', 'std5', 'mom3', 'rsi', 'macd', 'macd_signal']
            df_used = df_lab.iloc[:-1]
            X = df_used[feats].values
            y = df_used['label'].values
            X_list.append(X)
            y_list.append(y)
            meta += [sym] * len(y)
        if not X_list:
            return None, None, None
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all, meta

# App state for models & data
if "model_global" not in st.session_state:
    st.session_state.model_global = None
if "model_per_symbol" not in st.session_state:
    st.session_state.model_per_symbol = {}
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = {}
if "symbol_dfs" not in st.session_state:
    st.session_state.symbol_dfs = {}

if fetch_button:
    selected_symbols = []
    for s in bin_samples:
        selected_symbols.append(("crypto", s))
    for s in fx_samples:
        selected_symbols.append(("fx", s))
    if len(selected_symbols) == 0:
        st.error("Select at least one symbol to fetch and train on.")
    else:
        selected_symbols = selected_symbols[:int(sample_size)]
        st.info(f"Fetching OHLCV for {len(selected_symbols)} symbols (limit {limit}) — this may take a while.")
        progress = st.progress(0)
        total = len(selected_symbols)
        symbol_dfs = {}
        errors = []
        for i, (mtype, sym) in enumerate(selected_symbols):
            try:
                if mtype == "crypto":
                    df = fetch_binance_ohlcv(sym, limit=int(limit), api_key=bin_key, api_secret=bin_secret)
                else:
                    if not oanda_token:
                        raise RuntimeError("OANDA token not provided; cannot fetch FX OHLCV.")
                    df = fetch_oanda_ohlcv(sym, count=int(limit), token=oanda_token)
                df = df.rename(columns={df.columns[0]:'t'}) if df.columns[0] != 't' else df
                df = df[['t','o','h','l','c','v']]
                symbol_dfs[sym] = df
                st.write(f"Fetched {len(df)} bars for {sym}")
            except Exception as e:
                errors.append((sym, str(e)))
                st.warning(f"Error fetching {sym}: {e}")
            progress.progress((i+1)/total)
            time.sleep(0.3)
        progress.empty()

        if not symbol_dfs:
            st.error("No data fetched; cannot train.")
        else:
            st.session_state.symbol_dfs = symbol_dfs
            if train_mode == "Global model (all symbols)":
                st.write("Preparing global dataset...")
                X, y, meta = prepare_dataset(symbol_dfs, float(buy_t), float(sell_t), use_per_symbol=False)
                if X is None:
                    st.error("Insufficient data after preparation.")
                else:
                    st.write("Dataset shapes:", X.shape, y.shape)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))])
                    with st.spinner("Training global RandomForest..."):
                        pipe.fit(Xtr, ytr)
                    st.success("Global model trained.")
                    st.session_state.model_global = pipe
                    ypred = pipe.predict(Xte)
                    report = classification_report(yte, ypred, output_dict=True)
                    st.subheader("Global model validation classification report")
                    st.dataframe(pd.DataFrame(report).transpose())
                    cm = confusion_matrix(yte, ypred, labels=["buy","hold","sell"])
                    st.subheader("Confusion matrix (rows=true, cols=pred) [buy, hold, sell]")
                    st.write(cm)
                    # Latest predictions per symbol
                    st.session_state.latest_prediction = {}
                    for sym, df in symbol_dfs.items():
                        if df.shape[0] < 10:
                            continue
                        dff = featurize_df(df.sort_values('t'))
                        latest = dff.iloc[-1:]
                        feats = ['c', 'ret1', 'ema_diff', 'vol_rolling', 'std5', 'mom3', 'rsi', 'macd', 'macd_signal']
                        Xlatest = latest[feats].values
                        pred_proba = pipe.predict_proba(Xlatest)[0]
                        classes = pipe.named_steps['rf'].classes_
                        probs = {cls: float(p) for cls,p in zip(classes, pred_proba)}
                        best = max(probs, key=probs.get)
                        st.session_state.latest_prediction[sym] = {"action":best, "probs":probs}
            else:
                # per-symbol models
                st.write("Preparing per-symbol datasets and training models...")
                models_data = prepare_dataset(symbol_dfs, float(buy_t), float(sell_t), use_per_symbol=True)
                st.session_state.model_per_symbol = {}
                st.session_state.latest_prediction = {}
                for sym, (X, y) in models_data.items():
                    if X.shape[0] < 10:
                        continue
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42))])
                    with st.spinner(f"Training model for {sym}..."):
                        pipe.fit(Xtr, ytr)
                    st.session_state.model_per_symbol[sym] = pipe
                    ypred = pipe.predict(Xte)
                    report = classification_report(yte, ypred, output_dict=True)
                    st.write(f"Model {sym} validation report:")
                    st.dataframe(pd.DataFrame(report).transpose())
                    cm = confusion_matrix(yte, ypred, labels=["buy","hold","sell"])
                    st.write(f"Confusion matrix for {sym}:")
                    st.write(cm)
                    # Predict latest bar
                    df = symbol_dfs[sym]
                    dff = featurize_df(df.sort_values('t'))
                    latest = dff.iloc[-1:]
                    feats = ['c', 'ret1', 'ema_diff', 'vol_rolling', 'std5', 'mom3', 'rsi', 'macd', 'macd_signal']
                    Xlatest = latest[feats].values
                    pred_proba = pipe.predict_proba(Xlatest)[0]
