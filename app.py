import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob
import joblib
import warnings
warnings.filterwarnings('ignore')

# CONFIG 
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.error("Missing NEWSAPI_KEY in Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Stock Direction Predictor", layout="centered")
st.title("Stock Direction Predictor")
st.caption("Educational tool — not financial advice")

# NEWS FUNCTIONS 
def get_news_headlines(symbol, days=5):
    url = 'https://newsapi.org/v2/everything'
    params = {
        "q": symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "from": (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        "apiKey": NEWSAPI_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    titles = [a['title'] for a in data.get('articles', [])]
    return titles

def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def compute_sentiment_features(titles):
    if not titles:
        return 0
    scores = [sentiment_score(t) for t in titles]
    return np.mean(scores)

# INDICATORS 
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df["return"] = df["Close"].pct_change()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volatility"] = df["return"].rolling(10).std()
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()
    df["volume_change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    return df

# DATASET 
def build_dataset(symbol, period='5y'):
    df = yf.download(symbol, period=period, progress=False)
    df = add_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# PREDICTION 
def predict_next_day(symbol, model, scaler, features, sentiment_value):
    df = yf.download(symbol, period='30d', progress=False)
    df = add_indicators(df)
    latest = df.iloc[-1]

    base_features = np.array([
        latest['return'], latest['ma10'], latest['ma20'], latest['rsi'],
        latest['volatility'], latest['momentum'], latest['macd'],
        latest['signal'], latest['ema10'], latest['ema20'], latest['volume_change']
    ])

    X_scaled = scaler.transform(base_features.reshape(1, -1))

    predict_class = model.predict(X_scaled)[0]
    confidence = model.predict_proba(X_scaled)[0][1]  # probability of UP

    direction = "UP" if predict_class == 1 else "DOWN"
    return direction, confidence

# PLOT
def plot_2024_performance(symbol, model, scaler, features):
    df = build_dataset(symbol)
    df_2024 = df.loc["2024-01-01":"2024-12-31"].copy()

    X_2024_scaled = scaler.transform(df_2024[features])
    preds = model.predict(X_2024_scaled)
    df_2024["pred"] = preds

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_2024.index, df_2024["Close"], label="Actual Price", linewidth=3)
    ax.scatter(df_2024[df_2024.pred==1].index, df_2024[df_2024.pred==1]["Close"],
               marker="^", color="green", label="Predicted UP")
    ax.scatter(df_2024[df_2024.pred==0].index, df_2024[df_2024.pred==0]["Close"],
               marker="v", color="red", label="Predicted DOWN")
    ax.set_title(f"{symbol} — 2024 Model Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# LOAD MODEL
model = joblib.load("xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
features = joblib.load("features.joblib")

# STREAMLIT UI 
symbol = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", "AAPL").upper()

if st.button("Run Prediction"):
    with st.spinner("Fetching data and computing prediction..."):
        titles = get_news_headlines(symbol)
        sentiment_value = compute_sentiment_features(titles)

        direction, confidence = predict_next_day(symbol, model, scaler, features, sentiment_value)

        plot_2024_performance(symbol, model, scaler, features)

    st.subheader("Results")
    st.write(f"Predicted Direction: **{direction}**")
    st.write(f"Prediction Confidence: **{confidence:.1%}**")
    st.write(f"Recent News Sentiment Score: {sentiment_value:.3f}")
