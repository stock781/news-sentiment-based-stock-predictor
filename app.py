# Importing necessary libraries specific for my task
from scipy.optimize import direct
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]


# Using NewsAPI to acquire headlines pertaining to the stock the user is interested in
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
    with open("output.json", "w") as f:
        json.dump(data, f, indent=4)

    titles = [article['title'] for article in data.get('articles', [])]
    return titles


# Uses the TextBlob library to determine the sentiment of a piece of text
def sentiment_score(text):
    return TextBlob(text).sentiment.polarity


# This function computes the means of the sentiment scores of the news headline titles acquired from NewsAPI
def compute_sentiment_features(titles):
    if len(titles) == 0:
        return 0
    scores = [sentiment_score(t) for t in titles]
    return np.mean(scores)


# These indicators will help the model predict the direction of the stock (this part is all numbers)
def add_indicators(df):
    df["return"] = df["Close"].pct_change()

    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()

    df["ema10"] = df["Close"].ewm(span=10).mean()
    df["ema20"] = df["Close"].ewm(span=20).mean()

    df["rsi"] = compute_rsi(df["Close"], 14)

    df["volatility"] = df["return"].rolling(10).std()
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()

    df["volume_change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)

    return df


# This calculates average gains and losses
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss  # ratio of avg gain to loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# downloads 5 years of daily stock price data for the specific stock
def build_dataset(symbol):
    df = yf.download(symbol, period='5y')
    df = add_indicators(df)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    return df


# this trains the model using the specified indicators
def train_model(df):
    X = df[['return', 'ma10', 'ma20', 'rsi', 'volatility', 'momentum', 'macd', 'signal', 'ema10', 'ema20',
            'volume_change']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        # These parameters were fine tuned for max accuracy
        n_estimators=500,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        min_child_weight=3,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    plot_2024_performance("AAPL", model, scaler,
                          ['return', 'ma10', 'ma20', 'rsi', 'volatility', 'momentum', 'macd', 'signal', 'ema10',
                           'ema20', 'volume_change'])

    return model, accuracy


# Using the indicators, this will use the model to predict the direction of the stock
def predict_next_day(symbol, model, sentiment_value):
    df = yf.download(symbol, period='30d')
    df = add_indicators(df)

    latest = df.iloc[-1]

    base_features = np.array([
        latest['return'],
        latest['ma10'],
        latest['ma20'],
        latest['rsi'],
        latest['volatility'],
        latest['momentum'],
        latest['macd'],
        latest['signal'],
        latest['ema10'],
        latest['ema20'],
        latest['volume_change']
    ])

    combined = np.append(base_features, sentiment_value)

    predict = model.predict(base_features.reshape(1, -1))[0]

    return "UP" if predict == 1 else "DOWN"


# This is the order that everything will run in (line by line)...
def run(symbol):
    titles = get_news_headlines(symbol)
    sentiment_value = compute_sentiment_features(titles)

    df = build_dataset(symbol)
    model, accuracy = train_model(df)

    direction = predict_next_day(symbol, model, sentiment_value)

    return {
        "symbol": symbol,
        "accuracy": accuracy,
        "sentiment": sentiment_value,
        "direction": direction
    }


def plot_2024_performance(symbol, model, scaler, features):
    df = build_dataset(symbol)
    df_2024 = df.loc["2024-01-01":"2024-12-31"].copy()

    X_2024 = df_2024[features].values
    X_2024_scaled = scaler.transform(X_2024)

    preds = model.predict(X_2024_scaled)
    df_2024["pred"] = preds

    plt.figure(figsize=(14, 6))
    plt.plot(df_2024.index, df_2024["Close"], label="Actual Adj Close", linewidth=3)

    up_idx = df_2024[df_2024["pred"] == 1].index
    plt.scatter(up_idx, df_2024.loc[up_idx, "Close"],
                marker="^", color="green", s=70, label="Predicted UP")

    down_idx = df_2024[df_2024["pred"] == 0].index
    plt.scatter(down_idx, df_2024.loc[down_idx, "Close"],
                marker="v", color="red", s=70, label="Predicted DOWN")

    plt.title(f"{symbol} — 2024 Price + Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()


# This is what the end user will see
print(run(input("Enter the stock that you want to predict (For example, try AAPL or TSLA): ")))
print("This tool is for educational purposes only and is not financial advice.")
