# stocks.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from algorithms import manual_linear_regression

#Fetch Price

def fetch_current_price(symbol: str, fallback: float | None = None) -> float | None:

    if not symbol:
        return fallback

    try:
        df = yf.Ticker(symbol).history(period="1d", auto_adjust=True)
        if df.empty:
            return fallback
        return float(df["Close"].iloc[-1])
    except Exception:
        return fallback


def fetch_history(symbol: str, period: str = "1mo") -> pd.DataFrame:

    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_long_history(symbol: str, years: int = 2) -> pd.DataFrame:

    try:
        df = yf.Ticker(symbol).history(period=f"{years}y", auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def stock_snapshot(stock: dict) -> dict:


    symbol = stock.get("symbol")
    shares = stock.get("shares", 1)

    purchase_price = stock.get("purchase_price", stock.get("price", 0))

    # Best available current price
    current_price = (
        stock.get("current_price")
        or fetch_current_price(symbol, fallback=purchase_price)
    )

    # Calculations
    purchase_value = purchase_price * shares
    current_value = current_price * shares
    gain = current_value - purchase_value
    gain_pct = (gain / purchase_value * 100) if purchase_value > 0 else 0

    return {
        "purchase_price": float(purchase_price),
        "current_price": float(current_price),
        "purchase_value": float(purchase_value),
        "current_value": float(current_value),
        "gain": float(gain),
        "gain_pct": float(gain_pct),
    }


def portfolio_summary(stocks: list[dict]) -> dict:


    total_purchase = 0.0
    total_current = 0.0

    for s in stocks:
        snap = stock_snapshot(s)
        total_purchase += snap["purchase_value"]
        total_current += snap["current_value"]

    gain = total_current - total_purchase
    gain_pct = (gain / total_purchase * 100) if total_purchase > 0 else 0

    return {
        "total_purchase": total_purchase,
        "total_current": total_current,
        "gain": gain,
        "gain_pct": gain_pct,
    }

#Linear Regression

def linear_prediction(price_series: pd.Series, future_days: int = 365) -> dict | None:

    if price_series is None or len(price_series) < 30:
        return None

    # Convert pandas Series to list for custom algorithm
    y_values = price_series.values.tolist()
    
    
    slope, intercept = manual_linear_regression(y_values)

    # Generate regression line for historical data
    n = len(y_values)
    reg_line = [slope * i + intercept for i in range(n)]

    # Future predictions
    future_predictions = [
        slope * (n + i) + intercept 
        for i in range(future_days)
    ]

    predicted_price = future_predictions[-1]
    current_price = y_values[-1]

    # Calculate R² (model quality metric)
    # R² = 1 - (SS_res / SS_tot)
    y_mean = sum(y_values) / len(y_values)
    ss_res = sum((y_values[i] - reg_line[i]) ** 2 for i in range(n))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
        "future_predictions": future_predictions,
        "reg_line": reg_line,
        "r_squared": float(r_squared),
    }


class StockPredictor:
    def __init__(self, symbol: str):

        self._symbol = symbol
        self._price_history = None
        self._prediction = None
    
    @property
    def symbol(self):
        return self._symbol
    
    @property
    def has_prediction(self):
        return self._prediction is not None
    
    def load_history(self, years: int = 2):

        self._price_history = fetch_long_history(self._symbol, years=years)
        return not self._price_history.empty
    
    def predict(self, future_days: int = 365):

        if self._price_history is None or self._price_history.empty:
            if not self.load_history():
                return None
        
        price_series = self._price_history["Close"].dropna()
        
        if len(price_series) < 30:
            return None
        
        self._prediction = linear_prediction(price_series, future_days)
        return self._prediction
    
    def get_current_price(self):
        if self._prediction:
            return self._prediction.get("current_price")
        return fetch_current_price(self._symbol)
    
    def get_predicted_price(self):

        if not self._prediction:
            self.predict()
        return self._prediction.get("predicted_price") if self._prediction else None
    
    def get_prediction_summary(self):

        if not self._prediction:
            self.predict()
        
        if not self._prediction:
            return {"error": "Unable to generate prediction"}
        
        current = self._prediction["current_price"]
        predicted = self._prediction["predicted_price"]
        change = predicted - current
        pct = (change / current * 100) if current > 0 else 0
        
        return {
            "symbol": self._symbol,
            "current_price": current,
            "predicted_price": predicted,
            "change": change,
            "change_percent": pct,
            "trend": "Upward" if change > 0 else "Downward",
            "r_squared": self._prediction.get("r_squared", 0)
        }