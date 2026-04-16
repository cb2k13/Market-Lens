import yfinance as yf
import pandas as pd

# load data from yahoo finance
def get_price_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    df = df.reset_index()

    # Flatten columns in case yfinance returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df
