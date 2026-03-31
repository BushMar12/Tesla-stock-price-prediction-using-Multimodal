"""
Stock data fetcher using Yahoo Finance
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import STOCK_SYMBOL, START_DATE, END_DATE, RAW_DATA_DIR


def fetch_stock_data(
    symbol: str = STOCK_SYMBOL,
    start_date: str = START_DATE,
    end_date: str = None,
    save: bool = True
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        save: Whether to save the data to disk
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Clean up the dataframe
    df = df.reset_index()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Ensure we have the date column properly formatted
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    # Select relevant columns
    columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Add additional features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = df['price_range'] / df['close']
    
    # Drop first row with NaN returns
    df = df.dropna()
    
    if save:
        save_path = RAW_DATA_DIR / f"{symbol}_historical.csv"
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    print(f"Fetched {len(df)} rows of data")
    return df


def get_latest_data(symbol: str = STOCK_SYMBOL, days: int = 5) -> pd.DataFrame:
    """
    Get the most recent stock data.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of recent days to fetch
    
    Returns:
        DataFrame with recent OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days * 2)  # Fetch extra for weekends
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"))
    
    df = df.reset_index()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    return df.tail(days)


def load_stock_data(symbol: str = STOCK_SYMBOL) -> pd.DataFrame:
    """
    Load saved stock data from disk.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        DataFrame with historical data
    """
    file_path = RAW_DATA_DIR / f"{symbol}_historical.csv"
    
    if not file_path.exists():
        print(f"No saved data found for {symbol}. Fetching new data...")
        return fetch_stock_data(symbol)
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


# Add numpy import
import numpy as np


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_stock_data()
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
