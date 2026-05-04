"""
Technical indicators calculation for stock data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    MARKET_CONTEXT_CACHE,
    RAW_DATA_DIR,
    USE_MARKET_CONTEXT,
)


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add Simple and Exponential Moving Averages"""
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [12, 26, 50]:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    return df


def add_selected_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add the simple moving averages used by the current compact feature set."""
    for period in [20, 50]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index"""
    delta = df['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD indicator"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Add Bollinger Bands"""
    df['BB_Middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
    
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume"""
    obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['OBV'] = pd.Series(obv).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
    
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price (rolling approximation)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator"""
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    df['Stoch_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based features"""
    # Price momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['close'].pct_change(periods=period)
    
    # Rate of Change
    for period in [10, 20]:
        df[f'ROC_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    
    # Price acceleration
    df['Price_Acceleration'] = df['returns'].diff()
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based features"""
    # Rolling volatility
    for period in [5, 10, 20]:
        df[f'Volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252)
    
    # Parkinson volatility (using high-low range)
    df['Parkinson_Vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        (np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean()
    ) * np.sqrt(252)
    
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add price pattern features"""
    # Candlestick body
    df['Body'] = df['close'] - df['open']
    df['Body_Pct'] = df['Body'] / df['open']
    
    # Upper and lower shadows
    df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Gap detection
    df['Gap'] = df['open'] - df['close'].shift(1)
    df['Gap_Pct'] = df['Gap'] / df['close'].shift(1)
    
    return df


def add_target_variables(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add next-day price and return targets."""
    df['Target_Price'] = df['close'].shift(-horizon)
    df['Target_Return'] = df['close'].pct_change(periods=horizon).shift(-horizon)
    return df


def _load_or_fetch_market_close(
    symbol: str,
    output_col: str,
    cache_name: str,
    start: str,
    end: str
) -> Optional[pd.DataFrame]:
    """Load market close data from cache, falling back to Yahoo Finance."""
    cache_path = RAW_DATA_DIR / cache_name

    if MARKET_CONTEXT_CACHE and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {'date', output_col}.issubset(cached.columns):
            cached['date'] = pd.to_datetime(cached['date'])
            print(f"Loaded cached {symbol} market context from {cache_path}")
            return cached[['date', output_col]]
        print(f"Warning: Ignoring invalid {symbol} market context cache at {cache_path}")

    try:
        import yfinance as yf

        market_df = yf.download(symbol, start=start, end=end, progress=False)
        if market_df.empty:
            return None

        market_df = market_df.reset_index()
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = [
                col[0] if col[1] == '' or col[1] == symbol else col[0]
                for col in market_df.columns
            ]

        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.tz_localize(None)
        market_df = market_df.rename(columns={'Date': 'date', 'Close': output_col})
        market_df = market_df[['date', output_col]]

        if MARKET_CONTEXT_CACHE:
            market_df.to_csv(cache_path, index=False)
            print(f"Saved {symbol} market context to {cache_path}")

        return market_df
    except Exception as e:
        print(f"Warning: Could not fetch {symbol} data: {e}")
        return None


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add market context features: SPY (S&P 500) and VIX (Volatility Index)."""
    if not USE_MARKET_CONTEXT:
        df['spy_close'] = 0
        df['spy_return'] = 0
        df['tsla_vs_spy'] = 0
        df['vix'] = 20.0
        return df

    start = df['date'].min().strftime('%Y-%m-%d')
    end = df['date'].max().strftime('%Y-%m-%d')

    spy = _load_or_fetch_market_close(
        symbol="SPY",
        output_col="spy_close",
        cache_name="SPY_market_context.csv",
        start=start,
        end=end
    )
    if spy is not None:
        spy['spy_return'] = spy['spy_close'].pct_change()
        df = df.merge(spy[['date', 'spy_close', 'spy_return']], on='date', how='left')
        df['tsla_vs_spy'] = df['returns'] - df['spy_return']  # Alpha
        df[['spy_close', 'spy_return', 'tsla_vs_spy']] = (
            df[['spy_close', 'spy_return', 'tsla_vs_spy']].ffill().fillna(0)
        )
    else:
        print("Warning: SPY data unavailable, using neutral market context")
        df['spy_close'] = 0
        df['spy_return'] = 0
        df['tsla_vs_spy'] = 0

    vix = _load_or_fetch_market_close(
        symbol="^VIX",
        output_col="vix",
        cache_name="VIX_market_context.csv",
        start=start,
        end=end
    )
    if vix is not None:
        df = df.merge(vix[['date', 'vix']], on='date', how='left')
        df['vix'] = df['vix'].ffill().fillna(20.0)
    else:
        print("Warning: VIX data unavailable, using neutral market context")
        df['vix'] = 20.0

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features"""
    dates = pd.to_datetime(df['date'])
    df['day_of_week'] = dates.dt.dayofweek / 4.0  # Normalize to [0, 1]
    df['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
    df['is_month_end'] = dates.dt.is_month_end.astype(int)
    df['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)
    return df


def calculate_all_indicators(df: pd.DataFrame, add_targets: bool = True) -> pd.DataFrame:
    """
    Calculate all technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        add_targets: Whether to add target variables
    
    Returns:
        DataFrame with all indicators added
    """
    print("Calculating technical indicators...")
    
    df = df.copy()
    
    # Current compact feature set: OHLCV + MA + RSI14 + Bollinger Bands.
    df = add_selected_moving_averages(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    
    if add_targets:
        df = add_target_variables(df)
    
    print(f"Added {len(df.columns)} features")
    print(f"Current features: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    from src.data.stock_data import fetch_stock_data
    
    # Test the indicators
    stock_df = fetch_stock_data()
    df_with_indicators = calculate_all_indicators(stock_df)
    
    print(df_with_indicators.head())
    print(f"\nFeatures: {df_with_indicators.columns.tolist()}")
