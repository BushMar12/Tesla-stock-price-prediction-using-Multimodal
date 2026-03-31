"""
Technical indicators calculation for stock data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add Simple and Exponential Moving Averages"""
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [12, 26, 50]:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
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
    """Add target variables for prediction"""
    # Future price (regression target)
    df['Target_Price'] = df['close'].shift(-horizon)
    
    # Future return (for scaling)
    df['Target_Return'] = df['close'].pct_change(periods=horizon).shift(-horizon)
    
    # Direction (classification target)
    df['Target_Direction'] = np.where(
        df['Target_Return'] > 0.01, 2,  # Up (> 1%)
        np.where(df['Target_Return'] < -0.01, 0, 1)  # Down (< -1%) or Neutral
    )
    
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
    
    # Add all indicators
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_stochastic(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_price_patterns(df)
    
    if add_targets:
        df = add_target_variables(df)
    
    print(f"Added {len(df.columns)} features")
    
    return df


if __name__ == "__main__":
    from src.data.stock_data import fetch_stock_data
    
    # Test the indicators
    stock_df = fetch_stock_data()
    df_with_indicators = calculate_all_indicators(stock_df)
    
    print(df_with_indicators.head())
    print(f"\nFeatures: {df_with_indicators.columns.tolist()}")
