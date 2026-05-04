"""
Data preprocessing pipeline for multimodal stock prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    SEQUENCE_LENGTH, PREDICTION_HORIZON,
    PROCESSED_DATA_DIR, TRAINING_CONFIG, MODELS_DIR,
    PRICE_FEATURE_COLUMNS
)


class DataPreprocessor:
    """Preprocess and prepare data for training"""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.return_scaler = MinMaxScaler()  # For scaling next-day target returns
        self.feature_columns = None
        self.sentiment_columns = None
        
    def merge_data(
        self,
        stock_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        indicators_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge stock, sentiment, and technical indicators data.
        
        Args:
            stock_df: Stock price data
            sentiment_df: Sentiment data
            indicators_df: Technical indicators data
        
        Returns:
            Merged DataFrame
        """
        # Start with indicators (which includes stock data)
        df = indicators_df.copy()
        
        # Ensure date columns are compatible
        df['date'] = pd.to_datetime(df['date'])
        
        if 'date' in sentiment_df.columns:
            sentiment_df = sentiment_df.copy()
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            
            # Merge sentiment data
            sentiment_cols = [col for col in sentiment_df.columns if col != 'date']
            df = df.merge(sentiment_df, on='date', how='left')
            
            # Fill missing sentiment values
            for col in sentiment_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> tuple:
        """
        Select features for training.
        
        Returns:
            Tuple of (feature_columns, sentiment_columns)
        """
        # Exclude target columns from input features
        exclude_cols = ['date', 'Target_Price', 'Target_Return', 'Target_Return_Scaled']
        
        # All numeric columns except excluded
        all_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        # Identify sentiment columns. news_count comes from the sentiment
        # source, so keep it in the sentiment stream rather than the price stream.
        sentiment_cols = [
            col for col in all_features
            if 'sentiment' in col.lower() or col == 'news_count'
        ]
        
        # Price/technical features are intentionally restricted by config.
        missing_price_cols = [
            col for col in PRICE_FEATURE_COLUMNS
            if col not in df.columns
        ]
        if missing_price_cols:
            print(f"Warning: configured price features missing: {missing_price_cols}")
        other_cols = [
            col for col in PRICE_FEATURE_COLUMNS
            if col in all_features and col not in sentiment_cols
        ]
        
        self.feature_columns = other_cols
        self.sentiment_columns = sentiment_cols
        
        return other_cols, sentiment_cols
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and infinities"""
        df = df.copy()
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        df = df.ffill().bfill()
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        return df
    
    def fit_scalers(self, df: pd.DataFrame):
        """Fit scalers on training data ONLY (no leakage)"""
        if self.feature_columns is None:
            self.select_features(df)
        
        all_features = self.feature_columns + self.sentiment_columns
        
        # Fit feature scaler
        self.feature_scaler.fit(df[all_features])

        # Fit return scaler on next-day target returns
        self.return_scaler.fit(df[['Target_Return']])

        # Save scalers
        joblib.dump(self.feature_scaler, MODELS_DIR / 'feature_scaler.pkl')
        joblib.dump(self.return_scaler, MODELS_DIR / 'return_scaler.pkl')

        print("Scalers fitted and saved")
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        df = df.copy()

        all_features = self.feature_columns + self.sentiment_columns
        df[all_features] = self.feature_scaler.transform(df[all_features])

        # Scale next-day target return
        df['Target_Return_Scaled'] = self.return_scaler.transform(df[['Target_Return']].values)

        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        original_close_prices: np.ndarray = None
    ) -> tuple:
        """
        Create sliding-window sequences for next-day return prediction.

        Returns:
            Tuple of (X_price, X_sentiment, y_regression, dates, close_prices)
        """
        all_features = self.feature_columns + self.sentiment_columns

        X_all = df[all_features].values
        y_reg = df['Target_Return_Scaled'].values
        dates = df['date'].values

        if original_close_prices is not None:
            close_prices = original_close_prices
        else:
            close_prices = df['close'].values

        X_price_sequences = []
        X_sentiment_sequences = []
        y_regression = []
        sequence_dates = []
        sequence_close_prices = []

        n_price_features = len(self.feature_columns)
        n_sentiment_features = len(self.sentiment_columns)

        # Trim by PREDICTION_HORIZON (=1) since target is the next-day return.
        # Window includes row i (today's bar); target y_reg[i] is the return
        # from close[i] to close[i+1], so today's features are inputs to a
        # prediction about tomorrow.
        for i in range(self.sequence_length, len(df) - PREDICTION_HORIZON):
            X_price_sequences.append(X_all[i-self.sequence_length+1:i+1, :n_price_features])

            if n_sentiment_features > 0:
                X_sentiment_sequences.append(X_all[i-self.sequence_length+1:i+1, n_price_features:])

            y_regression.append(y_reg[i])
            sequence_dates.append(dates[i])
            sequence_close_prices.append(close_prices[i])

        X_price = np.array(X_price_sequences)
        X_sentiment = np.array(X_sentiment_sequences) if X_sentiment_sequences else None

        return (
            X_price,
            X_sentiment,
            np.array(y_regression),
            np.array(sequence_dates),
            np.array(sequence_close_prices)
        )

    def split_data(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray,
        y_reg: np.ndarray,
        dates: np.ndarray,
        close_prices: np.ndarray
    ) -> dict:
        """Chronological train/val/test split."""
        n = len(y_reg)
        train_end = int(n * TRAINING_CONFIG['train_split'])
        val_end = train_end + int(n * TRAINING_CONFIG['val_split'])

        def _slice(arr, start, end):
            if arr is None:
                return None
            return arr[start:end]

        splits = {
            'train': {
                'X_price': X_price[:train_end],
                'X_sentiment': _slice(X_sentiment, 0, train_end),
                'y_reg': y_reg[:train_end],
                'dates': dates[:train_end],
                'close_prices': close_prices[:train_end]
            },
            'val': {
                'X_price': X_price[train_end:val_end],
                'X_sentiment': _slice(X_sentiment, train_end, val_end),
                'y_reg': y_reg[train_end:val_end],
                'dates': dates[train_end:val_end],
                'close_prices': close_prices[train_end:val_end]
            },
            'test': {
                'X_price': X_price[val_end:],
                'X_sentiment': _slice(X_sentiment, val_end, None),
                'y_reg': y_reg[val_end:],
                'dates': dates[val_end:],
                'close_prices': close_prices[val_end:]
            }
        }

        print(f"Train: {len(splits['train']['y_reg'])} samples")
        print(f"Val: {len(splits['val']['y_reg'])} samples")
        print(f"Test: {len(splits['test']['y_reg'])} samples")

        return splits
    
    def prepare_data(
        self,
        stock_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        indicators_df: pd.DataFrame
    ) -> dict:
        """
        Full preprocessing pipeline.
        Scalers are fitted on TRAINING data only to prevent information leakage.
        
        Returns:
            Dict with train/val/test splits
        """
        print("Starting data preprocessing...")
        
        # Merge all data
        df = self.merge_data(stock_df, sentiment_df, indicators_df)
        
        # Select features
        self.select_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Save original close prices BEFORE scaling (for price reconstruction)
        original_close_prices = df['close'].values.copy()
        
        # --- FIX: Fit scalers on TRAINING portion only ---
        n = len(df)
        train_end_idx = int(n * TRAINING_CONFIG['train_split'])
        train_df = df.iloc[:train_end_idx]
        self.fit_scalers(train_df)
        
        # Transform ALL features using train-fitted scalers
        df = self.transform_features(df)
        
        # Create sequences, passing original close prices
        X_price, X_sentiment, y_reg, dates, close_prices = self.create_sequences(
            df, original_close_prices=original_close_prices
        )

        # Split data
        splits = self.split_data(X_price, X_sentiment, y_reg, dates, close_prices)

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'sentiment_columns': self.sentiment_columns,
            'sequence_length': self.sequence_length,
            'n_price_features': len(self.feature_columns),
            'n_sentiment_features': len(self.sentiment_columns),
        }
        joblib.dump(metadata, MODELS_DIR / 'preprocessing_metadata.pkl')
        
        print("Data preprocessing complete!")
        
        return splits


if __name__ == "__main__":
    from src.data.stock_data import fetch_stock_data
    from src.data.sentiment_data import fetch_sentiment_data
    from src.features.technical import calculate_all_indicators
    
    # Fetch all data
    stock_df = fetch_stock_data()
    sentiment_df = fetch_sentiment_data(stock_df)
    indicators_df = calculate_all_indicators(stock_df)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    
    print(f"\nPrice features shape: {splits['train']['X_price'].shape}")
    if splits['train']['X_sentiment'] is not None:
        print(f"Sentiment features shape: {splits['train']['X_sentiment'].shape}")
