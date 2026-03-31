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
    SEQUENCE_LENGTH, PREDICTION_HORIZON, PROCESSED_DATA_DIR,
    TRAINING_CONFIG, MODELS_DIR
)


class DataPreprocessor:
    """Preprocess and prepare data for training"""
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
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
        # Exclude non-feature columns
        exclude_cols = ['date', 'Target_Price', 'Target_Return', 'Target_Direction']
        
        # All numeric columns except excluded
        all_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        # Identify sentiment columns
        sentiment_cols = [col for col in all_features if 'sentiment' in col.lower()]
        
        # Other features (technical + price)
        other_cols = [col for col in all_features if col not in sentiment_cols]
        
        self.feature_columns = other_cols
        self.sentiment_columns = sentiment_cols
        
        return other_cols, sentiment_cols
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and infinities"""
        df = df.copy()
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        return df
    
    def fit_scalers(self, df: pd.DataFrame):
        """Fit scalers on training data"""
        if self.feature_columns is None:
            self.select_features(df)
        
        all_features = self.feature_columns + self.sentiment_columns
        
        # Fit feature scaler
        self.feature_scaler.fit(df[all_features])
        
        # Fit price scaler on close prices
        self.price_scaler.fit(df[['close']])
        
        # Save scalers
        joblib.dump(self.feature_scaler, MODELS_DIR / 'feature_scaler.pkl')
        joblib.dump(self.price_scaler, MODELS_DIR / 'price_scaler.pkl')
        
        print("Scalers fitted and saved")
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        df = df.copy()
        
        all_features = self.feature_columns + self.sentiment_columns
        df[all_features] = self.feature_scaler.transform(df[all_features])
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = 'Target_Price'
    ) -> tuple:
        """
        Create sequences for time-series modeling.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Target column name
        
        Returns:
            Tuple of (X_price, X_sentiment, y_regression, y_classification, dates)
        """
        all_features = self.feature_columns + self.sentiment_columns
        
        X_all = df[all_features].values
        y_reg = df['Target_Price'].values
        y_cls = df['Target_Direction'].values
        dates = df['date'].values
        
        X_price_sequences = []
        X_sentiment_sequences = []
        y_regression = []
        y_classification = []
        sequence_dates = []
        
        n_price_features = len(self.feature_columns)
        n_sentiment_features = len(self.sentiment_columns)
        
        for i in range(self.sequence_length, len(df) - PREDICTION_HORIZON):
            # Price/technical features sequence
            X_price_sequences.append(X_all[i-self.sequence_length:i, :n_price_features])
            
            # Sentiment features sequence
            if n_sentiment_features > 0:
                X_sentiment_sequences.append(X_all[i-self.sequence_length:i, n_price_features:])
            
            y_regression.append(y_reg[i])
            y_classification.append(y_cls[i])
            sequence_dates.append(dates[i])
        
        X_price = np.array(X_price_sequences)
        X_sentiment = np.array(X_sentiment_sequences) if X_sentiment_sequences else None
        
        return (
            X_price,
            X_sentiment,
            np.array(y_regression),
            np.array(y_classification),
            np.array(sequence_dates)
        )
    
    def split_data(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        dates: np.ndarray
    ) -> dict:
        """
        Split data into train/val/test sets (time-based).
        
        Returns:
            Dict with train/val/test splits
        """
        n = len(y_reg)
        train_end = int(n * TRAINING_CONFIG['train_split'])
        val_end = train_end + int(n * TRAINING_CONFIG['val_split'])
        
        splits = {
            'train': {
                'X_price': X_price[:train_end],
                'X_sentiment': X_sentiment[:train_end] if X_sentiment is not None else None,
                'y_reg': y_reg[:train_end],
                'y_cls': y_cls[:train_end],
                'dates': dates[:train_end]
            },
            'val': {
                'X_price': X_price[train_end:val_end],
                'X_sentiment': X_sentiment[train_end:val_end] if X_sentiment is not None else None,
                'y_reg': y_reg[train_end:val_end],
                'y_cls': y_cls[train_end:val_end],
                'dates': dates[train_end:val_end]
            },
            'test': {
                'X_price': X_price[val_end:],
                'X_sentiment': X_sentiment[val_end:] if X_sentiment is not None else None,
                'y_reg': y_reg[val_end:],
                'y_cls': y_cls[val_end:],
                'dates': dates[val_end:]
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
        
        # Fit scalers on all data (before splitting for consistent scaling)
        self.fit_scalers(df)
        
        # Transform features
        df = self.transform_features(df)
        
        # Create sequences
        X_price, X_sentiment, y_reg, y_cls, dates = self.create_sequences(df)
        
        # Split data
        splits = self.split_data(X_price, X_sentiment, y_reg, y_cls, dates)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'sentiment_columns': self.sentiment_columns,
            'sequence_length': self.sequence_length,
            'n_price_features': len(self.feature_columns),
            'n_sentiment_features': len(self.sentiment_columns)
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
