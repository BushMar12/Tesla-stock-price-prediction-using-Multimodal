"""
Configuration settings for Tesla Stock Price Prediction
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Stock settings
STOCK_SYMBOL = "TSLA"
START_DATE = "2010-06-29"
END_DATE = None  # None means today

# Feature settings
SEQUENCE_LENGTH = 60  # Number of days to look back
PREDICTION_HORIZON = 1  # Days ahead to predict (legacy, single-step)
PREDICTION_HORIZONS = [1, 3, 5, 7]  # Multi-day prediction horizons

# Technical indicators to use
TECHNICAL_INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'ATR_14', 'OBV', 'VWAP'
]

# Model hyperparameters
MODEL_CONFIG = {
    # Time-series encoder
    "ts_input_size": 64,  # Will be set dynamically based on features
    "ts_hidden_size": 128,
    "ts_num_layers": 2,
    "ts_dropout": 0.2,
    "ts_bidirectional": True,
    
    # Sentiment encoder
    "sentiment_embedding_dim": 768,  # FinBERT output dim
    "sentiment_hidden_dim": 256,
    
    # Fusion layer
    "fusion_hidden_dim": 256,
    "fusion_dropout": 0.3,
    
    # Output
    "num_classes": 2,  # Up, Down (binary classification)
}

# Training settings
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 200,
    "early_stopping_patience": 30,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "regression_weight": 0.5,  # Weight for regression loss
    "classification_weight": 0.5,  # Weight for classification loss
}

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    "use_finbert": True,
    "finbert_model": "ProsusAI/finbert",
    "max_headlines_per_day": 10,
    "sentiment_lookback_days": 3,
    "use_real_data_fetch": True,
}

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "Tesla Stock Predictor",
    "page_icon": "📈",
    "layout": "wide",
    "sentiment_use_real_data": False,
}
