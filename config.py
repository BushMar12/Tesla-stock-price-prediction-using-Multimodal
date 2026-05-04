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
START_DATE = "2021-01-01"
END_DATE = "2026-04-21"  # None means today

# Feature settings
SEQUENCE_LENGTH = 60  # Number of days to look back
PREDICTION_HORIZON = 1  # Days ahead to predict (next-day only)
USE_MARKET_CONTEXT = False  # Current feature set excludes SPY/VIX market context
MARKET_CONTEXT_CACHE = True  # Cache SPY/VIX downloads under data/raw/

# Technical indicators to use
TECHNICAL_INDICATORS = [
    'SMA_20', 'SMA_50',
    'RSI_14',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
]

# Active price/technical input features.
# Sentiment features are selected separately when a sentiment mode is enabled.
PRICE_FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50',
    'RSI_14',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
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
}

# Training settings
RANDOM_SEED = 42
DETERMINISTIC_TRAINING = True

TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 50,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    # LR schedule (ReduceLROnPlateau on val MAE in dollar space)
    "lr_patience": 5,
    "lr_factor": 0.5,
}

# Checkpoint version tag — bump when state_dict shape changes so loaders fail loudly.
MODEL_VERSION = "next-day-only-v2-seq60"

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    "source": "alpha_vantage",  # synthetic, rss, or alpha_vantage
    "use_finbert": True,
    "finbert_model": "ProsusAI/finbert",
    "max_headlines_per_day": 10,
    "sentiment_lookback_days": 3,
    "use_real_data_fetch": False,
    "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "alpha_vantage_start_date": "2021-01-01",
    "alpha_vantage_chunk_days": 365,
    "alpha_vantage_request_sleep": 12,
    "alpha_vantage_limit": 1000,
    "alpha_vantage_use_cache": True,
}

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "Tesla Stock Predictor",
    "page_icon": None,
    "layout": "wide",
    "sentiment_use_real_data": False,
    "sentiment_source": "alpha_vantage",
}
