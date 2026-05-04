"""
Main training script for Tesla Stock Price Prediction
Uses returns-based prediction for better accuracy
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.stock_data import fetch_stock_data, load_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.models.trainer import train_model
from config import DETERMINISTIC_TRAINING, RANDOM_SEED, SENTIMENT_CONFIG, START_DATE, END_DATE


TRAINING_MODES = {
    "current": {
        "description": "Use sentiment data and cross-attention.",
        "use_sentiment": True,
        "use_cross_attention": True,
    },
    "no-sentiment": {
        "description": "Train only on price, technical, market, and calendar features.",
        "use_sentiment": False,
        "use_cross_attention": False,
    },
    "sentiment-no-cross-attention": {
        "description": "Use sentiment data but disable cross-attention.",
        "use_sentiment": True,
        "use_cross_attention": False,
    },
}


def set_global_seed(seed: int = RANDOM_SEED, deterministic: bool = DETERMINISTIC_TRAINING):
    """Seed Python, NumPy, and PyTorch for repeatable training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the Tesla stock prediction fusion model."
    )
    parser.add_argument(
        "--mode",
        choices=TRAINING_MODES.keys(),
        default="current",
        help=(
            "Training mode: current = sentiment + cross-attention; "
            "no-sentiment = no sentiment branch; "
            "sentiment-no-cross-attention = sentiment branch without cross-attention."
        ),
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["synthetic", "rss", "alpha_vantage"],
        default=SENTIMENT_CONFIG.get("source", "alpha_vantage"),
        help=(
            "Sentiment source for modes that use sentiment. "
            "alpha_vantage uses the cached daily file when available; "
            "otherwise it requires ALPHA_VANTAGE_API_KEY."
        ),
    )
    parser.add_argument(
        "--refresh-stock-data",
        action="store_true",
        help=(
            "Fetch stock data from Yahoo Finance and overwrite the cached CSV. "
            "By default, training uses the cached stock CSV for repeatable runs."
        ),
    )
    return parser.parse_args()


def main():
    """Main training pipeline"""
    set_global_seed(RANDOM_SEED)
    args = parse_args()
    mode_config = TRAINING_MODES[args.mode]

    print("=" * 60)
    print("Tesla Stock Price Prediction - Training Pipeline")
    print("Predicting Returns -> Reconstructing Prices")
    print(f"Training mode: {args.mode}")
    print(f"Mode details: {mode_config['description']}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Deterministic training: {DETERMINISTIC_TRAINING}")
    if mode_config["use_sentiment"]:
        print(f"Sentiment source: {args.sentiment_source}")
    print("=" * 60)
    
    # Step 1: Load stock data. Cached data is the default for reproducibility.
    if args.refresh_stock_data:
        print("\n Step 1: Fetching stock data...")
        stock_df = fetch_stock_data(start_date=START_DATE, end_date=END_DATE)
    else:
        print("\n Step 1: Loading cached stock data...")
        stock_df = load_stock_data()
    
    # Step 2: Fetch or disable sentiment data
    if mode_config["use_sentiment"]:
        print("\n Step 2: Fetching sentiment data...")
        sentiment_df = fetch_sentiment_data(
            stock_df,
            use_real_data=args.sentiment_source != "synthetic",
            source=args.sentiment_source,
        )
    else:
        print("\n Step 2: Sentiment disabled for this training run.")
        sentiment_df = stock_df[['date']].copy()
    
    # Step 3: Calculate technical indicators
    print("\n Step 3: Calculating technical indicators...")
    indicators_df = calculate_all_indicators(stock_df)
    
    # Step 4: Preprocess data
    print("\n Step 4: Preprocessing data...")
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    
    # Step 5: Train model (pass return_scaler for proper evaluation)
    print("\n Step 5: Training model...")
    model, trainer, history = train_model(
        splits,
        return_scaler=preprocessor.return_scaler,
        use_cross_attention=mode_config["use_cross_attention"],
        training_mode=args.mode,
        use_sentiment=mode_config["use_sentiment"],
        random_seed=RANDOM_SEED,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nTo run the Streamlit app:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
