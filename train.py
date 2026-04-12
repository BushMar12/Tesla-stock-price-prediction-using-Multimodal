"""
Main training script for Tesla Stock Price Prediction
Uses returns-based prediction for better accuracy
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.stock_data import fetch_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.models.trainer import train_model
from config import SENTIMENT_CONFIG


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Tesla Stock Price Prediction - Training Pipeline")
    print("Predicting Returns → Reconstructing Prices")
    print("=" * 60)
    
    # Step 1: Fetch stock data
    print("\n Step 1: Fetching stock data...")
    stock_df = fetch_stock_data(start_date="2010-06-29")
    
    # Step 2: Fetch sentiment data
    print("\n Step 2: Fetching sentiment data...")
    sentiment_df = fetch_sentiment_data(
        stock_df, use_real_data=SENTIMENT_CONFIG["use_real_data_fetch"]
    )
    
    # Step 3: Calculate technical indicators
    print("\n Step 3: Calculating technical indicators...")
    indicators_df = calculate_all_indicators(stock_df)
    
    # Step 4: Preprocess data
    print("\n Step 4: Preprocessing data...")
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    
    # Step 5: Train model (pass return_scaler for proper evaluation)
    print("\n Step 5: Training model...")
    model, trainer, history = train_model(splits, return_scaler=preprocessor.return_scaler)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print("\nTo run the Streamlit app:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
