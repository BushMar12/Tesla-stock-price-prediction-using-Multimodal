"""
Prediction script for Tesla Stock Price Prediction
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from datetime import datetime

from src.data.stock_data import fetch_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.utils.helpers import load_trained_model
from config import MODELS_DIR, SENTIMENT_CONFIG


def predict_next_day():
    """Make prediction for the next trading day"""
    print("=" * 60)
    print("Tesla Stock Price Prediction")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    try:
        model, metadata = load_trained_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first using: python train.py")
        return
    
    # Fetch latest data
    print("\nFetching latest data...")
    stock_df = fetch_stock_data(save=False)
    sentiment_df = fetch_sentiment_data(
        stock_df,
        use_real_data=SENTIMENT_CONFIG["use_real_data_fetch"],
        save=False,
    )
    indicators_df = calculate_all_indicators(stock_df, add_targets=False)
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = DataPreprocessor(sequence_length=metadata['sequence_length'])
    preprocessor.feature_columns = metadata['feature_columns']
    preprocessor.sentiment_columns = metadata['sentiment_columns']
    
    # Merge data
    df = preprocessor.merge_data(stock_df, sentiment_df, indicators_df)
    df = preprocessor.clean_data(df)
    
    # Load scaler and transform
    import joblib
    feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
    
    all_features = preprocessor.feature_columns + preprocessor.sentiment_columns
    df[all_features] = feature_scaler.transform(df[all_features])
    
    # Get last sequence
    n_price_features = len(preprocessor.feature_columns)
    sequence_length = preprocessor.sequence_length
    
    X_all = df[all_features].values
    X_price = X_all[-sequence_length:, :n_price_features]
    X_sentiment = X_all[-sequence_length:, n_price_features:]
    
    # Convert to tensors
    X_price = torch.FloatTensor(X_price).unsqueeze(0).to(device)
    X_sentiment = torch.FloatTensor(X_sentiment).unsqueeze(0).to(device)
    
    # Predict
    print("\nMaking prediction...")
    with torch.no_grad():
        outputs = model(X_price, X_sentiment)
        
        direction_probs = torch.softmax(outputs['classification'], dim=-1).cpu().numpy()[0]
        direction = np.argmax(direction_probs)
    
    # Print results (binary: class 0 = Down, class 1 = Up)
    next_label = "Up" if direction == 1 else "Down"

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nNext trading day direction: {next_label}")
    print("=" * 60)
    


if __name__ == "__main__":
    predict_next_day()
