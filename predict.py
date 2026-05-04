"""
Inference CLI for Tesla next-day price prediction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import joblib
from datetime import datetime

from src.data.stock_data import fetch_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.utils.helpers import load_trained_model
from config import MODELS_DIR, SENTIMENT_CONFIG


def predict_next_day():
    print("=" * 60)
    print("Tesla Stock Price Prediction (next-day)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

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

    print("\nFetching latest data...")
    stock_df = fetch_stock_data(save=False)
    sentiment_df = fetch_sentiment_data(
        stock_df,
        use_real_data=SENTIMENT_CONFIG.get("source") != "synthetic",
        save=False,
        source=SENTIMENT_CONFIG.get("source", "synthetic"),
    )
    indicators_df = calculate_all_indicators(stock_df, add_targets=False)

    print("\nPreprocessing...")
    preprocessor = DataPreprocessor(sequence_length=metadata['sequence_length'])
    preprocessor.feature_columns = metadata['feature_columns']
    preprocessor.sentiment_columns = metadata['sentiment_columns']

    df = preprocessor.merge_data(stock_df, sentiment_df, indicators_df)
    df = preprocessor.clean_data(df)

    feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
    return_scaler = joblib.load(MODELS_DIR / 'return_scaler.pkl')

    all_features = preprocessor.feature_columns + preprocessor.sentiment_columns
    df[all_features] = feature_scaler.transform(df[all_features])

    n_price_features = len(preprocessor.feature_columns)
    sequence_length = preprocessor.sequence_length

    X_all = df[all_features].values
    X_price = X_all[-sequence_length:, :n_price_features]
    X_sentiment = X_all[-sequence_length:, n_price_features:]

    X_price = torch.FloatTensor(X_price).unsqueeze(0).to(device)
    X_sentiment = torch.FloatTensor(X_sentiment).unsqueeze(0).to(device)

    current_price = stock_df['close'].iloc[-1]

    print("\nMaking prediction...")
    with torch.no_grad():
        outputs = model(X_price, X_sentiment)
        pred_return_scaled = outputs['regression'].cpu().numpy()[0]
        pred_return = return_scaler.inverse_transform([[pred_return_scaled]])[0, 0]
        pred_price = current_price * (1 + pred_return)

    change_pct = pred_return * 100
    direction = "UP" if change_pct > 0 else "DOWN" if change_pct < 0 else "FLAT"

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"  Current Price:   ${current_price:.2f}")
    print(f"  Next-day Price:  ${pred_price:.2f}  ({direction} {change_pct:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    predict_next_day()
