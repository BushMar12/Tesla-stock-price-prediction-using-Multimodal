"""
Training script for multi-model regression comparison
Trains LSTM, GRU, and XGBoost models and compares their performance
Uses returns-based prediction for better accuracy
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.stock_data import fetch_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.models.regression_models import MultiModelRegressor
from config import MODELS_DIR


def main():
    """Train and compare multiple regression models"""
    print("=" * 60)
    print("Multi-Model Regression Training")
    print("LSTM vs GRU vs XGBoost Comparison")
    print("Predicting Returns → Reconstructing Prices")
    print("=" * 60)
    
    # Step 1: Fetch and preprocess data
    print("\n Step 1: Fetching and preprocessing data...")
    stock_df = fetch_stock_data(start_date="2010-06-29")
    sentiment_df = fetch_sentiment_data(stock_df, use_real_data=False)
    indicators_df = calculate_all_indicators(stock_df)
    
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    
    # Combine price and sentiment features
    X_train = splits['train']['X_price']
    X_val = splits['val']['X_price']
    X_test = splits['test']['X_price']
    
    # If sentiment features exist, concatenate them
    if splits['train']['X_sentiment'] is not None:
        X_train = np.concatenate([X_train, splits['train']['X_sentiment']], axis=2)
        X_val = np.concatenate([X_val, splits['val']['X_sentiment']], axis=2)
        X_test = np.concatenate([X_test, splits['test']['X_sentiment']], axis=2)
    
    # Target: scaled returns
    y_train = splits['train']['y_reg']
    y_val = splits['val']['y_reg']
    y_test = splits['test']['y_reg']
    
    # Close prices for reconstructing predicted prices
    close_prices_test = splits['test']['close_prices']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Step 2: Initialize and train models
    print("\n Step 2: Training models...")
    input_size = X_train.shape[2]
    multi_model = MultiModelRegressor(input_size=input_size)
    
    multi_model.train_all(X_train, y_train, X_val, y_val, epochs=100)
    
    # Step 3: Evaluate models (pass return_scaler and close_prices for price reconstruction)
    print("\n Step 3: Evaluating models...")
    results = multi_model.evaluate_all(
        X_test, y_test, 
        return_scaler=preprocessor.return_scaler,
        close_prices=close_prices_test
    )
    
    # Step 4: Display comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': f"${metrics['RMSE']:.2f}",
            'MAE': f"${metrics['MAE']:.2f}",
            'MAPE': f"{metrics['MAPE']:.2f}%",
            'Dir. Accuracy': f"{metrics['Directional_Accuracy']:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    print(f"\n Best Model (lowest RMSE): {best_model[0]}")
    
    # Step 5: Save models
    print("\n Step 5: Saving models...")
    multi_model.save_models(MODELS_DIR)
    
    # Save comparison results
    comparison_df.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print("\nTo run the Streamlit app with model comparison:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
