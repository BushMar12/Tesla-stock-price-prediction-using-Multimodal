"""
Utility helper functions
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, MODEL_VERSION


def load_trained_model(model_path: str = None):
    """Load a trained next-day fusion model from checkpoint."""
    from src.models.fusion import create_model

    if model_path is None:
        model_path = MODELS_DIR / 'best_model.pt'

    metadata_path = MODELS_DIR / 'preprocessing_metadata.pkl'
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
        ts_input_size = metadata['n_price_features']
        sentiment_input_size = metadata['n_sentiment_features']
        use_cross_attention = metadata.get('use_cross_attention', True)
    else:
        raise FileNotFoundError("Preprocessing metadata not found. Please train the model first.")

    model = create_model(
        ts_input_size=ts_input_size,
        sentiment_input_size=sentiment_input_size,
        use_cross_attention=use_cross_attention,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    version = checkpoint.get('model_version')
    if version != MODEL_VERSION:
        raise RuntimeError(
            f"Checkpoint at {model_path} has model_version={version!r}, "
            f"expected {MODEL_VERSION!r}. Retrain required."
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, metadata


def inverse_scale_price(scaled_price: np.ndarray) -> np.ndarray:
    """Convert scaled price back to original scale"""
    scaler_path = MODELS_DIR / 'price_scaler.pkl'
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        return scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()
    
    return scaled_price


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:+.2f}%"


def get_direction_label(direction: int) -> str:
    """Convert direction code to label."""
    labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}
    return labels.get(direction, 'Unknown')


def get_direction_color(direction: int) -> str:
    """Get color for direction."""
    colors = {0: 'red', 1: 'gray', 2: 'green'}
    return colors.get(direction, 'gray')
