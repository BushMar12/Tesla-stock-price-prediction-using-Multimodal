# Tesla Stock Price Prediction Using Multimodal Deep Learning

A comprehensive stock price prediction system that combines time-series analysis with sentiment data using a **multimodal fusion** model (PyTorch), plus **standalone LSTM, GRU, and XGBoost** baselines for comparison.

![Training History](training_history.png)

## Overview

This project predicts Tesla (TSLA) stock prices using a **returns-based approach**:
1. Predicts daily percentage returns (more stationary than raw prices)
2. Reconstructs actual prices: `predicted_price = today_close × (1 + predicted_return)`

### Models Implemented
- **Multimodal fusion** (`train.py`) — time-series encoder + sentiment encoder + optional cross-modal attention (`src/models/fusion.py`)
- **LSTM / GRU / XGBoost** (`model_comparison.py`) — separate bidirectional recurrent regressors and **XGBoost** (`xgboost.XGBRegressor`) on flattened sequences

### Features Used
- **Price Data**: OHLCV (Open, High, Low, Close, Volume)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP
- **Sentiment Data**: News sentiment scores (synthetic or real)
- **Derived Features**: Daily returns, price momentum, volatility

## Project Structure

```
├── app/
│   └── streamlit_app.py      # Interactive web dashboard
├── data/
│   ├── raw/                  # Raw stock & sentiment data
│   └── processed/            # Preprocessed sequences
├── models/                   # Saved model checkpoints
├── notebooks/
│   └── exploration.ipynb     # Data exploration & analysis
├── src/
│   ├── data/
│   │   ├── stock_data.py     # Yahoo Finance data fetcher
│   │   ├── sentiment_data.py # Sentiment data generator
│   │   └── preprocessing.py  # Data preprocessing pipeline
│   ├── features/
│   │   └── technical.py      # Technical indicator calculations
│   ├── models/
│   │   ├── fusion.py         # Multimodal fusion + attention
│   │   ├── time_series.py    # Time-series encoder
│   │   ├── text_encoder.py   # Sentiment encoders
│   │   ├── regression_models.py  # Standalone LSTM, GRU, XGBoost
│   │   └── trainer.py        # Training loop for fusion model
│   └── utils/                # Helper functions
├── config.py                 # Configuration settings
├── train.py                  # Train multimodal fusion model
├── model_comparison.py       # Train & compare LSTM, GRU, XGBoost
├── predict.py                # Prediction script (fusion checkpoint)
└── requirements.txt          # Dependencies
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Stock price prediction using multimodal"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Multimodal fusion model (saved under models/, e.g. best_model.pt)
python train.py

# LSTM vs GRU vs XGBoost baselines (saves models + model_comparison.csv)
python model_comparison.py
```

### 3. Run Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app uses `STREAMLIT_CONFIG["sentiment_use_real_data"]` (default **false**) so the dashboard loads quickly without RSS/network sentiment fetching. Training scripts use `SENTIMENT_CONFIG["use_real_data_fetch"]` instead (default **true**).

##  Configuration

Edit `config.py` to customize:

```python
# Data settings
STOCK_SYMBOL = "TSLA"
START_DATE = "2010-06-29"
SEQUENCE_LENGTH = 60  # Look-back window (days)

# Align sentiment source across train.py, model_comparison.py, predict.py
SENTIMENT_CONFIG = {
    ...
    "use_real_data_fetch": True,   # False = synthetic only (faster, offline-friendly)
}

# Dashboard: synthetic sentiment by default
STREAMLIT_CONFIG = {
    ...
    "sentiment_use_real_data": False,
}

# Model architecture
MODEL_CONFIG = {
    "ts_hidden_size": 128,
    "ts_num_layers": 2,
    "ts_dropout": 0.2,
    "ts_bidirectional": True,
}

# Training settings
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 200,
    "early_stopping_patience": 30,
}
```

## GPU Support

The project supports:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon M1/M2/M3)
- **CPU** (fallback)

Device selection is automatic. To verify:
```python
import torch
print(torch.backends.mps.is_available())  # For Mac
print(torch.cuda.is_available())          # For NVIDIA
```

## Model Architecture

### Multimodal fusion (`train.py`)
Time-series branch (e.g. LSTM-based encoder in `time_series.py`), sentiment branch (`text_encoder.py`), optional cross-modal attention, then fusion MLPs with regression and direction-classification heads. See `src/models/fusion.py`.

### Standalone LSTM / GRU (`model_comparison.py`)
```
Input (60 timesteps × N features)
    ↓
Bidirectional LSTM/GRU (128 hidden, 2 layers)
    ↓
Dropout (0.2)
    ↓
Dense (1 output - scaled return)
```

### XGBoost (comparison script)
- **Library**: `xgboost.XGBRegressor`
- Default hyperparameters in `XGBoostRegressor`: 400 estimators, max depth 4, learning rate 0.05, subsample 0.8

## Metrics

- **RMSE** - Root Mean Square Error (in dollars)
- **MAE** - Mean Absolute Error (in dollars)
- **MAPE** - Mean Absolute Percentage Error
- **Direction Accuracy** - Correctly predicted Up/Down movements

##  Tasks

The system performs:
1. **Regression**: Predict next-day closing price
2. **Classification**: Predict price direction (Up/Down)

##  Usage Examples

### Load comparison models after `model_comparison.py`

```python
from pathlib import Path
from src.models.regression_models import MultiModelRegressor
from config import MODELS_DIR

regressor = MultiModelRegressor(input_size=64)  # set to saved input_size
regressor.load_models(MODELS_DIR)
# Use regressor.models['LSTM'], etc., with your preprocessed tensors
```

### Make Predictions (CLI)

```bash
python predict.py
```

Requires a trained fusion checkpoint from `train.py` (see `src/utils/helpers.py`).

### Streamlit Dashboard Features
- Real-time stock data visualization
- Model predictions with confidence intervals
- Technical indicator charts
- Model performance comparison
- Historical accuracy analysis

##  Technical Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch, Transformers |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy, yfinance |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| NLP | NLTK, VADER, FinBERT |

## References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber
- [GRU](https://arxiv.org/abs/1406.1078) - Cho et al.
- [XGBoost](https://arxiv.org/abs/1603.02754) - Chen & Guestrin
- [FinBERT](https://arxiv.org/abs/1908.10063) - Financial Sentiment Analysis

##  License

This project is for educational purposes (UTS 49275 Neural Networks and Fuzzy Logic).

