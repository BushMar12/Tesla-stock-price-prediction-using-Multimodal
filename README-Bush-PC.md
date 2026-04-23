# Tesla Stock Price Prediction Using Multimodal Deep Learning

This project predicts Tesla (TSLA) stock movement by combining price time-series data, technical indicators, market context, and sentiment features. The main model is a PyTorch multimodal fusion network, with standalone LSTM, GRU, Transformer, and XGBoost regressors available for comparison. The dashboard also includes SHAP-based feature explainability for next-day return.

The project is built for UTS 49275 Neural Networks and Fuzzy Logic.

![Training History](training_history.png)

## Overview

The prediction target is future return rather than raw closing price. The model predicts a percentage return, then reconstructs the future price from the latest close:

```text
predicted_price = current_close * (1 + predicted_return)
```

This is a more stable formulation than directly predicting prices because returns are usually closer to stationary than raw price levels.

The current configuration supports multi-horizon forecasting for 1, 3, 5, and 7 trading days.

## Models

### Multimodal fusion model

The main model is trained by `train.py` and implemented in `src/models/fusion.py`.

It contains:

- a time-series encoder for OHLCV, technical, market, and calendar features
- a temporal sentiment encoder for sentiment sequences
- optional cross-modal attention between price sequences and sentiment
- a fusion MLP
- output heads for single-day return regression, multi-day return regression, and up/down direction classification

### Baseline models

The comparison script, `model_comparison.py`, trains standalone regressors from `src/models/regression_models.py`:

- LSTM
- GRU
- Transformer
- XGBoost

Saved comparison results currently exist in `models/model_comparison.csv`. The saved CSV currently lists LSTM, GRU, and XGBoost results.

## Features Used

The feature pipeline combines:

- OHLCV price data from Yahoo Finance
- daily returns, log returns, and price range features
- technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, stochastic oscillator, momentum, volatility, and candlestick features
- SPY and VIX market context features
- calendar features such as day of week, month seasonality, month end, and quarter end
- sentiment features from RSS/news headlines or synthetic lagged sentiment

## Project Structure

```text
.
|-- app/
|   `-- streamlit_app.py              # Interactive Streamlit dashboard
|-- data/
|   |-- raw/                          # Saved stock and sentiment data
|   `-- processed/                    # Reserved for processed data artifacts
|-- models/                           # Trained checkpoints, scalers, metadata, and baseline models
|-- notebooks/
|   `-- exploration.ipynb             # Exploratory analysis notebook
|-- src/
|   |-- data/
|   |   |-- stock_data.py             # Yahoo Finance stock fetcher
|   |   |-- sentiment_data.py         # RSS/VADER sentiment and synthetic sentiment
|   |   `-- preprocessing.py          # Merge, scale, sequence, and split pipeline
|   |-- features/
|   |   `-- technical.py              # Technical indicators and target creation
|   |-- models/
|   |   |-- fusion.py                 # Multimodal fusion model
|   |   |-- time_series.py            # LSTM/GRU sequence encoders
|   |   |-- text_encoder.py           # Sentiment encoders
|   |   |-- regression_models.py      # LSTM, GRU, Transformer, XGBoost baselines
|   |   `-- trainer.py                # Training loop, loss, early stopping
|   `-- utils/
|       `-- helpers.py                # Model loading, metrics, formatting helpers
|-- ARCHITECTURE.md                   # Report-ready architecture explanation
|-- CODE_REVIEW.md                    # Code review notes and improvement plan
|-- config.py                         # Project configuration
|-- train.py                          # Train the multimodal fusion model
|-- model_comparison.py               # Train and compare baseline models
|-- predict.py                        # Run CLI prediction using the fusion checkpoint
|-- diagram.mmd                       # Mermaid architecture diagram source
|-- pipeline_architecture.png         # Rendered architecture diagram
|-- requirements.txt                  # Python dependencies
`-- Tesla_Stock_Prediction_Report.pdf # Project report artifact
```

## Quick Start

### 1. Create an environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the fusion model

```bash
python train.py
```

This saves checkpoints and preprocessing artifacts under `models/`, including:

- `best_model.pt`
- `final_model.pt`
- `feature_scaler.pkl`
- `return_scaler.pkl`
- `multi_return_scalers.pkl`
- `preprocessing_metadata.pkl`

### 4. Train baseline models

```bash
python model_comparison.py
```

This trains the standalone comparison models and writes outputs such as:

- `lstm_regressor.pt`
- `gru_regressor.pt`
- `xgboost_regressor.pkl`
- `multi_model_metadata.pkl`
- `model_comparison.csv`

### 5. Run prediction from the command line

```bash
python predict.py
```

This loads the trained fusion checkpoint, fetches current data, prepares the latest 60-day sequence, and prints 1, 3, 5, and 7 day forecasts.

### 6. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes price charts, technical indicators, sentiment visualizations, one-day prediction, multi-day forecast, model comparison, and SHAP explainability.

## Configuration

Edit `config.py` to change the stock symbol, date range, sequence length, prediction horizons, training settings, and sentiment behavior.

Important settings:

```python
STOCK_SYMBOL = "TSLA"
START_DATE = "2021-01-01"
SEQUENCE_LENGTH = 60
PREDICTION_HORIZONS = [1, 3, 5, 7]

SENTIMENT_CONFIG = {
    "use_finbert": True,
    "finbert_model": "ProsusAI/finbert",
    "max_headlines_per_day": 10,
    "sentiment_lookback_days": 3,
    "use_real_data_fetch": True,
}

STREAMLIT_CONFIG = {
    "page_title": "Tesla Stock Predictor",
    "layout": "wide",
    "sentiment_use_real_data": False,
}
```

Note: `train.py` and `model_comparison.py` currently call `fetch_stock_data(start_date="2010-06-29")` directly, so they override the `START_DATE` value in `config.py` unless those scripts are changed.

## Training Details

The preprocessing pipeline:

1. Merges stock, technical indicator, market context, calendar, and sentiment data by date.
2. Cleans missing and infinite values.
3. Fits feature and target scalers on the training portion only.
4. Builds 60-day sliding-window sequences.
5. Splits data chronologically into train, validation, and test sets.

The fusion model uses:

- Smooth L1 loss for return regression
- cross-entropy loss for direction classification
- an additional Smooth L1 loss for multi-day return outputs
- AdamW optimizer
- OneCycleLR scheduler
- gradient clipping
- early stopping

## Metrics

The project reports:

- RMSE in reconstructed dollar-price space
- MAE in reconstructed dollar-price space
- MAPE
- direction accuracy for up/down movement

## SHAP Explainability

The Streamlit dashboard now uses SHAP values for feature explainability. In the `SHAP Explainability` tab, an XGBoost surrogate model is trained on the engineered tabular features to explain next-day return. The tab shows:

- top features by mean absolute SHAP value
- a table of positive or negative average SHAP contribution
- a dependence view for selected important features

This method is more informative than simple correlation because it captures non-linear feature effects and interactions learned by the surrogate model.

Current saved baseline comparison:

| Model | RMSE | MAE | MAPE | Direction Accuracy |
| --- | ---: | ---: | ---: | ---: |
| LSTM | $13.51 | $10.51 | 3.08% | 49.9% |
| GRU | $14.56 | $11.06 | 3.25% | 49.1% |
| XGBoost | $14.71 | $11.51 | 3.31% | 49.1% |

## Notes For Evaluation

This project is educational and should not be treated as financial advice. Stock prices are noisy, non-stationary, and strongly affected by external events. The model is useful for comparing neural architectures and multimodal feature design, but real trading use would require stronger validation, walk-forward backtesting, transaction cost modeling, and risk controls.

See `ARCHITECTURE.md` for a report-ready explanation and `CODE_REVIEW.md` for review findings and improvement priorities.

## References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter and Schmidhuber
- [GRU](https://arxiv.org/abs/1406.1078) - Cho et al.
- [XGBoost](https://arxiv.org/abs/1603.02754) - Chen and Guestrin
- [FinBERT](https://arxiv.org/abs/1908.10063) - Financial Sentiment Analysis
