# Tesla Stock Price Prediction Using Multimodal Deep Learning

This project predicts Tesla (`TSLA`) stock movement using a returns-based multimodal learning pipeline. It combines OHLCV market data, technical indicators, market context, and sentiment features, then compares a two-stream multimodal fusion model against standalone LSTM, GRU, Transformer, and XGBoost baselines.

The project is for educational/research use only. It is not a trading system.

## Overview

The core prediction target is next-day return:

```text
predicted_price = today_close * (1 + predicted_return)
```

Predicting returns is generally more stable than predicting raw prices directly. The current project scope is next-day prediction only.

## Current Project Settings

Key defaults in `config.py`:

```python
STOCK_SYMBOL = "TSLA"
START_DATE = "2021-01-06"
END_DATE = "2026-04-29"
SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 1
RANDOM_SEED = 42
DETERMINISTIC_TRAINING = True

TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 50,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
}
```

The multimodal trainer optimizes only the next-day return regression objective.

## Features

- **OHLCV data** from Yahoo Finance
- **Returns and price-derived features**: returns, log returns, price range, price range percentage
- **Technical indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, stochastic oscillator, momentum, volatility, candlestick features, gap features
- **Market context**: SPY close/return, TSLA-vs-SPY alpha, VIX
- **Calendar features**: day of week, month sine/cosine, month end, quarter end
- **Sentiment features**:
  - synthetic sentiment proxy
  - RSS headline sentiment
  - Alpha Vantage TSLA news sentiment

## Model Architectures

### Multimodal Fusion Model

Implemented in `src/models/fusion.py`.

The multimodal model uses two temporal streams:

```text
Price/technical stream -> TimeSeriesEncoder
Sentiment stream       -> TemporalSentimentEncoder
```

If enabled, cross-modal attention lets the time-series sequence attend to the sentiment representation. The fused representation is passed into one output head:

- 1-day return regression head

### Standalone Baselines

Implemented in `src/models/regression_models.py`.

The standalone models use the same preprocessing pipeline but concatenate price/technical and sentiment features into a single input tensor. They are therefore **single-stream baselines**, not OHLCV-only baselines.

Models:

- **LSTM**: input projection, 2-layer bidirectional LSTM, dense regression head
- **GRU**: input projection, 2-layer bidirectional GRU, dense regression head
- **Transformer**: input projection, sinusoidal positional encoding, Transformer encoder, dense regression head
- **XGBoost**: flattened 60-day sequence features passed into `xgboost.XGBRegressor`

## Training Modes

`train.py` supports three multimodal training modes:

```bash
python train.py --mode current
```

Uses sentiment data and cross-attention.

```bash
python train.py --mode no-sentiment
```

Disables sentiment. The model trains only on price, technical, market, and calendar features.

```bash
python train.py --mode sentiment-no-cross-attention
```

Uses sentiment but disables cross-attention. Price and sentiment representations are fused by concatenation.

## Sentiment Sources

For modes that use sentiment, choose one source:

```bash
python train.py --mode current --sentiment-source synthetic
python train.py --mode current --sentiment-source rss
python train.py --mode current --sentiment-source alpha_vantage
```

### Synthetic Sentiment

The default `synthetic` source creates a reproducible sentiment proxy from lagged market data and noise. It is useful for validating the multimodal pipeline, but it is not real news or social media sentiment.

### RSS Sentiment

The `rss` source fetches current RSS headlines and scores them with VADER. This is limited because RSS does not provide reliable historical sentiment for the full training period.

### Alpha Vantage TSLA News Sentiment

The `alpha_vantage` source uses Alpha Vantage `NEWS_SENTIMENT` data for TSLA.

Set your API key first:

```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

Then fetch/cache sentiment separately:

```bash
python fetch_alpha_vantage_sentiment.py
```

This writes:

```text
data/raw/alpha_vantage_tsla_news_raw.csv
data/raw/alpha_vantage_tsla_sentiment_data.csv
data/raw/sentiment_data.csv
```

Then train using the cached Alpha Vantage sentiment:

```bash
python train.py --mode current --sentiment-source alpha_vantage
```

Notes:

- Alpha Vantage free-tier limits can stop fetching mid-run.
- The fetch script saves partial raw results after each successful request.
- If interrupted by rate limits, rerun after the daily quota resets; it will resume from cached progress.
- Do not commit or share API keys.

## Standalone Model Comparison

Train and evaluate the standalone models:

```bash
python model_comparison.py
```

This trains:

```text
LSTM
GRU
Transformer
XGBoost
```

and saves:

```text
models/model_comparison.csv
models/lstm_regressor.pt
models/gru_regressor.pt
models/transformer_regressor.pt
models/xgboost_regressor.pkl
models/multi_model_metadata.pkl
```

The current comparison is:

```text
single-stream models using engineered features
vs
two-stream multimodal fusion model using separated price and sentiment streams
```

It is not an OHLCV-only baseline comparison.

## Evaluation Metrics

Primary metric:

- **MAE**: mean absolute error on reconstructed next-day price

Secondary metrics:

- **RMSE**: root mean squared dollar error
- **MAPE**: mean absolute percentage error

The model is evaluated as a next-day price predictor.

## Streamlit Dashboard

Use Python 3.11 or 3.12 for the dashboard. Python 3.14 is not recommended for this stack.

Recommended setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-stable.txt
```

Run the dashboard safely:

```bash
zsh run_streamlit_safe.sh
```

The safe launcher disables Streamlit file watching, which avoids native crashes in some macOS/OneDrive/PyTorch setups.

Important: Streamlit skips live loading of the saved XGBoost pickle because incompatible XGBoost native versions can segfault during unpickling. It can still display XGBoost metrics from `models/model_comparison.csv`.

## Project Structure

```text
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   │   ├── stock_data.py
│   │   ├── sentiment_data.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── technical.py
│   ├── models/
│   │   ├── fusion.py
│   │   ├── time_series.py
│   │   ├── text_encoder.py
│   │   ├── regression_models.py
│   │   └── trainer.py
│   └── utils/
├── config.py
├── train.py
├── model_comparison.py
├── fetch_alpha_vantage_sentiment.py
├── predict.py
├── requirements.txt
├── requirements-stable.txt
└── run_streamlit_safe.sh
```

## Common Commands

Check training options:

```bash
python train.py --help
```

Train current multimodal model with synthetic sentiment:

```bash
python train.py --mode current --sentiment-source synthetic
```

Train without sentiment:

```bash
python train.py --mode no-sentiment
```

Fetch Alpha Vantage sentiment:

```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
python fetch_alpha_vantage_sentiment.py
```

Train with Alpha Vantage sentiment:

```bash
python train.py --mode current --sentiment-source alpha_vantage
```

Train standalone baselines:

```bash
python model_comparison.py
```

Run prediction CLI:

```bash
python predict.py
```

Run dashboard:

```bash
zsh run_streamlit_safe.sh
```

## Scope and Limitations

- This project is educational and should not be used for real trading decisions.
- TSLA returns are noisy, non-stationary, and event-driven.
- Synthetic sentiment is not real sentiment.
- RSS sentiment is not a true historical sentiment source.
- Alpha Vantage free-tier API limits may require multiple days to fetch a full historical cache.
- A fixed chronological train/validation/test split is useful, but walk-forward validation would be stronger.
- The current implementation is intentionally limited to next-day price prediction.

## References

- Hochreiter and Schmidhuber, Long Short-Term Memory, 1997
- Cho et al., Learning Phrase Representations using RNN Encoder-Decoder, 2014
- Vaswani et al., Attention Is All You Need, 2017
- Chen and Guestrin, XGBoost: A Scalable Tree Boosting System, 2016
- Araci, FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, 2019

## License

This project is for educational purposes for UTS 49275 Neural Networks and Fuzzy Logic.
