# Tesla Stock Price Prediction Using Multimodal Learning

This project predicts Tesla (`TSLA`) next-day stock movement with a returns-based
forecasting pipeline. It combines OHLCV market data, technical indicators, and
daily sentiment features, then compares a two-stream multimodal fusion model
against standalone LSTM, GRU, Transformer, and XGBoost regressors.

This is an educational/research project for UTS 49275 Neural Networks and Fuzzy
Logic. It is not a trading system and should not be used for financial advice.

## Project Summary

The prediction target is next-day return:

```text
predicted_next_price = current_close * (1 + predicted_next_day_return)
```

The model is evaluated in reconstructed dollar-price space using RMSE, MAE,
MAPE, and MSE.

Current defaults in `config.py`:

```python
STOCK_SYMBOL = "TSLA"
START_DATE = "2021-01-01"
END_DATE = "2026-04-21"
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

## Data and Features

Data sources:

- TSLA OHLCV data from Yahoo Finance, cached at `data/raw/TSLA_historical.csv`
- Optional Alpha Vantage TSLA news sentiment, cached under `data/raw/`
- Optional RSS or synthetic sentiment sources for experimentation

Active price/technical feature set:

- `open`, `high`, `low`, `close`, `volume`
- `SMA_20`, `SMA_50`
- `RSI_14`
- `BB_Upper`, `BB_Middle`, `BB_Lower`

Sentiment features are selected separately by `DataPreprocessor`. Columns whose
name contains `sentiment`, plus `news_count`, are routed into the sentiment
stream when sentiment is enabled.

The preprocessing pipeline:

1. Merges stock, technical indicator, and sentiment data by date.
2. Cleans missing and infinite values.
3. Fits feature and return scalers on the training portion only.
4. Creates 60-day chronological sliding windows.
5. Splits data chronologically into train/validation/test sets.

## Model Pipelines

### Multimodal Fusion Model

Implemented in `src/models/fusion.py`.

The multimodal model keeps price/technical and sentiment data in separate
temporal streams:

```text
Price/technical sequence -> TimeSeriesEncoder
Sentiment sequence       -> TemporalSentimentEncoder
```

When enabled, cross-modal attention lets the price/technical sequence attend to
the sentiment representation before final fusion. The fused representation is
passed to a single next-day return regression head.

Training modes in `train.py`:

```bash
python train.py --mode current
python train.py --mode no-sentiment
python train.py --mode sentiment-no-cross-attention
```

Sentiment source options:

```bash
python train.py --mode current --sentiment-source synthetic
python train.py --mode current --sentiment-source rss
python train.py --mode current --sentiment-source alpha_vantage
```

By default, training uses cached stock data for repeatability. To refresh Yahoo
Finance stock data:

```bash
python train.py --refresh-stock-data
```

### Standalone Baseline Models

Implemented in `model_comparison.py` and `src/models/regression_models.py`.

The standalone pipeline trains:

- LSTM
- GRU
- Transformer
- XGBoost

Feature configurations:

- `technical`: price and technical features only
- `sentiment`: sentiment features only
- `technical_sentiment`: price/technical and sentiment features concatenated

Run the comparison:

```bash
python model_comparison.py
```

Optional controls:

```bash
python model_comparison.py --seed 42 --epochs 50
```

Outputs include:

- `models/model_comparison.csv`
- `models/result/evaluation_results.csv`
- `models/result/training_summary.csv`
- Per-configuration result folders under `models/result/`
- Saved comparison checkpoints under `models/comparison_*`

## Latest Saved Comparison Results

Source: `models/model_comparison.csv`.

| Configuration | Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|
| technical | Transformer | 10.89 | 8.75 | 2.08% |
| sentiment | Transformer | 10.86 | 8.77 | 2.08% |
| technical | LSTM | 11.01 | 8.82 | 2.10% |
| technical | GRU | 10.96 | 8.84 | 2.10% |
| technical_sentiment | LSTM | 10.91 | 8.85 | 2.10% |
| technical_sentiment | Transformer | 11.01 | 8.92 | 2.12% |
| technical_sentiment | GRU | 11.04 | 8.94 | 2.12% |
| technical | XGBoost | 12.56 | 10.02 | 2.37% |
| sentiment | LSTM | 12.75 | 10.30 | 2.44% |
| technical_sentiment | XGBoost | 13.01 | 10.48 | 2.48% |
| sentiment | GRU | 14.59 | 11.97 | 2.84% |
| sentiment | XGBoost | 15.14 | 12.37 | 2.94% |

Lower is better for all metrics. These are single-seed results with seed 42.
Use the seed comparison scripts for more robust reporting.

## Setup

Recommended: Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you do not have Python 3.12 installed, use a compatible Python 3.11+
interpreter instead.

## Alpha Vantage Sentiment

Set an API key before fetching Alpha Vantage news sentiment:

```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

Fetch and cache sentiment:

```bash
PYTHONPATH=. python scripts/fetch_alpha_vantage_sentiment.py
```

This writes:

```text
data/raw/alpha_vantage_tsla_news_raw.csv
data/raw/alpha_vantage_tsla_sentiment_data.csv
data/raw/sentiment_data.csv
```

Then train with the cached Alpha Vantage source:

```bash
python train.py --mode current --sentiment-source alpha_vantage
```

Alpha Vantage free-tier limits may require multiple runs across quota resets.
The fetch script is designed to reuse cached progress.

## Common Commands

Train the current multimodal model:

```bash
python train.py --mode current --sentiment-source alpha_vantage
```

Train without sentiment:

```bash
python train.py --mode no-sentiment
```

Train with sentiment but no cross-attention:

```bash
python train.py --mode sentiment-no-cross-attention --sentiment-source alpha_vantage
```

Run standalone model comparison:

```bash
python model_comparison.py
```

Run multimodal ablation validation:

```bash
python scripts/run_ablation_validation.py --sentiment-source alpha_vantage --epochs 50
```

Run standalone seed comparison:

```bash
python scripts/run_standalone_seed_comparison.py --epochs 50
```

Run next-day inference from the saved multimodal checkpoint:

```bash
python predict.py
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Generate SHAP feature importance artifacts:

```bash
python scripts/shap_feature_importance.py
```

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── architectures/
│   ├── ARCHITECTURE.md
│   ├── multimodal_pipeline.png
│   └── standalone_models_pipeline.jpg
├── data/
│   └── raw/
├── docs/
├── models/
│   ├── comparison_technical/
│   ├── comparison_sentiment/
│   ├── comparison_technical_sentiment/
│   ├── result/
│   └── standalone_seed_results/
├── scripts/
│   ├── audit_sentiment_alignment.py
│   ├── fetch_alpha_vantage_sentiment.py
│   ├── run_ablation_validation.py
│   ├── run_standalone_seed_comparison.py
│   └── shap_feature_importance.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── config.py
├── model_comparison.py
├── predict.py
├── requirements.txt
└── train.py
```

## Reports and Architecture Notes

- `architectures/ARCHITECTURE.md` describes the end-to-end pipeline.
- `PIPELINE_COMPARISON_REPORT.md` summarizes the multimodal-vs-standalone
  comparison and research interpretation.
- `models/ablation_results.csv` stores multimodal ablation metrics.
- `models/standalone_seed_results/` stores multi-seed standalone validation
  outputs.

## Limitations

- Stock returns are noisy, non-stationary, and event-driven.
- Single chronological train/validation/test splits are useful but weaker than
  walk-forward validation.
- Synthetic sentiment is a proxy signal, not real market news sentiment.
- RSS sentiment is limited by current headline availability and is not a full
  historical sentiment source.
- Alpha Vantage data quality and coverage depend on API availability and quota.
- The current implementation is intentionally limited to next-day prediction.

## References

- Hochreiter and Schmidhuber, Long Short-Term Memory, 1997
- Cho et al., Learning Phrase Representations using RNN Encoder-Decoder, 2014
- Vaswani et al., Attention Is All You Need, 2017
- Chen and Guestrin, XGBoost: A Scalable Tree Boosting System, 2016
- Araci, FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, 2019

## License

Educational project for UTS 49275 Neural Networks and Fuzzy Logic.
