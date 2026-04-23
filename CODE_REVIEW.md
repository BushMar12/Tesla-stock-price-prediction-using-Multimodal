# Code Review Notes

This review focuses on correctness, reproducibility, methodology, and project polish. I avoided changing existing Python files because the worktree already contains unrelated local modifications in `app/streamlit_app.py` and `config.py`.

## Summary

The project has a solid structure for a coursework ML pipeline: data fetching, feature engineering, leakage-aware scaling, chronological splits, multimodal modeling, baseline comparison, CLI inference, and a Streamlit dashboard. The strongest parts are the returns-based target design, train-only scaler fitting, and the separation between fusion and baseline models.

The main risks are around reproducibility, historical sentiment quality, and keeping saved artifacts aligned with the current code.

## Findings

### 1. Real sentiment mode does not provide true historical sentiment

Location: `src/data/sentiment_data.py:129`, `src/data/sentiment_data.py:271`, `src/data/sentiment_data.py:298`

`fetch_sentiment_data(..., use_real_data=True)` fetches current RSS headlines, aggregates them by headline publication date, then merges them across the full historical stock date range. For old dates, almost every row is filled with neutral values. This means a 2010-to-present training run is mostly training on neutral sentiment, not real historical Tesla sentiment.

Impact: the model may appear multimodal while the real sentiment signal contributes little historical information. Synthetic sentiment may actually be more useful for experiments because it covers the full date range.

Recommended fix: for the coursework report, clearly state whether experiments use synthetic or live RSS sentiment. For a stronger project, cache dated historical news from a source that supports historical lookup, or restrict experiments to dates for which real headlines are available.

### 2. Feature engineering depends on live external market downloads

Location: `src/features/technical.py:191`, `src/features/technical.py:199`, `src/features/technical.py:223`, `src/features/technical.py:282`

`calculate_all_indicators()` always calls `add_market_context()`, which downloads SPY and VIX data from Yahoo Finance. This makes training, prediction, and the dashboard depend on network availability even when TSLA data is already cached.

Impact: runs can become slow or non-reproducible, and offline execution may silently replace SPY/VIX with default values. That changes the feature distribution between runs.

Recommended fix: add a config flag such as `USE_MARKET_CONTEXT`, cache SPY/VIX data under `data/raw/`, and prefer cached data when available.

### 3. Saved baseline artifacts are behind the current comparison code

Location: `model_comparison.py:25`, `src/models/regression_models.py:286`, `src/models/regression_models.py:413`, `models/model_comparison.csv`

The current code trains LSTM, GRU, Transformer, and XGBoost. The saved `models/model_comparison.csv` currently lists only LSTM, GRU, and XGBoost, and there is no saved `transformer_regressor.pt` in the `models/` folder.

Impact: the code and saved results are inconsistent. A reader may expect Transformer results in the final comparison but not see them in the artifact table.

Recommended fix: rerun `python model_comparison.py` after dependencies are installed, or update the report/README to say the saved comparison artifact only reflects the three completed baseline runs.

### 4. Training scripts override the configured start date

Location: `train.py:26`, `model_comparison.py:30`, `config.py`

`config.py` defines `START_DATE = "2021-01-01"`, but `train.py` and `model_comparison.py` call `fetch_stock_data(start_date="2010-06-29")` directly.

Impact: changing `START_DATE` in config does not affect the main training scripts, which can confuse reproduction.

Recommended fix: either use `START_DATE` from config consistently or add a clearly named training start date config field.

### 5. CLI/dashboard output contains encoding artifacts in some files

Location: `train.py`, `predict.py`, `model_comparison.py`, `src/models/regression_models.py`, `src/utils/helpers.py`, `app/streamlit_app.py`

Several print strings and labels contain mojibake from emojis and arrows. This does not usually break the code, but it makes terminal output and documentation look less professional.

Impact: report screenshots, terminal logs, and dashboard labels can look corrupted.

Recommended fix: replace those strings with ASCII labels such as `UP`, `DOWN`, `Saved`, and `->`, or ensure all files are saved as UTF-8 and the terminal uses UTF-8.

## Methodology Strengths

- Predicting returns instead of raw prices is a good design choice for this task.
- The scaler fitting happens on the training portion only, which reduces leakage risk.
- Chronological splitting is appropriate for stock forecasting.
- Multi-horizon targets make the model more useful than a one-step-only predictor.
- Baselines make the fusion model easier to evaluate fairly.

## Suggested Next Improvements

1. Regenerate baseline results so Transformer is included or intentionally removed from the comparison.
2. Add a cached-data path for SPY, VIX, TSLA, and sentiment to improve reproducibility.
3. Add a short experiment log recording date range, sentiment mode, model config, and metrics.
4. Add a simple smoke test that validates preprocessing output shapes and model forward-pass output keys.
5. Consider walk-forward validation for a stronger time-series evaluation than one fixed train/validation/test split.

## Verification Performed

- Reviewed `README.md`, `config.py`, `train.py`, `predict.py`, `model_comparison.py`, `app/streamlit_app.py`, and the core `src/` modules.
- Checked current saved baseline metrics in `models/model_comparison.csv`.
- Confirmed existing trained artifacts are present in `models/`.
- Did not rerun training because it is long-running and depends on network/data downloads.
