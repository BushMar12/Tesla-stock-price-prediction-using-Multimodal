# Next-Day-Only Refactor — Design

**Date:** 2026-04-30
**Status:** Draft, pending implementation plan
**Owner:** BushMar12

## Goal & Scope

Strip the project down to a single objective: predict next-day TSLA return, reconstruct next-day price, and report MAE on the held-out test set. Multi-day forecasting and 3-class direction classification are removed entirely from the model, training, inference, dashboard, and config.

Add `ReduceLROnPlateau` plus early stopping (on val MAE) to both the multimodal model and the standalone baselines. Save the best-val-MAE checkpoint as the final model; reload it before test evaluation. The single chronological 80/10/10 split is retained.

### Out of Scope

- Walk-forward / rolling-window cross-validation
- Naïve baseline (random-walk / persistence)
- Multi-seed runs and bootstrap confidence intervals
- Synthetic-sentiment leakage audit
- Adding an automated test harness
- Dependency cleanup or `requirements.txt` consolidation

These remain candidates for separate future specs.

## Components & Files Touched

### Model (`src/models/fusion.py`)
- Remove `multi_day_head` and `direction_head` from `MultimodalFusionModel`.
- Forward pass returns a single tensor (1-day return prediction) instead of a dict.

### Trainer (`src/models/trainer.py`)
- Remove multi-day SmoothL1 loss term and direction cross-entropy loss term. Loss reduces to SmoothL1 on the 1-day return.
- Replace `OneCycleLR` with `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` monitoring val MAE in dollar space.
- Add early-stopping logic: stop after `EARLY_STOPPING_PATIENCE = 15` epochs without val-MAE improvement, with a `MIN_EPOCHS = 10` floor.
- On training end, reload best-val-MAE checkpoint before test evaluation.
- Remove direction-accuracy logging and reporting.

### Baselines (`src/models/regression_models.py`, `model_comparison.py`)
- Apply the same `ReduceLROnPlateau` + early stopping + best-checkpoint pattern to LSTM, GRU, and Transformer training loops.
- XGBoost uses its native `early_stopping_rounds` against the same val-MAE objective via `eval_set=[(X_val, y_val)]`.
- Drop direction-accuracy column from `model_comparison.csv`.

### Preprocessing (`src/data/preprocessing.py`)
- Remove multi-day target tensor construction. Only the 1-day return target is built.
- Remove the 3-class direction label construction.

### Config (`config.py`)
- Delete `PREDICTION_HORIZONS`, `DIRECTION_RETURN_THRESHOLD`, `multi_day_weight`, `classification_weight`.
- Add `EARLY_STOPPING_PATIENCE = 15`, `MIN_EPOCHS = 10`, `LR_PATIENCE = 5`, `LR_FACTOR = 0.5`.

### Inference (`predict.py`)
- Output a single value: predicted next-day price.

### Dashboard (`app/streamlit_app.py`)
- Remove multi-day forecast section, direction prediction card, and direction-accuracy chart.
- Keep: next-day price card, historical chart, model comparison table (MAE/RMSE/MAPE only), SHAP tab.

### Train Script (`train.py`)
- Retain `--mode {current, no-sentiment, sentiment-no-cross-attention}` — these remain meaningful ablations for next-day MAE.
- Retain `--sentiment-source {synthetic, rss, alpha_vantage}`.

## Data Flow

```
OHLCV + indicators + market + calendar + sentiment
  -> merged feature table
  -> scalers fit on train portion only
  -> 60-day sliding windows
  -> chronological 80/10/10 split
  -> (price_seq_tensor, sentiment_seq_tensor, target_1d_return)
```

Single target per sample. No multi-day target tensor. No direction label tensor.

## Training Loop

```
for epoch in range(MAX_EPOCHS):
    train_loss = SmoothL1(pred_1d, target_1d)   # only loss term
    val_mae   = evaluate(val_loader)            # dollar space, inverse-transformed

    scheduler.step(val_mae)                     # ReduceLROnPlateau

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        save_checkpoint("best.pt")
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    if epoch >= MIN_EPOCHS and epochs_since_improve >= EARLY_STOPPING_PATIENCE:
        break

load_checkpoint("best.pt")
test_metrics = evaluate(test_loader)            # MAE, RMSE, MAPE
```

### Invariants

- Val MAE is computed in dollar space (inverse-transformed predicted return → reconstructed price → MAE vs. actual price). Same metric the test set reports — no scaled-vs-dollar mismatch.
- `MAX_EPOCHS = 100` is a budget ceiling, not a training duration.
- The same loop applies to LSTM, GRU, and Transformer (each with their own `best.pt`). XGBoost uses native `early_stopping_rounds` against the same val-MAE objective.

## Error Handling & Edge Cases

- **Early stopping never triggers.** Training runs the full `MAX_EPOCHS = 100` budget. Acceptable — matches existing behavior.
- **Early stopping triggers too early.** Mitigated by `MIN_EPOCHS = 10` floor before early stopping is allowed to fire.
- **No improvement ever.** Save an initial checkpoint at epoch 0 so `best.pt` always exists when the loop ends.
- **Checkpoint file paths.** Each model writes to its own path: `models/multimodal_best.pt`, `models/lstm_best.pt`, `models/gru_best.pt`, `models/transformer_best.pt`, `models/xgboost_best.pkl`. Existing files (`lstm_regressor.pt`, etc.) are deleted as part of the refactor commit.
- **Old checkpoints unloadable.** Existing multi-head checkpoints have a different `state_dict` shape and forward signature; they cannot be loaded by the simplified model. Git history preserves them; user retrains from scratch.
- **Streamlit loads stale checkpoint.** Add a `MODEL_VERSION` string to the checkpoint dict; the app shows a clear "retrain required" message on version mismatch instead of a stack trace.
- **XGBoost early stopping.** `xgboost.XGBRegressor.fit` requires `eval_set=[(X_val, y_val)]` and `early_stopping_rounds` passed at fit time. Small refactor in `model_comparison.py`.

## Testing & Validation

This is a refactor without new behavior. Validation is regression-style: prove the simplified pipeline still trains and evaluates cleanly, and prove no loose ends remain.

### Smoke Tests (manual, after the refactor)

1. `python train.py --mode current --sentiment-source synthetic` — completes without error, produces `models/multimodal_best.pt`, prints test MAE/RMSE/MAPE.
2. `python train.py --mode no-sentiment` — same.
3. `python train.py --mode sentiment-no-cross-attention` — same.
4. `python model_comparison.py` — produces `models/model_comparison.csv` with 4 rows (LSTM, GRU, Transformer, XGBoost), columns MAE/RMSE/MAPE only.
5. `python predict.py` — prints a single next-day price.
6. Streamlit dashboard launches; next-day card, model-comparison table, and SHAP tab all render with no broken sections.

### Sanity Checks on the Training Run

- Early stopping triggers on at least one of the three multimodal modes (otherwise patience is too high).
- Best-checkpoint test MAE is less than or equal to last-epoch test MAE (otherwise checkpoint reload is broken).
- `ReduceLROnPlateau` reduces LR at least once over a full run (visible in logs).

### Code-Level Checks

- Grep confirms no remaining references to `multi_day`, `direction`, `PREDICTION_HORIZONS`, `DIRECTION_RETURN_THRESHOLD`, `classification_weight`, or `multi_day_weight` outside of git history and the `.docx` reports.
- Old checkpoint files (`lstm_regressor.pt`, `gru_regressor.pt`, `transformer_regressor.pt`, `xgboost_regressor.pkl`) are deleted from `models/`.

### Acceptance Criterion

All 6 smoke tests pass, and the headline next-day MAE on the multimodal model is within ±20% of the pre-refactor value. This is a sanity check that the refactor did not silently break the model, not a performance target.
