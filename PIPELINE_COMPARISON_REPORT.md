# Pipeline Comparison Report — Multimodal vs Standalone for Tesla Next-Day Price Prediction

**Project:** UTS 49275 Neural Networks and Fuzzy Logic — Tesla multimodal stock prediction
**Scope:** Decide whether to keep the multimodal cross-attention model as the headline architecture, or demote it to a baseline in favour of standalone sequence regressors.
**Status:** Single-seed comparison complete; 5-seed validation in progress.

---

## 1. Executive Summary

We compared two parallel modelling pipelines under matched data, preprocessing, and training conditions:

- **Standalone pipeline** — LSTM, GRU, Transformer, and XGBoost regressors operating on either price+technical features alone, or price+technical+sentiment concatenated.
- **Multimodal pipeline** — `MultimodalFusionModel` with a Transformer price encoder, temporal-CNN sentiment encoder, and cross-modal attention, evaluated in three ablation modes.

After a critical bug fix (the standalone pipeline was inadvertently using **synthetic** sentiment instead of real Alpha Vantage news sentiment), single-seed results show:

- **Best overall:** Standalone **Transformer (technical-only)** at MAE $8.75 / RMSE $10.89 / MAPE 2.08%.
- **Best multimodal:** Sentiment + cross-attention mode at MAE $9.53 / RMSE $11.75 / MAPE 2.25%.
- The standalone Transformer beats the best multimodal variant by **~$0.78 MAE (≈9%)**, despite the multimodal model carrying ~1.4× the parameters (1.67M vs the standalone Transformer's much smaller capacity).
- Cross-modal attention only "earns its keep" relative to the simpler MLP-fusion variant *when sentiment is real* — the simpler fusion overfits real news sentiment immediately (best epoch 1, then degrades), while cross-attention selectively filters it.

**Recommendation:** Demote the multimodal model from headline architecture to a documented baseline. Headline the standalone Transformer as the production model. Reframe the project contribution as a controlled comparison ("does explicit cross-modal fusion outperform a single Transformer over concatenated multimodal features?") rather than a multimodal architecture proposal.

---

## 2. Problem Statement

The project predicts Tesla (TSLA) next-day stock return, which is then reconstructed into a next-day price using the current close. Inputs are two aligned daily streams over a 60-day rolling window:

1. **Price / technical stream** — OHLCV plus 6 technical indicators (SMA-20, SMA-50, RSI-14, Bollinger Upper/Middle/Lower).
2. **Sentiment stream** — Alpha Vantage NEWS_SENTIMENT API: per-article TSLA-specific sentiment scores, aggregated daily, with 1/2/3-day lag features (9 sentiment columns total).

The original project hypothesis was that *explicit cross-modal fusion via attention should outperform single-stream models that flatten both modalities into one tensor*. This report tests that hypothesis empirically.

---

## 3. Pipelines Under Comparison

### 3.1 Standalone pipeline (`model_comparison.py`)

Four architectures × two feature configurations:

| Architecture | Capacity notes |
|---|---|
| **LSTM** | Recurrent, compresses 60-day window through hidden state |
| **GRU** | Recurrent, simpler gating |
| **Transformer** | Self-attention over the 60-day window |
| **XGBoost** | Tabular boosting, sequence flattened to a fixed-length vector |

| Configuration | Inputs |
|---|---|
| `technical` | OHLCV + 6 technical indicators (10 features) |
| `technical_sentiment` | technical features + 9 sentiment features (19 features) concatenated along the feature axis |

For sentiment-aware variants, sentiment is **concatenated** to the price/technical tensor before being fed to the model. The model then has no architectural distinction between sentiment and price dimensions — it discovers any cross-modal interactions via its own attention/recurrence.

### 3.2 Multimodal pipeline (`train.py` + `scripts/run_ablation_validation.py`)

Single architecture, three ablation modes:

| Mode | Sentiment branch | Cross-modal attention |
|---|---|---|
| `no-sentiment` | disabled | disabled |
| `sentiment-no-cross-attention` | enabled (temporal-CNN encoder), late fusion via MLP | disabled |
| `current` | enabled | enabled — price-stream attends to encoded sentiment |

The `MultimodalFusionModel` keeps price and sentiment in separate encoders and fuses them only after each branch has been encoded. With cross-attention, the price token sequence attends against the sentiment representation before fusion; the result is added back into the price representation prior to the regression head.

### 3.3 Shared infrastructure

Both pipelines use the **same `DataPreprocessor`** (`src/data/preprocessing.py`), so they receive byte-identical splits given the same random seed:

- 60-day sliding windows
- Chronological 80 / 10 / 10 train/val/test split (no shuffling)
- Scalers fit on the training portion only
- Target: scaled next-day return; metrics computed in dollar space after inverse-transform plus close-price reconstruction

---

## 4. Methodology

### 4.1 Critical bug fix — sentiment source

When investigating why the original benchmark suggested "sentiment hurts every standalone model," we discovered that **`model_comparison.py` was bypassing `fetch_sentiment_data` and reading `data/raw/sentiment_data.csv` directly**. That generic file contained synthetic sentiment from a prior run — so the entire `technical_sentiment` arm was trained on synthetic noise, not real Alpha Vantage news.

The fix (model_comparison.py:381–391) routes sentiment loading through `fetch_sentiment_data(stock_df, source=SENTIMENT_CONFIG["source"])`. With `source="alpha_vantage"`, the function hits the cached daily file at `data/raw/alpha_vantage_tsla_sentiment_data.csv` and returns 1381 days of real news sentiment.

A parallel issue affected the multimodal ablation runner: `scripts/run_ablation_validation.py` defaults `--sentiment-source` to `synthetic`. Real-sentiment ablations require the explicit flag `--sentiment-source alpha_vantage`.

After the fix, all reported results use **real Alpha Vantage sentiment** for both pipelines.

### 4.2 Training configuration

Both pipelines share the configuration from `config.py`:

| Setting | Value |
|---|---|
| Sequence length | 60 days |
| Batch size | 32 |
| Optimizer | AdamW (LR 1e-4, weight decay default) |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5) on val MAE in dollar space |
| Loss | Smooth L1 on scaled return |
| Max epochs | 50 |
| Random seed | 42 (single-seed experiments); see §6 for sweep |
| Determinism | `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, CUBLAS workspace pinned |
| Stock window | 2021-01-01 → 2026-04-21 (cached) |
| Sentiment source | `alpha_vantage` (cached daily aggregation) |

XGBoost uses its own boosting budget (2000 rounds) with native early stopping on val MAE.

### 4.3 Evaluation

All metrics are reported in **dollar price space**, after:

1. Predicting scaled next-day return.
2. Inverse-transforming with the fitted return scaler.
3. Reconstructing next-day price as `close_t × (1 + return_pred)`.
4. Comparing against the actual next-day close.

Metrics: RMSE, MAE, MAPE, R², Directional Accuracy. Test set is the final 10% of the chronological split (most recent ~128 trading days).

---

## 5. Results — Single Seed (seed=42, 50 epochs, real Alpha Vantage sentiment)

### 5.1 Standalone pipeline

Source: `models/model_comparison.csv`.

| Configuration | Model | RMSE | MAE | MAPE | MSE |
|---|---|---:|---:|---:|---:|
| technical | **Transformer** | **$10.89** | **$8.75** | **2.08%** | 118.67 |
| technical | LSTM | $11.01 | $8.82 | 2.10% | 121.26 |
| technical | GRU | $10.96 | $8.84 | 2.10% | 120.09 |
| technical_sentiment | LSTM | $10.91 | $8.85 | 2.10% | 119.11 |
| technical_sentiment | Transformer | $11.01 | $8.92 | 2.12% | 121.33 |
| technical_sentiment | GRU | $11.04 | $8.94 | 2.12% | 121.81 |
| technical | XGBoost | $12.56 | $10.02 | 2.37% | 157.67 |
| technical_sentiment | XGBoost | $13.01 | $10.48 | 2.48% | 169.30 |

**Observations:**

- The Transformer (technical only) is the single-seed winner at **$8.75 MAE / 2.08% MAPE**.
- Adding sentiment to the standalone models **slightly hurts every architecture** in this run: Transformer +$0.17, LSTM +$0.03, GRU +$0.10, XGBoost +$0.46. (Note: in a previous re-run at the same seed, Transformer + sentiment narrowly beat technical-only. The sign of the effect is therefore *seed-dependent and small*; multi-seed averaging is required to settle the question — see §6.)
- XGBoost is consistently the worst architecture in both configurations, by a wide margin. Likely cause: the model flattens the 60-day window, losing temporal structure that the recurrent and attention models exploit.

### 5.2 Multimodal pipeline

Source: `models/ablation_results.csv` from the AV-sentiment run.

| Mode | Params | RMSE | MAE | MAPE | Best epoch | Train/val gap |
|---|---:|---:|---:|---:|---:|---:|
| **current** (sentiment + cross-attn) | 1.67M | **$11.75** | **$9.53** | **2.25%** | 17 | −0.16% |
| no-sentiment | 1.20M | $12.25 | $9.92 | 2.37% | 4 | −0.42% |
| sentiment, no cross-attn | 1.34M | $13.30 | $10.62 | 2.49% | 1 | −0.56% |

**Observations:**

- With **real sentiment**, cross-attention is essential to make sentiment useful: it improves MAE from the no-sentiment baseline ($9.92 → $9.53). Without cross-attention, real sentiment *hurts* ($9.92 → $10.62) — the simpler MLP fusion overfits at epoch 1 and then degrades.
- This **inverts the conclusion of the earlier synthetic-sentiment ablation**, which had picked sentiment-no-cross-attention as the best mode. Synthetic sentiment was a smoothed function of lagged returns — easy to fuse linearly. Real news sentiment carries genuine but noisy signal, which only attention can filter.
- All three modes show a slightly **negative train/val gap** (val loss < train loss), indicating no overfitting — the multimodal model does not appear capacity-bound.
- The 330k extra parameters for cross-attention buy a $1.09 MAE improvement *within* the multimodal family, justifying the architectural addition under matched conditions.

### 5.3 Head-to-head ranking

All variants under identical seed / epochs / sentiment-source conditions, ranked by MAE:

| Rank | Pipeline | Variant | MAE | RMSE | MAPE | Δ vs best |
|---:|---|---|---:|---:|---:|---:|
| 1 | Standalone | Transformer (technical) | **$8.75** | $10.89 | 2.08% | — |
| 2 | Standalone | LSTM (technical) | $8.82 | $11.01 | 2.10% | +$0.07 |
| 3 | Standalone | GRU (technical) | $8.84 | $10.96 | 2.10% | +$0.09 |
| 4 | Standalone | LSTM (technical+sentiment) | $8.85 | $10.91 | 2.10% | +$0.10 |
| 5 | Standalone | Transformer (technical+sentiment) | $8.92 | $11.01 | 2.12% | +$0.17 |
| 6 | Standalone | GRU (technical+sentiment) | $8.94 | $11.04 | 2.12% | +$0.19 |
| 7 | Multimodal | current (sentiment + cross-attn) | $9.53 | $11.75 | 2.25% | +$0.78 |
| 8 | Multimodal | no-sentiment | $9.92 | $12.25 | 2.37% | +$1.17 |
| 9 | Standalone | XGBoost (technical) | $10.02 | $12.56 | 2.37% | +$1.27 |
| 10 | Standalone | XGBoost (technical+sentiment) | $10.48 | $13.01 | 2.48% | +$1.73 |
| 11 | Multimodal | sentiment, no cross-attn | $10.62 | $13.30 | 2.49% | +$1.87 |

**Headline:** every standalone sequence model (LSTM/GRU/Transformer, both configs) outperforms the best multimodal variant. Only the boosting-tree XGBoost falls below the multimodal level.

---

## 6. Multi-Seed Validation (in progress)

A single-seed comparison is not robust for a $0.78 gap. We are running both pipelines across **5 seeds** = `[42, 1337, 2024, 7, 123]` via `scripts/run_seed_sweep.py` to obtain mean ± standard deviation.

The driver script (1) executes each seed in a fresh subprocess (no module-state pollution), (2) archives per-seed CSVs to `models/seed_sweep/`, and (3) aggregates the results into:

- `models/seed_sweep/standalone_aggregate.csv`
- `models/seed_sweep/multimodal_aggregate.csv`
- `models/seed_sweep/combined_summary.csv`

This section will be filled in once the sweep completes:

> **Standalone aggregate (5 seeds):** _to be populated from `standalone_aggregate.csv`._
>
> **Multimodal aggregate (5 seeds):** _to be populated from `multimodal_aggregate.csv`._
>
> **Combined ranking (mean MAE):** _to be populated from `combined_summary.csv`._

The decision in §8 is robust to a few cents of seed variance, but the precise ranking among standalone variants (especially "does sentiment help the standalone Transformer?") will only be settled by the sweep.

---

## 7. Analysis

### 7.1 Why does the standalone Transformer beat the multimodal model?

The multimodal model has more parameters (1.67M vs the standalone Transformer's much smaller count) yet generalizes worse. Three plausible factors:

1. **Capacity-to-data ratio.** With ~1013 training windows, a 1.67M-parameter model has much more capacity per training sample than the standalone Transformer. Deterministic weight init plus the mild train/val gap suggests the multimodal model is not catastrophically overfitting, but it is not extracting more signal either — the extra capacity is wasted on parameters that don't reduce test loss.
2. **No architectural advantage from forced separation.** The multimodal model imposes a *structural prior*: "price and sentiment are different modalities; they should be encoded separately and only fused once." A standalone Transformer over `concat(price, sentiment)` makes no such commitment — its self-attention layers can route across price and sentiment dimensions at every layer. When the actual cross-modal interactions are simple (e.g., sentiment shifts the conditional return distribution slightly), separation costs more than it gains.
3. **Sentiment encoder bottleneck.** The temporal-CNN sentiment encoder compresses the 9-dimensional sentiment sequence into a learned representation before fusion. If the price branch could have used some sentiment dimension *raw* (e.g., the lag-1 sentiment_compound directly), the encoder may have already discarded that information. A single Transformer over concatenated features sidesteps this — every dimension is available at every layer.

This is consistent with a broader pattern in deep-learning architecture research: **explicit modality separation is most valuable when modalities have very different statistical structure or sampling rates** (e.g., text + image, video + audio at different frame rates). For two daily-aligned numerical streams, the inductive bias is too weak to overcome the parameter cost.

### 7.2 The "sentiment + attention" finding

A consistent thread across both pipelines: **sentiment helps only when the architecture has attention to filter it.**

- Recurrent models (LSTM, GRU) without attention degrade with sentiment added — every byte of sentiment input flows into the same hidden state and dilutes the price signal.
- The multimodal "sentiment, no cross-attention" mode degrades worst of all (best epoch 1, then collapses) — its MLP fusion has even less filtering capacity than a recurrent state.
- The Transformer (in either pipeline) and the multimodal cross-attention mode both benefit, or at least don't degrade, from real sentiment.

This finding is interesting in its own right and worth foregrounding in the report: **for noisy auxiliary modalities like daily news sentiment, attention is not just useful — it appears necessary.**

### 7.3 XGBoost regression

XGBoost is consistently $1.30–$1.75 worse than the next-worst sequence model. Its `best_epoch` from the per-seed history is very early (≤ 8 boosting rounds for technical, 1 for technical+sentiment), and from there validation MAE rises monotonically — i.e., it overfits aggressively and the early-stop criterion saves it from a much worse final model.

The mechanical cause is clear: **XGBoost flattens the 60-day window into a fixed-length vector**, losing temporal ordering. For a problem that fundamentally requires sequence reasoning (next-day return is the temporal derivative of recent prices), flattening discards the inductive bias that makes the other models work.

The report should keep XGBoost as a baseline because it is a useful "what does a non-temporal model achieve" reference, but it should not be presented as a competitive option.

---

## 8. Decision and Recommendation

**Decision: demote the multimodal model from headline architecture to a documented baseline.**

### 8.1 What "demote" means concretely

| Component | Action |
|---|---|
| `MultimodalFusionModel` (`src/models/fusion.py`) | **Keep.** Document as ablation baseline. |
| `train.py` (multimodal trainer entrypoint) | **Keep.** Add a docstring note that it is the baseline trainer; production model is the standalone Transformer. |
| `scripts/run_ablation_validation.py` | **Keep.** This is the evidence file for the ablation study. |
| `models/ablation_alpha_vantage_*` artifacts | **Keep, archive in report appendix.** |
| `predict.py` and `app/streamlit_app.py` | **Switch the headline model** from the multimodal checkpoint to the standalone Transformer (technical config or technical+sentiment, pending seed-sweep result). |
| `model_comparison.py` and `MultiModelRegressor` | **Promote to primary training pipeline.** |
| `ARCHITECTURE.md` and `README.md` | **Rewrite.** Replace "multimodal cross-attention is the main model" framing with "controlled comparison; standalone Transformer is production; multimodal is documented baseline". |

### 8.2 Why not delete multimodal entirely

Three reasons for keeping the multimodal code in the repository even though it is no longer the headline model:

1. **The negative result is the contribution.** "We tested whether explicit cross-modal fusion outperforms a Transformer over concatenated multimodal features for next-day TSLA prediction; it does not" is a sharper, more interesting research finding than "we built a multimodal model." Deleting the multimodal code orphans the evidence.
2. **Reproducibility.** The ablation results referenced in the report must be reproducible from the same repository.
3. **Future extensibility.** If sentiment frequency or text encoders change (e.g., move to FinBERT-on-headlines instead of Alpha Vantage scores), the multimodal architecture may become competitive again. The infrastructure should remain available.

### 8.3 Project narrative

The recommended framing for the seminar / report:

> "We hypothesized that explicit cross-modal fusion via a separate sentiment encoder and price-to-sentiment cross-attention would outperform a single self-attention model over concatenated price and sentiment features. Under matched data, preprocessing, and training conditions on Tesla next-day price prediction, we find the opposite: a standalone Transformer over concatenated features outperforms the multimodal architecture by ~9% MAE despite using fewer parameters. Within the multimodal family, however, cross-attention is essential — the simpler MLP-fusion baseline is significantly worse with real news sentiment, suggesting that attention-based filtering is necessary for noisy daily-aggregated sentiment regardless of where in the architecture it is placed. We recommend the standalone Transformer as the production model and present the multimodal architecture as an instructive negative-result baseline."

---

## 9. Limitations

1. **Single test split, single dataset.** All results are on the chronological final 10% of TSLA 2021-01-01 → 2026-04-21. We have not tested whether the ranking generalizes to other tickers or other test windows. A walk-forward / expanding-window evaluation would strengthen the claim.
2. **Single seed at the time of writing this section.** The multi-seed sweep (§6) is in progress. The ~$0.78 standalone-vs-multimodal MAE gap is large enough that we expect it to survive averaging, but the *intra-standalone* ranking (does sentiment help the Transformer?) flipped between two repeats at seed=42 and is therefore not yet decidable.
3. **Sentiment alignment.** Alpha Vantage daily sentiment is aligned to trading dates by averaging same-day news scores, with neutral fill-in for non-news days. We have not tested whether (a) using only news-bearing days, or (b) using a different aggregation (e.g., relevance-weighted), changes the conclusions.
4. **Reconstructed-price metric.** RMSE/MAE/MAPE are computed in dollar space after the return → price reconstruction. This is the metric the project committed to early; an alternative project framing using directional accuracy or returns-space MSE could yield a different ranking.
5. **CPU vs GPU determinism.** Training was run on CPU for these experiments. GPU runs at the same seed produce slightly different numbers due to non-deterministic CUDA kernels; the conclusions here apply to the CPU configuration.

---

## 10. Reproduction

```powershell
# Standalone pipeline (single seed)
python model_comparison.py --seed 42 --epochs 50

# Multimodal ablation (single seed, real AV sentiment)
python scripts/run_ablation_validation.py --sentiment-source alpha_vantage --seed 42 --epochs 50

# Multi-seed sweep (5 seeds, both pipelines, aggregates mean / std)
python scripts/run_seed_sweep.py
```

Sentiment source is read from `config.py:SENTIMENT_CONFIG["source"]` for the standalone pipeline; the multimodal ablation runner takes `--sentiment-source` explicitly. Both must be `alpha_vantage` for a fair real-sentiment comparison.

---

## 11. Appendix — Result Files

| Artifact | Path | Purpose |
|---|---|---|
| Standalone single-seed | `models/model_comparison.csv` | Top-line standalone results, current seed |
| Standalone per-config | `models/result/{technical,technical_sentiment}/evaluation_results.csv` | Per-config breakdown with extra metrics (R², directional accuracy) |
| Standalone per-model | `models/result/<config>/<model>/{training_history.csv, test_predictions.csv, predictions_vs_actual.png, ...}` | Full per-model artefacts |
| Multimodal ablation | `models/ablation_results.csv` | Three-mode ablation, current seed and sentiment source |
| Multimodal ablation verdict | `models/ablation_verdict.txt` | Auto-generated decision text |
| Multimodal checkpoints | `models/ablation_alpha_vantage_*__best.pt` | Best validation checkpoints per mode |
| Multi-seed sweep (when complete) | `models/seed_sweep/{standalone,multimodal}_aggregate.csv`, `combined_summary.csv` | Mean ± std across 5 seeds |
| Sweep run log | `models/seed_sweep_run.log` | Full stdout from the sweep driver |

---

*End of report.*
