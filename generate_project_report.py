"""
Generate an updated detailed PDF report for the Tesla stock prediction project.

The generator uses Pillow only, so it does not require LaTeX, pandoc,
wkhtmltopdf, reportlab, or matplotlib.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import joblib
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401

from config import (
    DIRECTION_RETURN_THRESHOLD,
    MARKET_CONTEXT_CACHE,
    MODEL_CONFIG,
    PREDICTION_HORIZONS,
    SEQUENCE_LENGTH,
    SENTIMENT_CONFIG,
    START_DATE,
    STOCK_SYMBOL,
    STREAMLIT_CONFIG,
    TRAINING_CONFIG,
    USE_MARKET_CONTEXT,
)


ROOT = Path(__file__).parent
OUTPUT = ROOT / "Tesla_Stock_Prediction_Updated_Report.pdf"

PAGE_W, PAGE_H = 1240, 1754
MARGIN_X = 105
TOP_Y = 115
BOTTOM_Y = 1660
BLUE = (31, 78, 121)
TEXT = (35, 35, 35)
MUTED = (95, 95, 95)
LIGHT_BG = (245, 247, 251)

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR_FONT = FONT_DIR / "Arial.ttf"
BOLD_FONT = FONT_DIR / "Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD_FONT if bold and BOLD_FONT.exists() else REGULAR_FONT
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


F_TITLE = font(42, True)
F_SUBTITLE = font(29)
F_H1 = font(28, True)
F_H2 = font(19, True)
F_BODY = font(15)
F_BODY_BOLD = font(15, True)
F_SMALL = font(12)
F_FOOTER = font(10)


def blank_page(title: str | None = None) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    if title:
        draw.text((MARGIN_X, 58), title, font=F_H1, fill=TEXT)
        draw.line((MARGIN_X, 101, PAGE_W - MARGIN_X, 101), fill=BLUE, width=3)
        draw.text(
            (MARGIN_X, PAGE_H - 58),
            "Tesla Stock Price Prediction Using Multimodal Deep Learning",
            font=F_FOOTER,
            fill=MUTED,
        )
    return page


def wrap_pixels(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=fnt) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def add_text_pages(pages: list[Image.Image], title: str, sections: list[tuple[str, list[str]]]):
    page = blank_page(title)
    draw = ImageDraw.Draw(page)
    y = TOP_Y

    def flush():
        nonlocal page, draw, y
        pages.append(page)
        page = blank_page(title)
        draw = ImageDraw.Draw(page)
        y = TOP_Y

    for heading, paragraphs in sections:
        if y > BOTTOM_Y - 90:
            flush()
        if heading:
            draw.text((MARGIN_X, y), heading, font=F_H2, fill=BLUE)
            y += 34
        for paragraph in paragraphs:
            bullet = paragraph.startswith("- ")
            text = paragraph[2:] if bullet else paragraph
            left = MARGIN_X + 28 if bullet else MARGIN_X
            width = PAGE_W - MARGIN_X - left
            lines = wrap_pixels(draw, text, F_BODY, width)
            if y + 24 * len(lines) > BOTTOM_Y:
                flush()
            if bullet:
                draw.text((MARGIN_X + 6, y), "-", font=F_BODY_BOLD, fill=TEXT)
            for line in lines:
                draw.text((left, y), line, font=F_BODY, fill=TEXT)
                y += 24
            y += 15
        y += 8

    pages.append(page)


def add_title_page(pages: list[Image.Image]):
    page = Image.new("RGB", (PAGE_W, PAGE_H), LIGHT_BG)
    draw = ImageDraw.Draw(page)
    draw.rectangle((90, 150, PAGE_W - 90, PAGE_H - 150), outline=BLUE, width=4)
    draw.text((130, 330), "Tesla Stock Price Prediction", font=F_TITLE, fill=BLUE)
    draw.text((130, 400), "Using Multimodal Deep Learning", font=F_SUBTITLE, fill=TEXT)
    draw.text((130, 535), "Updated Detailed Project Report", font=F_H1, fill=TEXT)
    lines = [
        "Includes model architectures, training settings, retrained baseline results,",
        "multimodal training options, and project scope/limitations.",
        "UTS 49275 Neural Networks and Fuzzy Logic",
    ]
    y = 660
    for line in lines:
        draw.text((130, y), line, font=F_BODY, fill=TEXT)
        y += 31
    draw.text((130, PAGE_H - 260), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", font=F_SMALL, fill=MUTED)
    pages.append(page)


def add_image_page(pages: list[Image.Image], title: str, image_path: Path, caption: str):
    if not image_path.exists():
        return
    page = blank_page(title)
    draw = ImageDraw.Draw(page)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((PAGE_W - 2 * MARGIN_X, 980), Image.Resampling.LANCZOS)
    x = (PAGE_W - img.width) // 2
    y = 210
    page.paste(img, (x, y))
    caption_y = y + img.height + 45
    for line in wrap_pixels(draw, caption, F_SMALL, PAGE_W - 2 * MARGIN_X):
        draw.text((MARGIN_X, caption_y), line, font=F_SMALL, fill=MUTED)
        caption_y += 20
    pages.append(page)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return joblib.load(path)
    except Exception:
        return {}


def format_baseline_results() -> list[str]:
    rows = read_csv_rows(ROOT / "models" / "model_comparison.csv")
    if not rows:
        return ["No saved model_comparison.csv file was found."]
    return [
        f"- {row['Model']}: RMSE {row['RMSE']}, MAE {row['MAE']}, MAPE {row['MAPE']}, Direction Accuracy {row['Dir. Accuracy']}"
        for row in rows
    ]


def active_model_lines() -> list[str]:
    metadata = load_metadata(ROOT / "models" / "preprocessing_metadata.pkl")
    if not metadata:
        return ["No active preprocessing metadata was found."]
    return [
        f"- Active fusion training mode: {metadata.get('training_mode', 'not recorded')}",
        f"- Uses sentiment branch: {metadata.get('use_sentiment', 'not recorded')}",
        f"- Uses cross-attention: {metadata.get('use_cross_attention', 'not recorded')}",
        f"- Price/technical feature count: {metadata.get('n_price_features', 'not recorded')}",
        f"- Sentiment feature count: {metadata.get('n_sentiment_features', 'not recorded')}",
        f"- Sequence length: {metadata.get('sequence_length', SEQUENCE_LENGTH)} trading days",
        f"- Prediction horizons: {metadata.get('horizons', PREDICTION_HORIZONS)}",
    ]


def report_content() -> list[tuple[str, list[tuple[str, list[str]]]]]:
    return [
        (
            "Executive Summary",
            [
                (
                    "Project Aim",
                    [
                        f"This project predicts {STOCK_SYMBOL} stock movement using a returns-based forecasting pipeline. Instead of predicting raw closing prices directly, models predict future percentage returns and reconstruct prices afterward.",
                        "The key research question is whether a dedicated multimodal fusion architecture improves next-day price prediction compared with simpler standalone models trained on the same engineered feature set.",
                    ],
                ),
                (
                    "What Changed in the Latest Version",
                    [
                        "- The multimodal model now supports three training options: current setting, no sentiment, and sentiment without cross-attention.",
                        "- Regression is prioritized for MAE-focused evaluation with regression_weight = 1.0 and classification_weight = 0.1.",
                        "- Multi-day prediction is treated as a secondary objective with multi_day_weight = 0.05.",
                        "- All neural training is configured for 100 epochs because convergence was observed around epoch 50.",
                    ],
                ),
            ],
        ),
        (
            "Scope and Limitations",
            [
                (
                    "Project Scope",
                    [
                        "The project is an educational financial sequence-modelling pipeline, not a production trading system. It focuses on model design, reproducibility, evaluation, and interpretability for next-day Tesla stock forecasting.",
                        "The primary metric is MAE on reconstructed next-day price. RMSE, MAPE, direction accuracy, and multi-day forecasts are secondary diagnostics.",
                    ],
                ),
                (
                    "Limitations",
                    [
                        "- Stock returns are noisy, non-stationary, and strongly affected by news/events not captured in OHLCV history.",
                        "- Live RSS sentiment is not true historical sentiment; long historical runs mostly rely on synthetic sentiment or neutral-filled current RSS data.",
                        "- A fixed chronological split is useful but weaker than walk-forward validation across multiple market regimes.",
                        "- Direction accuracy is a secondary three-class task and can be below 50% even when price MAE improves.",
                        "- The XGBoost model is saved as a pickle artifact; incompatible XGBoost native versions can crash while loading, so Streamlit skips live XGBoost loading.",
                    ],
                ),
            ],
        ),
        (
            "Data and Feature Engineering",
            [
                (
                    "Data Sources",
                    [
                        f"- Stock symbol: {STOCK_SYMBOL}.",
                        f"- Historical start date: {START_DATE}.",
                        f"- Market context enabled: {USE_MARKET_CONTEXT}; market context cache enabled: {MARKET_CONTEXT_CACHE}.",
                        f"- Sentiment real-data fetch default: {SENTIMENT_CONFIG['use_real_data_fetch']}; Streamlit real sentiment default: {STREAMLIT_CONFIG['sentiment_use_real_data']}.",
                    ],
                ),
                (
                    "Feature Groups",
                    [
                        "- Price/OHLCV: open, high, low, close, volume, returns, log returns, and price range features.",
                        "- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, stochastic oscillator, momentum, volatility, candlestick bodies/shadows, and gaps.",
                        "- Market context: SPY close/return, TSLA-vs-SPY alpha, and VIX.",
                        "- Calendar context: day of week, month sine/cosine, month end, and quarter end.",
                        "- Sentiment: compound, positive, negative, neutral, news count, and lagged sentiment values.",
                    ],
                ),
            ],
        ),
        (
            "Training Settings and Configuration",
            [
                (
                    "Core Configuration",
                    [
                        f"- Sequence length: {SEQUENCE_LENGTH} trading days.",
                        f"- Prediction horizons: {PREDICTION_HORIZONS}.",
                        f"- Direction neutral threshold: +/-{DIRECTION_RETURN_THRESHOLD * 100:.2f}%.",
                        f"- Batch size: {TRAINING_CONFIG['batch_size']}.",
                        f"- Learning rate: {TRAINING_CONFIG['learning_rate']}.",
                        f"- Epochs: {TRAINING_CONFIG['epochs']}.",
                        f"- Chronological split: train {TRAINING_CONFIG['train_split']:.0%}, validation {TRAINING_CONFIG['val_split']:.0%}, test {TRAINING_CONFIG['test_split']:.0%}.",
                    ],
                ),
                (
                    "Loss Priorities",
                    [
                        f"- 1-day regression loss weight: {TRAINING_CONFIG['regression_weight']}.",
                        f"- Direction classification loss weight: {TRAINING_CONFIG['classification_weight']}.",
                        "- Multi-day regression loss weight: 0.05.",
                        "- The objective is intentionally MAE/regression-oriented; classification and multi-day forecasts are auxiliary.",
                    ],
                ),
                (
                    "Active Saved Fusion Model",
                    active_model_lines(),
                ),
            ],
        ),
        (
            "Architecture: Multimodal Fusion Model",
            [
                (
                    "Input Structure",
                    [
                        "The multimodal model uses two streams: a price/technical stream and a sentiment stream. This differs from the standalone models, which concatenate all features into a single input tensor.",
                        "The active saved model currently uses sentiment and cross-attention according to preprocessing metadata.",
                    ],
                ),
                (
                    "Architecture Details",
                    [
                        f"- Time-series encoder: LSTM-based encoder with hidden size {MODEL_CONFIG['ts_hidden_size']}, {MODEL_CONFIG['ts_num_layers']} layers, dropout {MODEL_CONFIG['ts_dropout']}, bidirectional = {MODEL_CONFIG['ts_bidirectional']}.",
                        f"- Sentiment encoder: temporal CNN encoder that maps sentiment sequences into a {MODEL_CONFIG['sentiment_hidden_dim']}-dimensional representation.",
                        "- Cross-modal attention: optional multi-head attention where the time-series sequence attends to the encoded sentiment representation.",
                        f"- Fusion MLP: hidden size {MODEL_CONFIG['fusion_hidden_dim']} with dropout {MODEL_CONFIG['fusion_dropout']}.",
                        "- Output heads: 1-day return regression, multi-day return regression, and 3-class direction classification.",
                    ],
                ),
                (
                    "Training Options",
                    [
                        "- current: sentiment branch enabled and cross-attention enabled.",
                        "- no-sentiment: sentiment branch disabled; model uses price, technical, market, and calendar features only.",
                        "- sentiment-no-cross-attention: sentiment branch enabled but cross-attention disabled; streams are fused by concatenation only.",
                    ],
                ),
            ],
        ),
        (
            "Architecture: Standalone Baseline Models",
            [
                (
                    "Shared Baseline Input",
                    [
                        "The standalone models use the same preprocessing pipeline, but they concatenate price/technical and sentiment features into one sequence tensor. They are therefore not OHLCV-only baselines; they are single-stream baselines using the engineered feature set.",
                    ],
                ),
                (
                    "LSTM Regressor",
                    [
                        "- Input projection: Linear -> LayerNorm -> ReLU -> Dropout.",
                        "- Sequence model: 2-layer bidirectional LSTM with hidden size 128 and dropout 0.2.",
                        "- Output: concatenates forward/backward sequence representations, then Dense -> ReLU -> Dropout -> Dense(1).",
                    ],
                ),
                (
                    "GRU Regressor",
                    [
                        "- Input projection: Linear -> LayerNorm -> ReLU -> Dropout.",
                        "- Sequence model: 2-layer bidirectional GRU with hidden size 128 and dropout 0.2.",
                        "- Output: concatenates forward/backward sequence representations, then Dense -> ReLU -> Dropout -> Dense(1).",
                    ],
                ),
                (
                    "Transformer Regressor",
                    [
                        "- Input projection to d_model = 128.",
                        "- Sinusoidal positional encoding.",
                        "- Transformer encoder with 2 layers, 4 attention heads, feed-forward size 256, and dropout 0.1.",
                        "- Output uses the final token embedding followed by Dense -> ReLU -> Dropout -> Dense(1).",
                    ],
                ),
                (
                    "XGBoost Regressor",
                    [
                        "- The 60-day sequence is flattened into a tabular feature vector.",
                        "- XGBRegressor settings: 400 estimators, max_depth 4, learning_rate 0.05, subsample 0.8, random_state 42, n_jobs 1.",
                        "- This baseline tests whether a tree-based tabular learner can compete with neural sequence models on engineered features.",
                    ],
                ),
            ],
        ),
        (
            "Evaluation Design and Retrained Results",
            [
                (
                    "Evaluation Design",
                    [
                        "The primary reported metric is MAE after reconstructing next-day prices from predicted returns. RMSE and MAPE are also reported for interpretability. Direction accuracy is secondary because the model is optimized mainly for regression.",
                        "The comparison is reasonable when framed as single-stream baselines versus a two-stream multimodal fusion architecture. It is not a comparison between OHLCV-only baselines and sentiment-aware multimodal modelling.",
                    ],
                ),
                (
                    "Saved Retrained Baseline Results",
                    format_baseline_results(),
                ),
                (
                    "Interpretation",
                    [
                        "The latest saved baseline results show LSTM, GRU, and Transformer tied on MAE at $9.70, while XGBoost is slightly worse at $10.22. The exact tie among three neural models should be treated cautiously and checked against training logs or prediction distributions.",
                        "If the multimodal fusion model underperforms these baselines, the result is still academically meaningful: the available sentiment signal may not add enough information to offset the additional model complexity.",
                    ],
                ),
            ],
        ),
        (
            "Dashboard and Deployment Notes",
            [
                (
                    "Streamlit Usage",
                    [
                        "Streamlit loads the active fusion model from models/best_model.pt and reconstructs the architecture using preprocessing_metadata.pkl.",
                        "Because XGBoost pickle loading caused native crashes with incompatible library versions, Streamlit skips live XGBoost artifact loading while still displaying saved XGBoost metrics from CSV.",
                    ],
                ),
                (
                    "Reproducibility Notes",
                    [
                        "- Use Python 3.11 or 3.12 with requirements-stable.txt for the Streamlit app.",
                        "- Use run_streamlit_safe.sh to disable Streamlit file watching and reduce native crash risk.",
                        "- Each new training run overwrites active artifacts in models/, so copy artifacts into experiment folders before training another mode.",
                    ],
                ),
            ],
        ),
        (
            "Recommendations and Conclusion",
            [
                (
                    "Recommended Next Steps",
                    [
                        "- Save fusion test metrics to a CSV after every training run so the report can compare fusion modes directly.",
                        "- Run the three fusion ablations: current, no-sentiment, and sentiment-no-cross-attention.",
                        "- Select best_model.pt by validation 1-day regression loss or validation MAE, because MAE is the primary evaluation metric.",
                        "- Add walk-forward validation for a stronger time-series evaluation.",
                        "- Replace synthetic/current RSS sentiment with a genuine historical news source if the project scope allows.",
                    ],
                ),
                (
                    "Conclusion",
                    [
                        "The current project setting is suitable for coursework because it includes a full ML workflow, defensible baselines, multimodal architecture, ablation options, and a clear scope statement. The most important caveat is that the sentiment signal is limited, so standalone models outperforming the multimodal model is a valid and explainable result rather than a project failure.",
                    ],
                ),
            ],
        ),
        (
            "References",
            [
                (
                    "Core References",
                    [
                        "- Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.",
                        "- Cho, K. et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, 2014.",
                        "- Vaswani, A. et al. Attention Is All You Need, 2017.",
                        "- Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System, 2016.",
                        "- Araci, D. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, 2019.",
                        "- Project source files: config.py, train.py, model_comparison.py, src/data, src/features, src/models, app/streamlit_app.py.",
                    ],
                )
            ],
        ),
    ]


def build_report():
    pages: list[Image.Image] = []
    add_title_page(pages)

    content = report_content()
    for title, sections in content[:4]:
        add_text_pages(pages, title, sections)

    add_image_page(
        pages,
        "Pipeline Architecture",
        ROOT / "pipeline_architecture.png",
        "Figure 1. End-to-end workflow from TSLA data and sentiment features through preprocessing, fusion, and prediction heads.",
    )

    for title, sections in content[4:7]:
        add_text_pages(pages, title, sections)

    add_image_page(
        pages,
        "Training History",
        ROOT / "training_history.png",
        "Figure 2. Saved training history visualisation for the multimodal project.",
    )
    add_image_page(
        pages,
        "Baseline Comparison Chart",
        ROOT / "models" / "model_comparison_chart.png",
        "Figure 3. Saved baseline comparison visualisation.",
    )

    for title, sections in content[7:]:
        add_text_pages(pages, title, sections)

    pages[0].save(OUTPUT, save_all=True, append_images=pages[1:], resolution=150.0)


if __name__ == "__main__":
    build_report()
    print(f"Report written to {OUTPUT}")
