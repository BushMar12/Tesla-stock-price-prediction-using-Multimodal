"""
Generate the pipeline architecture diagram.

Writes:
    pipeline_architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "pipeline_architecture.png"


COLORS = {
    "text": "#111827",
    "muted": "#475569",
    "arrow": "#1f2937",
    "blue": "#2f6fb7",
    "blue_bg": "#eef6ff",
    "green": "#2e8b57",
    "green_bg": "#f0faf4",
    "orange": "#ea6a16",
    "orange_bg": "#fff7ed",
    "purple": "#6a45c9",
    "purple_bg": "#f3efff",
    "red": "#cc2f2f",
    "red_bg": "#fff1f2",
    "teal": "#0f766e",
    "teal_bg": "#ecfeff",
    "gray": "#334155",
    "gray_bg": "#f8fafc",
}


def add_box(ax, xy, width, height, text, edge, face, fontsize=10.5, lw=1.4):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        zorder=3,
        linespacing=1.18,
    )
    return box


def add_section(ax, y, height, number, title, edge, face):
    section = FancyBboxPatch(
        (0.25, y),
        15.5,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=0,
    )
    ax.add_patch(section)

    circle = Circle((0.62, y + height / 2), 0.22, facecolor=edge, edgecolor=edge, zorder=1)
    ax.add_patch(circle)
    ax.text(0.62, y + height / 2, str(number), color="white", ha="center", va="center",
            fontsize=15, fontweight="bold", zorder=2)
    ax.text(1.0, y + height / 2, title, color=edge, ha="left", va="center",
            fontsize=14, fontweight="bold", linespacing=1.1, zorder=2)


def center_top(box):
    return (box.get_x() + box.get_width() / 2, box.get_y() + box.get_height())


def center_bottom(box):
    return (box.get_x() + box.get_width() / 2, box.get_y())


def center_left(box):
    return (box.get_x(), box.get_y() + box.get_height() / 2)


def center_right(box):
    return (box.get_x() + box.get_width(), box.get_y() + box.get_height() / 2)


def add_arrow(ax, start, end, color=None, style="-", rad=0.0, lw=1.45):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        linestyle=style,
        color=color or COLORS["arrow"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=4,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 12), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        8,
        11.72,
        "Tesla Stock Price Prediction Multimodal Pipeline",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        8,
        11.43,
        "Updated architecture: Transformer price encoder + multi-kernel 1D CNN sentiment encoder with cross-modal attention",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    add_section(ax, 9.86, 1.25, 1, "Data\nSources", COLORS["blue"], COLORS["blue_bg"])
    add_section(ax, 8.45, 1.25, 2, "Feature\nEngineering", COLORS["green"], COLORS["green_bg"])
    add_section(ax, 5.05, 3.15, 3, "Preprocessing\n& Splitting", COLORS["orange"], COLORS["orange_bg"])
    add_section(ax, 3.05, 1.75, 4, "Multimodal Fusion\nNeural Network", COLORS["purple"], COLORS["purple_bg"])
    add_section(ax, 0.42, 2.35, 5, "Model\nOutputs", COLORS["red"], COLORS["red_bg"])

    # Data sources
    yahoo = add_box(ax, (2.4, 10.2), 2.7, 0.62, "Yahoo Finance API\nTSLA OHLCV", COLORS["blue"], COLORS["blue_bg"])
    av = add_box(ax, (6.0, 10.2), 3.0, 0.62, "Alpha Vantage API\nTSLA News Sentiment", COLORS["blue"], COLORS["blue_bg"])
    rss = add_box(ax, (10.0, 10.2), 3.0, 0.62, "RSS Headlines\nFallback Current News", COLORS["blue"], COLORS["blue_bg"])

    # Feature engineering
    tech = add_box(ax, (2.95, 8.75), 2.85, 0.62, "Technical Indicators\nSMA, RSI, Bollinger Bands", COLORS["green"], COLORS["green_bg"])
    calendar = add_box(ax, (6.1, 8.75), 2.45, 0.62, "Time Features\nMonth, Quarter, Day", COLORS["green"], COLORS["green_bg"])
    cache = add_box(ax, (9.2, 9.02), 2.9, 0.52, "Raw Article Cache\nalpha_vantage_tsla_news_raw.csv", COLORS["teal"], COLORS["teal_bg"], fontsize=9.4)
    daily = add_box(ax, (9.2, 8.38), 2.9, 0.52, "Daily Aggregation\nmean scores + news_count", COLORS["green"], COLORS["green_bg"], fontsize=9.4)
    vader = add_box(ax, (12.25, 8.75), 2.45, 0.62, "VADER / FinBERT Scoring\nRSS fallback sentiment", COLORS["green"], COLORS["green_bg"], fontsize=9.4)
    aligned_sent = add_box(ax, (11.45, 7.66), 3.0, 0.58, "Aligned Sentiment Features\ncompound, pos, neg, neutral, lags", COLORS["green"], COLORS["green_bg"], fontsize=9.2)

    # Preprocessing
    merge = add_box(ax, (5.35, 7.34), 4.7, 0.58, "Align by Trading Date\nMerge price, indicators, time, sentiment", COLORS["orange"], COLORS["orange_bg"])
    target = add_box(ax, (5.35, 6.65), 4.7, 0.52, "Calculate Target\nNext-day return only", COLORS["orange"], COLORS["orange_bg"])
    scale = add_box(ax, (5.35, 5.95), 4.7, 0.52, "Fit Scalers on TRAIN Set\nStandardScaler / MinMaxScaler", COLORS["orange"], COLORS["orange_bg"])
    window = add_box(ax, (5.35, 5.25), 4.7, 0.52, "Sliding Window\nSequence Length = 60 Trading Days", COLORS["orange"], COLORS["orange_bg"])
    price_tensor = add_box(ax, (4.45, 4.5), 2.45, 0.48, "Price Tensor\nBatch, 60, 11 features", COLORS["gray"], COLORS["gray_bg"], fontsize=9.4)
    sent_tensor = add_box(ax, (9.4, 4.5), 2.8, 0.48, "Sentiment Tensor\nBatch, 60, 8 features", COLORS["gray"], COLORS["gray_bg"], fontsize=9.4)

    # Neural network
    ts_encoder = add_box(ax, (4.15, 3.78), 3.0, 0.52, "Time-Series Encoder\nTransformer + Attention Pooling", COLORS["purple"], COLORS["purple_bg"], fontsize=9.6)
    sent_encoder = add_box(ax, (9.25, 3.78), 3.25, 0.52, "Sentiment Encoder\nMulti-Kernel 1D CNN + Mean Pooling", COLORS["purple"], COLORS["purple_bg"], fontsize=9.6)
    cross = add_box(ax, (6.05, 3.08), 3.9, 0.52, "Optional Cross-Modal Attention\n4-Head Multi-Head Attention", COLORS["purple"], COLORS["purple_bg"], fontsize=9.5)

    # Outputs
    fusion = add_box(ax, (6.05, 2.36), 3.9, 0.52, "Deep Fusion MLP\nDense Layers + Dropout", COLORS["purple"], COLORS["purple_bg"], fontsize=9.5)
    reg = add_box(ax, (6.4, 1.64), 3.2, 0.52, "Regression Head\nNext-day return", COLORS["red"], COLORS["red_bg"], fontsize=9.5)
    eval_box = add_box(ax, (5.55, 0.8), 4.9, 0.58, "Next-Day Price Reconstruction & Evaluation\nInverse-scale return, compute MAE / RMSE / MAPE", COLORS["red"], COLORS["red_bg"], fontsize=9.2)

    # Arrows: data and feature engineering
    add_arrow(ax, center_bottom(yahoo), center_top(tech))
    add_arrow(ax, center_bottom(yahoo), center_top(calendar), rad=0.04)
    add_arrow(ax, center_bottom(av), center_top(cache))
    add_arrow(ax, center_bottom(rss), center_top(vader))
    add_arrow(ax, center_bottom(cache), center_top(daily))
    add_arrow(ax, center_bottom(daily), center_top(aligned_sent))
    add_arrow(ax, center_bottom(vader), center_right(aligned_sent))

    # Arrows into merge
    add_arrow(ax, center_bottom(tech), center_left(merge), rad=0.04)
    add_arrow(ax, center_bottom(calendar), center_top(merge))
    add_arrow(ax, center_left(aligned_sent), center_right(merge), rad=0.0)
    add_arrow(ax, center_bottom(merge), center_top(target))
    add_arrow(ax, center_bottom(target), center_top(scale))
    add_arrow(ax, center_bottom(scale), center_top(window))
    add_arrow(ax, center_bottom(window), center_top(price_tensor), rad=-0.05)
    add_arrow(ax, center_bottom(window), center_top(sent_tensor), rad=0.05)

    # Model arrows
    add_arrow(ax, center_bottom(price_tensor), center_top(ts_encoder))
    add_arrow(ax, center_bottom(sent_tensor), center_top(sent_encoder))
    add_arrow(ax, center_bottom(ts_encoder), center_left(cross))
    add_arrow(ax, center_bottom(sent_encoder), center_right(cross))
    add_arrow(ax, center_bottom(cross), center_top(fusion))
    add_arrow(ax, center_bottom(ts_encoder), center_left(fusion), color=COLORS["purple"], style="--", rad=0.0, lw=1.25)
    ax.text(4.62, 2.82, "no-cross-attention option", color=COLORS["purple"], fontsize=9.0, style="italic")
    add_arrow(ax, center_bottom(fusion), center_top(reg))
    add_arrow(ax, center_bottom(reg), center_top(eval_box))

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
