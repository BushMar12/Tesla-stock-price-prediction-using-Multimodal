"""
Evaluation utilities for Tesla stock price prediction.

Provides standardised metric computation and publication-ready
visualisation functions used by the experiment runner.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of regression and directional metrics.

    Args:
        actual: Ground-truth prices (or returns).
        predicted: Model-predicted prices (or returns).

    Returns:
        Dict with MSE, RMSE, MAE, MAPE (%), R², and Directional Accuracy (%).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    # MAPE — guard against zero actuals
    nonzero = actual != 0
    if nonzero.any():
        mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
    else:
        mape = float("nan")

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Directional accuracy — fraction of days where predicted direction
    # matches actual direction (using day-over-day changes)
    if len(actual) > 1:
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = float("nan")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Directional_Accuracy": dir_acc,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (shared style)
# ---------------------------------------------------------------------------

_STYLE_DEFAULTS = dict(
    figure_facecolor="#fafafa",
    axes_facecolor="#ffffff",
    grid_alpha=0.25,
    title_size=14,
    label_size=11,
    dpi=180,
)

_MODEL_COLOURS = {
    "LSTM": "#2563eb",
    "GRU": "#16a34a",
    "Transformer": "#dc2626",
    "XGBoost": "#9333ea",
}


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent styling to a matplotlib Axes."""
    ax.set_title(title, fontsize=_STYLE_DEFAULTS["title_size"], fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=_STYLE_DEFAULTS["label_size"])
    ax.set_ylabel(ylabel, fontsize=_STYLE_DEFAULTS["label_size"])
    ax.grid(True, alpha=_STYLE_DEFAULTS["grid_alpha"], linestyle="--")
    ax.tick_params(labelsize=9)


def _save_fig(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_STYLE_DEFAULTS["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved plot → {path}")


# ---------------------------------------------------------------------------
# Individual model plots
# ---------------------------------------------------------------------------

def plot_predictions_vs_actual(
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    config_name: str,
    save_path: Path,
):
    """Line chart of predicted vs actual prices on the test set."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_STYLE_DEFAULTS["figure_facecolor"])
    ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

    colour = _MODEL_COLOURS.get(model_name, "#333333")
    ax.plot(dates, actual, color="#333333", linewidth=1.8, label="Actual Price", zorder=3)
    ax.plot(dates, predicted, color=colour, linewidth=1.4, linestyle="--", alpha=0.85,
            label=f"{model_name} Predicted", zorder=4)
    ax.fill_between(dates, actual, predicted, alpha=0.10, color=colour)

    _style_ax(ax,
              title=f"{model_name} — Predicted vs Actual Price [{config_name}]",
              xlabel="Date", ylabel="Price (USD)")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    fig.autofmt_xdate(rotation=30)
    _save_fig(fig, save_path)


def plot_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    config_name: str,
    save_path: Path,
):
    """Scatter plot of predicted vs actual with perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_STYLE_DEFAULTS["figure_facecolor"])
    ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

    colour = _MODEL_COLOURS.get(model_name, "#333333")
    ax.scatter(actual, predicted, alpha=0.5, s=18, c=colour, edgecolors="white", linewidths=0.3)

    lo = min(actual.min(), predicted.min()) * 0.95
    hi = max(actual.max(), predicted.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="Perfect prediction")

    _style_ax(ax,
              title=f"{model_name} — Scatter [{config_name}]",
              xlabel="Actual Price (USD)", ylabel="Predicted Price (USD)")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    _save_fig(fig, save_path)


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    config_name: str,
    save_path: Path,
):
    """Histogram of prediction residuals (predicted − actual)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=_STYLE_DEFAULTS["figure_facecolor"])
    ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

    residuals = predicted - actual
    colour = _MODEL_COLOURS.get(model_name, "#333333")
    ax.hist(residuals, bins=40, color=colour, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(np.mean(residuals), color="red", linestyle="-", linewidth=1.2, alpha=0.8,
               label=f"Mean = ${np.mean(residuals):.2f}")

    _style_ax(ax,
              title=f"{model_name} — Residual Distribution [{config_name}]",
              xlabel="Residual (USD)", ylabel="Frequency")
    ax.legend(fontsize=9)
    _save_fig(fig, save_path)


def plot_training_curves(
    history: dict,
    model_name: str,
    config_name: str,
    save_path: Path,
):
    """Training and validation loss / MAE curves for a single model."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_STYLE_DEFAULTS["figure_facecolor"])
    ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

    colour = _MODEL_COLOURS.get(model_name, "#333333")

    if model_name == "XGBoost":
        train_mae = history.get("train_mae", [])
        val_mae = history.get("val_mae", [])
        if train_mae:
            ax.plot(range(1, len(train_mae) + 1), train_mae, color=colour, linewidth=1.5,
                    label="Train MAE (scaled return)")
        if val_mae:
            ax.plot(range(1, len(val_mae) + 1), val_mae, color=colour, linewidth=1.5,
                    linestyle="--", label="Val MAE (scaled return)")
            best_ep = int(np.argmin(val_mae)) + 1
            ax.axvline(best_ep, color="green", linestyle=":", alpha=0.7, label=f"Best epoch {best_ep}")
        _style_ax(ax, title=f"{model_name} — Training Curve [{config_name}]",
                  xlabel="Boosting Round", ylabel="MAE (scaled return)")
    else:
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        val_mae = history.get("val_mae", [])
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, color=colour, linewidth=1.5, label="Train Loss")
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, color=colour, linewidth=1.5,
                    linestyle="--", alpha=0.7, label="Val Loss")
        _style_ax(ax, title=f"{model_name} — Training Curve [{config_name}]",
                  xlabel="Epoch", ylabel="Loss (SmoothL1)")

        if val_mae:
            ax2 = ax.twinx()
            ax2.plot(range(1, len(val_mae) + 1), val_mae, color="red", linewidth=1.5,
                     label="Val MAE ($)")
            ax2.set_ylabel("Val MAE ($)", fontsize=_STYLE_DEFAULTS["label_size"], color="red")
            ax2.tick_params(axis="y", labelcolor="red", labelsize=9)
            best_ep = int(np.argmin(val_mae)) + 1
            ax.axvline(best_ep, color="green", linestyle=":", alpha=0.7, label=f"Best epoch {best_ep}")
            # Merge legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.9)
        else:
            ax.legend(fontsize=9, framealpha=0.9)

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Cross-model comparison plots
# ---------------------------------------------------------------------------

def plot_comparison_bar_chart(
    results_df: pd.DataFrame,
    save_path: Path,
):
    """Grouped bar chart comparing metrics across all model × config combos.

    Expects columns: Configuration, Model, RMSE, MAE, MAPE, R2, Directional_Accuracy.
    """
    metrics = ["RMSE", "MAE", "MAPE", "R2", "Directional_Accuracy"]
    metric_labels = {
        "RMSE": "RMSE ($)",
        "MAE": "MAE ($)",
        "MAPE": "MAPE (%)",
        "R2": "R²",
        "Directional_Accuracy": "Dir. Accuracy (%)",
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6),
                             facecolor=_STYLE_DEFAULTS["figure_facecolor"])

    # Create a combined label for each bar
    results_df = results_df.copy()
    results_df["label"] = results_df["Model"] + "\n(" + results_df["Configuration"].str.replace("_", "+") + ")"

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

        labels = results_df["label"].values
        values = results_df[metric].values
        colours = [_MODEL_COLOURS.get(m, "#888888") for m in results_df["Model"]]

        bars = ax.bar(range(len(labels)), values, color=colours, alpha=0.85,
                      edgecolor="white", linewidth=0.8, width=0.7)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if metric in ("MAPE", "Directional_Accuracy") else (
                f"{val:.3f}" if metric == "R2" else f"${val:.1f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    fmt, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        _style_ax(ax, title=metric_labels.get(metric, metric))

    fig.suptitle("Model Comparison — All Configurations", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_all_predictions_overlay(
    dates: np.ndarray,
    actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    config_name: str,
    save_path: Path,
):
    """Overlay all model predictions on one chart for a given configuration."""
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=_STYLE_DEFAULTS["figure_facecolor"])
    ax.set_facecolor(_STYLE_DEFAULTS["axes_facecolor"])

    ax.plot(dates, actual, color="#111111", linewidth=2.2, label="Actual Price", zorder=5)

    for model_name, preds in predictions_dict.items():
        colour = _MODEL_COLOURS.get(model_name, "#888888")
        ax.plot(dates, preds, color=colour, linewidth=1.2, linestyle="--", alpha=0.8,
                label=f"{model_name}", zorder=4)

    _style_ax(ax,
              title=f"All Models — Predicted vs Actual [{config_name}]",
              xlabel="Date", ylabel="Price (USD)")
    ax.legend(fontsize=10, framealpha=0.9, ncol=3)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    fig.autofmt_xdate(rotation=30)
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_comparison_table(results_df: pd.DataFrame):
    """Print a nicely formatted comparison table to stdout."""
    display = results_df.copy()
    display["RMSE"] = display["RMSE"].map(lambda v: f"${v:.2f}")
    display["MAE"] = display["MAE"].map(lambda v: f"${v:.2f}")
    display["MAPE"] = display["MAPE"].map(lambda v: f"{v:.2f}%")
    display["R2"] = display["R2"].map(lambda v: f"{v:.4f}")
    display["Directional_Accuracy"] = display["Directional_Accuracy"].map(lambda v: f"{v:.1f}%")

    col_order = ["Configuration", "Model", "MAE", "RMSE", "MAPE", "R2", "Directional_Accuracy"]
    existing = [c for c in col_order if c in display.columns]
    print(display[existing].to_string(index=False))
