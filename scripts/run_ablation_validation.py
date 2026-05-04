"""
Run controlled multimodal ablations and save report-ready validation artifacts.

The runner trains the three modes supported by train.py, preserves each run's
model/scaler metadata, evaluates the copied best checkpoint on the same test
split, and writes:

    models/ablation_results.csv
    models/ablation_comparison.png
    models/ablation_train_val_curves.png
    models/ablation_verdict.txt

By default it uses cached stock data and synthetic sentiment so the experiment
is deterministic and does not depend on live downloads. Use
``--sentiment-source alpha_vantage`` after the Alpha Vantage daily sentiment
cache is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "tesla_ablation_matplotlib"),
)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import END_DATE, MODELS_DIR, RAW_DATA_DIR, START_DATE, STOCK_SYMBOL
from src.data.preprocessing import DataPreprocessor
from src.data.sentiment_data import fetch_sentiment_data
from src.data.stock_data import fetch_stock_data, load_stock_data
from src.features.technical import calculate_all_indicators
from src.models.fusion import create_model
from src.models.trainer import StockDataset, Trainer, train_model


TRAINING_MODES = {
    "no-sentiment": {
        "label": "No sentiment",
        "use_sentiment": False,
        "use_cross_attention": False,
    },
    "sentiment-no-cross-attention": {
        "label": "Sentiment, no attention",
        "use_sentiment": True,
        "use_cross_attention": False,
    },
    "current": {
        "label": "Sentiment + attention",
        "use_sentiment": True,
        "use_cross_attention": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare multimodal ablation modes."
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["synthetic", "cached", "rss", "alpha_vantage"],
        default="synthetic",
        help="Sentiment source used by sentiment-enabled modes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.TRAINING_CONFIG["epochs"],
        help="Epochs per ablation run.",
    )
    parser.add_argument(
        "--refresh-stock",
        action="store_true",
        help="Fetch stock data from Yahoo instead of using cached data/raw CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training initialization and loaders.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mode_slug(mode: str, sentiment_source: str) -> str:
    return f"ablation_{sentiment_source}_{mode.replace('-', '_')}"


def load_sentiment(stock_df: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "cached":
        sentiment_path = RAW_DATA_DIR / "sentiment_data.csv"
        if not sentiment_path.exists():
            raise FileNotFoundError(f"Cached sentiment not found: {sentiment_path}")
        sentiment_df = pd.read_csv(sentiment_path)
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        print(f"Loaded cached sentiment from {sentiment_path}")
        return sentiment_df

    return fetch_sentiment_data(
        stock_df,
        use_real_data=source != "synthetic",
        source=source,
        save=True,
    )


def build_splits(
    stock_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    mode_config: dict,
) -> tuple[dict, DataPreprocessor]:
    if mode_config["use_sentiment"]:
        mode_sentiment_df = sentiment_df
    else:
        mode_sentiment_df = stock_df[["date"]].copy()

    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, mode_sentiment_df, indicators_df)
    return splits, preprocessor


def copy_artifact(source_name: str, dest_name: str) -> Path | None:
    source = MODELS_DIR / source_name
    if not source.exists():
        return None
    dest = MODELS_DIR / dest_name
    shutil.copy2(source, dest)
    return dest


def copy_mode_artifacts(slug: str) -> dict[str, str]:
    artifact_map = {
        "best_model": ("best_model.pt", f"{slug}__best.pt"),
        "final_model": ("final_model.pt", f"{slug}__final.pt"),
        "metadata": ("preprocessing_metadata.pkl", f"{slug}__metadata.pkl"),
        "feature_scaler": ("feature_scaler.pkl", f"{slug}__feature_scaler.pkl"),
        "return_scaler": ("return_scaler.pkl", f"{slug}__return_scaler.pkl"),
    }
    copied = {}
    for key, (source, dest) in artifact_map.items():
        copied_path = copy_artifact(source, dest)
        if copied_path is not None:
            copied[key] = str(copied_path)
    return copied


def load_history(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("history", {})


def count_parameters(model_path: Path, metadata: dict) -> int:
    model = create_model(
        ts_input_size=metadata["n_price_features"],
        sentiment_input_size=metadata["n_sentiment_features"],
        use_cross_attention=metadata.get("use_cross_attention", True),
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return sum(p.numel() for p in model.parameters())


def evaluate_checkpoint(
    model_path: Path,
    metadata: dict,
    splits: dict,
    return_scaler,
) -> dict:
    model = create_model(
        ts_input_size=metadata["n_price_features"],
        sentiment_input_size=metadata["n_sentiment_features"],
        use_cross_attention=metadata.get("use_cross_attention", True),
    )
    trainer = Trainer(model)
    checkpoint = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = StockDataset(
        splits["test"]["X_price"],
        splits["test"]["X_sentiment"],
        splits["test"]["y_reg"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAINING_CONFIG["batch_size"],
        shuffle=False,
    )
    metrics = trainer.evaluate(
        test_loader,
        return_scaler=return_scaler,
        close_prices=splits["test"].get("close_prices"),
    )

    return metrics


def numeric_baselines() -> pd.DataFrame:
    path = MODELS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    def parse_money(value):
        return float(str(value).replace("$", "").replace(",", ""))

    parsed = pd.DataFrame(
        {
            "model": df["Model"],
            "rmse": df["RMSE"].map(parse_money),
            "mae": df["MAE"].map(parse_money),
        }
    )
    return parsed


def save_comparison_plot(results: pd.DataFrame) -> None:
    baselines = numeric_baselines()
    metrics = [
        ("rmse", "RMSE ($)", "lower"),
        ("mae", "MAE ($)", "lower"),
        ("mape", "MAPE (%)", "lower"),
    ]
    colors = ["#2563eb", "#16a34a", "#9333ea"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = results["label"].tolist()

    for ax, (metric, title, direction) in zip(axes, metrics):
        values = results[metric].to_numpy()
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        if not baselines.empty:
            for baseline_name in ["Transformer", "XGBoost", "LSTM", "GRU"]:
                row = baselines[baselines["model"] == baseline_name]
                if row.empty:
                    continue
                if metric not in row.columns:
                    continue
                value = float(row.iloc[0][metric])
                ax.axhline(
                    value,
                    linestyle="--",
                    linewidth=1.5,
                    label=baseline_name,
                )
        ax.text(
            0.02,
            0.95,
            f"{direction} is better",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#475569",
        )

    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_for_legend, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(MODELS_DIR / "ablation_comparison.png", dpi=220)
    plt.close(fig)


def save_curve_plot(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)

    for ax, (_, row) in zip(axes, results.iterrows()):
        final_path = Path(row["final_model_path"])
        history = load_history(final_path)
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        epochs = np.arange(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label="Train", color="#2563eb")
        ax.plot(epochs, val_loss, label="Validation", color="#f97316")
        if val_loss:
            best_epoch = int(np.argmin(val_loss)) + 1
            ax.axvline(best_epoch, color="#16a34a", linestyle="--", linewidth=1.2)
            ax.scatter([best_epoch], [min(val_loss)], color="#16a34a", zorder=3)
        ax.set_title(row["label"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "ablation_train_val_curves.png", dpi=220)
    plt.close(fig)


def verdict(results: pd.DataFrame) -> str:
    current = results[results["mode"] == "current"].iloc[0]
    best_mae = results.loc[results["mae"].idxmin()]
    current_gap_ratio = (
        current["train_val_gap"] / max(abs(current["train_loss_final"]), 1e-8)
    )

    lines = [
        "Multimodal validation verdict",
        "============================",
        f"Best MAE mode: {best_mae['mode']} (${best_mae['mae']:.2f})",
        f"Current-mode final train/val gap ratio: {current_gap_ratio:.2%}",
    ]

    if current_gap_ratio > 0.30 and current["mae"] >= best_mae["mae"]:
        decision = (
            "Verdict: fix or simplify sentiment fusion before adding model depth. "
            "The current mode shows a large generalization gap and does not lead MAE."
        )
    elif current["mode"] != best_mae["mode"]:
        decision = (
            "Verdict: keep the ablation evidence and validate again with real "
            "Alpha Vantage sentiment before expanding the architecture."
        )
    else:
        decision = (
            "Verdict: current multimodal fusion is helping on MAE; additional "
            "capacity may be worth testing after regularization checks."
        )
    lines.append(decision)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config.TRAINING_CONFIG["epochs"] = args.epochs

    if args.refresh_stock:
        stock_df = fetch_stock_data(
            symbol=STOCK_SYMBOL,
            start_date=START_DATE,
            end_date=END_DATE,
            save=True,
        )
    else:
        stock_df = load_stock_data(symbol=STOCK_SYMBOL)

    sentiment_df = load_sentiment(stock_df, args.sentiment_source)
    indicators_df = calculate_all_indicators(stock_df)

    rows = []
    for mode, mode_config in TRAINING_MODES.items():
        print("\n" + "=" * 72)
        print(f"Running ablation mode: {mode}")
        print("=" * 72)
        set_seed(args.seed)

        splits, preprocessor = build_splits(
            stock_df,
            indicators_df,
            sentiment_df,
            mode_config,
        )

        train_model(
            splits,
            return_scaler=preprocessor.return_scaler,
            use_cross_attention=mode_config["use_cross_attention"],
            training_mode=mode,
            use_sentiment=mode_config["use_sentiment"],
        )

        slug = mode_slug(mode, args.sentiment_source)
        copied = copy_mode_artifacts(slug)
        metadata = joblib.load(MODELS_DIR / f"{slug}__metadata.pkl")
        best_model_path = MODELS_DIR / f"{slug}__best.pt"
        final_model_path = MODELS_DIR / f"{slug}__final.pt"
        history = load_history(final_model_path)
        metrics = evaluate_checkpoint(
            best_model_path,
            metadata,
            splits,
            preprocessor.return_scaler,
        )
        params = count_parameters(best_model_path, metadata)

        train_loss_final = float(history["train_loss"][-1])
        val_loss_final = float(history["val_loss"][-1])
        row = {
            "mode": mode,
            "label": mode_config["label"],
            "sentiment_source": args.sentiment_source,
            "params": params,
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "mape": float(metrics["mape"]),
            "train_loss_final": train_loss_final,
            "val_loss_final": val_loss_final,
            "train_val_gap": val_loss_final - train_loss_final,
            "best_epoch": int(np.argmin(history["val_loss"]) + 1),
            "epochs": len(history["train_loss"]),
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
        }
        rows.append(row)

        with (MODELS_DIR / f"{slug}__metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row | {"copied_artifacts": copied}, f, indent=2)

    results = pd.DataFrame(rows)
    results_path = MODELS_DIR / "ablation_results.csv"
    results.to_csv(results_path, index=False)
    save_comparison_plot(results)
    save_curve_plot(results)

    verdict_text = verdict(results)
    verdict_path = MODELS_DIR / "ablation_verdict.txt"
    verdict_path.write_text(verdict_text + "\n", encoding="utf-8")

    print("\nAblation results:")
    print(
        results[
            [
                "mode",
                "params",
                "rmse",
                "mae",
                "mape",
                "train_loss_final",
                "val_loss_final",
                "train_val_gap",
                "best_epoch",
            ]
        ].to_string(index=False)
    )
    print("\n" + verdict_text)
    print("\nArtifacts written:")
    print(f"  {results_path}")
    print(f"  {MODELS_DIR / 'ablation_comparison.png'}")
    print(f"  {MODELS_DIR / 'ablation_train_val_curves.png'}")
    print(f"  {verdict_path}")


if __name__ == "__main__":
    main()
