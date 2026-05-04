"""
Generate SHAP feature-importance artifacts for the trained fusion model.

This script explains the trained ``MultimodalFusionModel`` with
``shap.GradientExplainer`` and writes report-ready artifacts:

    models/shap_feature_importance.csv
    models/shap_feature_importance.png
    models/shap_temporal_heatmap.png

Prerequisites:
    pip install shap matplotlib seaborn

By default the script uses cached CSV data under data/raw/ so the analysis is
reproducible and does not depend on live downloads. Use ``--refresh-data`` to
rebuild inputs through the same fetch path used by train.py.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "tesla_shap_matplotlib"),
)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import END_DATE, MODELS_DIR, MODEL_VERSION, RAW_DATA_DIR, START_DATE, STOCK_SYMBOL
from src.data.preprocessing import DataPreprocessor
from src.data.sentiment_data import fetch_sentiment_data
from src.data.stock_data import fetch_stock_data, load_stock_data
from src.features.technical import calculate_all_indicators
from src.models.fusion import create_model


class SHAPWrapper(nn.Module):
    """Expose one tensor output from the fusion model for SHAP."""

    def __init__(self, model: nn.Module, target: str):
        super().__init__()
        self.model = model
        self.target = target

    def forward(
        self,
        ts_input: torch.Tensor,
        sentiment_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(ts_input, sentiment_input)
        if self.target == "regression":
            return outputs["regression"].unsqueeze(-1)
        raise ValueError(f"Unsupported SHAP target: {self.target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank fusion-model input features with SHAP GradientExplainer."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "best_model.pt",
        help="Path to a trained fusion-model checkpoint.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of train sequences used as SHAP background.",
    )
    parser.add_argument(
        "--explain-size",
        type=int,
        default=200,
        help="Number of test sequences to explain.",
    )
    parser.add_argument(
        "--target",
        choices=["regression"],
        default="regression",
        help="Model output to explain. The current next-day-only model has one regression head.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Path to preprocessing metadata. Defaults to the checkpoint's matching metadata when available.",
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["cached", "synthetic", "rss", "alpha_vantage"],
        default="cached",
        help="Sentiment source. cached reads data/raw/sentiment_data.csv.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch stock/sentiment inputs instead of using cached CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for background and explanation sample selection.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cached_sentiment() -> pd.DataFrame:
    sentiment_path = RAW_DATA_DIR / "sentiment_data.csv"
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"Cached sentiment not found at {sentiment_path}. "
            "Run with --sentiment-source synthetic or --refresh-data."
        )
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    return sentiment_df


def default_metadata_path(model_path: Path) -> Path:
    """Use copied ablation metadata when explaining a copied ablation checkpoint."""
    stem = model_path.stem
    if stem.endswith("__best") or stem.endswith("__final"):
        slug = stem.rsplit("__", 1)[0]
        candidate = MODELS_DIR / f"{slug}__metadata.pkl"
        if candidate.exists():
            return candidate
    return MODELS_DIR / "preprocessing_metadata.pkl"


def load_model_and_metadata(
    model_path: Path,
    metadata_path: Path | None,
) -> tuple[nn.Module, dict]:
    metadata_path = metadata_path or default_metadata_path(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Preprocessing metadata not found: {metadata_path}")

    metadata = joblib.load(metadata_path)
    model = create_model(
        ts_input_size=metadata["n_price_features"],
        sentiment_input_size=metadata["n_sentiment_features"],
        use_cross_attention=metadata.get("use_cross_attention", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    version = checkpoint.get("model_version")
    if version != MODEL_VERSION:
        raise RuntimeError(
            f"Checkpoint at {model_path} has model_version={version!r}, "
            f"expected {MODEL_VERSION!r}. Retrain required."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


def build_splits(args: argparse.Namespace, trained_metadata: dict) -> tuple[dict, dict]:
    if args.refresh_data:
        stock_df = fetch_stock_data(
            symbol=STOCK_SYMBOL,
            start_date=START_DATE,
            end_date=END_DATE,
            save=True,
        )
    else:
        stock_df = load_stock_data(symbol=STOCK_SYMBOL)

    if not trained_metadata.get("use_sentiment", True):
        sentiment_df = stock_df[["date"]].copy()
    elif args.sentiment_source == "cached":
        sentiment_df = load_cached_sentiment()
    else:
        sentiment_df = fetch_sentiment_data(
            stock_df,
            use_real_data=args.sentiment_source != "synthetic",
            source=args.sentiment_source,
            save=True,
        )

    indicators_df = calculate_all_indicators(stock_df)
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    metadata = {
        "feature_columns": preprocessor.feature_columns,
        "sentiment_columns": preprocessor.sentiment_columns,
        "n_price_features": len(preprocessor.feature_columns),
        "n_sentiment_features": len(preprocessor.sentiment_columns),
        "sequence_length": preprocessor.sequence_length,
    }
    return splits, metadata


def choose_samples(
    split: dict,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    available = len(split["X_price"])
    if available == 0:
        raise ValueError("Cannot sample from an empty split.")
    sample_size = min(size, available)
    indices = np.sort(rng.choice(available, size=sample_size, replace=False))
    X_sentiment = split["X_sentiment"]
    if X_sentiment is not None:
        X_sentiment = X_sentiment[indices]
    return split["X_price"][indices], X_sentiment


def to_tensor(array: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if array is None:
        return None
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def normalize_shap_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise ValueError(f"Expected SHAP values with 3 dimensions, got {values.shape}")
    return values


def split_shap_values(shap_values) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(shap_values, np.ndarray):
        return normalize_shap_array(shap_values), None
    if isinstance(shap_values, list) and len(shap_values) == 1:
        return normalize_shap_array(shap_values[0]), None
    if isinstance(shap_values, tuple) and len(shap_values) == 1:
        return normalize_shap_array(shap_values[0]), None
    if isinstance(shap_values, list) and len(shap_values) == 2:
        ts_values, sent_values = shap_values
    elif isinstance(shap_values, tuple) and len(shap_values) == 2:
        ts_values, sent_values = shap_values
    else:
        raise ValueError(
            "Expected SHAP values for two model inputs: time-series and sentiment."
        )
    return normalize_shap_array(ts_values), normalize_shap_array(sent_values)


def feature_modality(feature: str, stream: str) -> str:
    lower = feature.lower()
    if "sentiment" in lower or feature == "news_count":
        return "sentiment"
    if feature in {"spy_close", "spy_return", "tsla_vs_spy", "vix"}:
        return "market"
    if feature in {"open", "high", "low", "close", "volume", "returns", "log_returns"}:
        return "price"
    if stream == "sentiment":
        return "sentiment"
    return "technical"


def build_importance_frame(
    shap_ts: np.ndarray,
    shap_sent: np.ndarray | None,
    feature_columns: list[str],
    sentiment_columns: list[str],
) -> pd.DataFrame:
    ts_importance = np.abs(shap_ts).mean(axis=(0, 1))
    sent_importance = (
        np.abs(shap_sent).mean(axis=(0, 1)) if shap_sent is not None else np.array([])
    )

    rows = []
    for feature, value in zip(feature_columns, ts_importance):
        rows.append(
            {
                "feature": feature,
                "stream": "time_series",
                "modality": feature_modality(feature, "time_series"),
                "mean_abs_shap": float(value),
            }
        )
    for feature, value in zip(sentiment_columns, sent_importance):
        rows.append(
            {
                "feature": feature,
                "stream": "sentiment",
                "modality": feature_modality(feature, "sentiment"),
                "mean_abs_shap": float(value),
            }
        )

    importance = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    return importance


def save_bar_chart(importance: pd.DataFrame, output_path: Path, top_n: int = 25) -> None:
    top = importance.head(top_n).iloc[::-1]
    palette = {
        "price": "#2563eb",
        "technical": "#16a34a",
        "market": "#f97316",
        "sentiment": "#9333ea",
    }

    plt.figure(figsize=(10, 8))
    colors = [palette.get(modality, "#64748b") for modality in top["modality"]]
    plt.barh(top["feature"], top["mean_abs_shap"], color=colors)
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Top {len(top)} SHAP Feature Importances")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=label.title(),
                   markerfacecolor=color, markersize=10)
        for label, color in palette.items()
        if label in set(top["modality"])
    ]
    plt.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_temporal_heatmap(
    shap_ts: np.ndarray,
    shap_sent: np.ndarray | None,
    importance: pd.DataFrame,
    feature_columns: list[str],
    sentiment_columns: list[str],
    output_path: Path,
    top_n: int = 8,
) -> None:
    all_feature_names = feature_columns + sentiment_columns
    temporal_arrays = [np.abs(shap_ts).mean(axis=0)]
    if shap_sent is not None:
        temporal_arrays.append(np.abs(shap_sent).mean(axis=0))
    all_temporal = np.concatenate(temporal_arrays, axis=1)

    top_features = importance.head(top_n)["feature"].tolist()
    row_indices = [all_feature_names.index(feature) for feature in top_features]
    heatmap_data = all_temporal[:, row_indices].T

    lag_labels = [f"t-{all_temporal.shape[0] - i}" for i in range(all_temporal.shape[0])]
    xtick_step = max(1, all_temporal.shape[0] // 10)

    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        yticklabels=top_features,
        xticklabels=lag_labels,
        cbar_kws={"label": "Mean absolute SHAP value"},
    )
    ax.set_xticks(np.arange(0.5, len(lag_labels), xtick_step))
    ax.set_xticklabels(lag_labels[::xtick_step], rotation=45, ha="right")
    plt.xlabel("Sequence timestep")
    plt.ylabel("Feature")
    plt.title(f"Temporal SHAP Heatmap for Top {len(top_features)} Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Loading trained model...")
    model, trained_metadata = load_model_and_metadata(args.model_path, args.metadata_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Building preprocessing splits...")
    splits, metadata = build_splits(args, trained_metadata)
    if metadata["n_price_features"] != trained_metadata["n_price_features"]:
        raise ValueError(
            "Prepared price feature count does not match trained model metadata: "
            f"{metadata['n_price_features']} != {trained_metadata['n_price_features']}"
        )
    if metadata["n_sentiment_features"] != trained_metadata["n_sentiment_features"]:
        raise ValueError(
            "Prepared sentiment feature count does not match trained model metadata: "
            f"{metadata['n_sentiment_features']} != {trained_metadata['n_sentiment_features']}"
        )

    ts_bg_np, sent_bg_np = choose_samples(splits["train"], args.background_size, rng)
    ts_explain_np, sent_explain_np = choose_samples(splits["test"], args.explain_size, rng)

    ts_bg = to_tensor(ts_bg_np, device)
    sent_bg = to_tensor(sent_bg_np, device)
    ts_explain = to_tensor(ts_explain_np, device)
    sent_explain = to_tensor(sent_explain_np, device)

    print(
        "Running SHAP GradientExplainer "
        f"(background={len(ts_bg)}, explain={len(ts_explain)}, target={args.target})..."
    )
    wrapper = SHAPWrapper(model, args.target).to(device)
    wrapper.eval()
    if sent_bg is None:
        explainer = shap.GradientExplainer(wrapper, ts_bg)
        shap_values = explainer.shap_values(ts_explain)
    else:
        explainer = shap.GradientExplainer(wrapper, [ts_bg, sent_bg])
        shap_values = explainer.shap_values([ts_explain, sent_explain])
    shap_ts, shap_sent = split_shap_values(shap_values)

    importance = build_importance_frame(
        shap_ts,
        shap_sent,
        metadata["feature_columns"],
        metadata["sentiment_columns"],
    )

    target_suffix = "" if args.target == "regression" else f"_{args.target}"
    csv_path = MODELS_DIR / f"shap_feature_importance{target_suffix}.csv"
    bar_path = MODELS_DIR / f"shap_feature_importance{target_suffix}.png"
    heatmap_path = MODELS_DIR / f"shap_temporal_heatmap{target_suffix}.png"

    importance.to_csv(csv_path, index=False)
    save_bar_chart(importance, bar_path)
    save_temporal_heatmap(
        shap_ts,
        shap_sent,
        importance,
        metadata["feature_columns"],
        metadata["sentiment_columns"],
        heatmap_path,
    )

    print("SHAP artifacts written:")
    print(f"  {csv_path}")
    print(f"  {bar_path}")
    print(f"  {heatmap_path}")
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
