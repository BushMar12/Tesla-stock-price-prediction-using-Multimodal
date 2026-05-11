"""
Train standalone models across multiple seeds and compare test results.

Runs the standalone LSTM, GRU, Transformer, and XGBoost baselines for the
technical and technical+sentiment feature configurations. Results are written to:

    models/standalone_seed_results/per_seed_results.csv
    models/standalone_seed_results/aggregate_results.csv
    models/standalone_seed_results/model_ranking.csv

By default this runs five seeds: 42, 43, 44, 45, 46.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import DETERMINISTIC_TRAINING, MODELS_DIR, SENTIMENT_CONFIG, TRAINING_CONFIG
from model_comparison import (
    CONFIGS,
    backup_root_preprocessing_artifacts,
    build_features,
    restore_root_preprocessing_artifacts,
    safe_name,
    summarize_history,
)
from src.data.preprocessing import DataPreprocessor
from src.data.sentiment_data import fetch_sentiment_data
from src.data.stock_data import fetch_stock_data, load_stock_data
from src.features.technical import calculate_all_indicators
from src.models.regression_models import MultiModelRegressor


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone model comparison over five random seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to run. Default: 42 43 44 45 46.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["epochs"],
        help=f"Epochs per PyTorch model per seed/config. Default: {TRAINING_CONFIG['epochs']}.",
    )
    parser.add_argument(
        "--configs",
        choices=CONFIGS,
        nargs="+",
        default=CONFIGS,
        help="Feature configurations to run.",
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["synthetic", "rss", "alpha_vantage"],
        default=SENTIMENT_CONFIG.get("source", "alpha_vantage"),
        help="Sentiment source used for the technical_sentiment configuration.",
    )
    parser.add_argument(
        "--refresh-stock-data",
        action="store_true",
        help="Fetch fresh Yahoo Finance stock data instead of using the cached CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "standalone_seed_results",
        help="Directory for comparison CSV outputs.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save per-seed model artifacts under the output directory.",
    )
    return parser.parse_args()


def set_global_seed(seed: int, deterministic: bool = DETERMINISTIC_TRAINING) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def prepare_splits(sentiment_source: str, refresh_stock_data: bool) -> tuple[dict, DataPreprocessor]:
    if refresh_stock_data:
        stock_df = fetch_stock_data(start_date=config.START_DATE, end_date=config.END_DATE)
    else:
        stock_df = load_stock_data()

    sentiment_df = fetch_sentiment_data(
        stock_df,
        use_real_data=sentiment_source != "synthetic",
        source=sentiment_source,
    )
    indicators_df = calculate_all_indicators(stock_df)

    preprocessing_backups = backup_root_preprocessing_artifacts()
    preprocessor = DataPreprocessor()
    splits = preprocessor.prepare_data(stock_df, sentiment_df, indicators_df)
    restore_root_preprocessing_artifacts(preprocessing_backups)
    return splits, preprocessor


def train_one_config(
    *,
    seed: int,
    config_name: str,
    splits: dict,
    preprocessor: DataPreprocessor,
    epochs: int,
    output_dir: Path,
    save_models: bool,
) -> tuple[list[dict], list[dict]]:
    set_global_seed(seed)

    X_train, X_val, X_test = build_features(splits, config_name)
    y_train = splits["train"]["y_reg"]
    y_val = splits["val"]["y_reg"]
    y_test = splits["test"]["y_reg"]
    close_prices_val = splits["val"]["close_prices"]
    close_prices_test = splits["test"]["close_prices"]

    print("\n" + "=" * 72)
    print(f"Seed {seed} | Configuration: {config_name}")
    print("=" * 72)
    print(f"Training data shape:   {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape:       {X_test.shape}")

    model_dir = None
    if save_models:
        model_dir = output_dir / "model_artifacts" / f"seed_{seed}" / safe_name(config_name)
        model_dir.mkdir(parents=True, exist_ok=True)

    multi_model = MultiModelRegressor(input_size=X_train.shape[2], random_seed=seed)
    multi_model.train_all(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        plot_history=False,
        return_scaler=preprocessor.return_scaler,
        val_close_prices=close_prices_val,
        checkpoint_dir=model_dir,
    )
    multi_model.histories["XGBoost"] = multi_model.models["XGBoost"].training_history()

    results = multi_model.evaluate_all(
        X_test,
        y_test,
        return_scaler=preprocessor.return_scaler,
        close_prices=close_prices_test,
    )

    if save_models and model_dir is not None:
        multi_model.save_models(model_dir)

    eval_rows = []
    training_rows = []
    for model_name, metrics in results.items():
        history = multi_model.histories.get(model_name, {})
        training_summary = summarize_history(model_name, history)
        training_summary.update(
            {
                "Seed": seed,
                "Configuration": config_name,
                "Model": model_name,
            }
        )
        training_rows.append(training_summary)

        eval_rows.append(
            {
                "Seed": seed,
                "Configuration": config_name,
                "Model": model_name,
                "RMSE": float(metrics["RMSE"]),
                "MAE": float(metrics["MAE"]),
                "MAPE": float(metrics["MAPE"]),
                "MSE": float(metrics["MSE"]),
            }
        )

    return eval_rows, training_rows


def aggregate_results(per_seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["RMSE", "MAE", "MAPE", "MSE"]
    grouped = per_seed_df.groupby(["Configuration", "Model"], as_index=False)

    aggregate = grouped.agg(
        Runs=("Seed", "nunique"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        RMSE_min=("RMSE", "min"),
        RMSE_max=("RMSE", "max"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        MAE_min=("MAE", "min"),
        MAE_max=("MAE", "max"),
        MAPE_mean=("MAPE", "mean"),
        MAPE_std=("MAPE", "std"),
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
    )
    aggregate["MAE_rank_within_config"] = aggregate.groupby("Configuration")["MAE_mean"].rank(
        method="dense"
    )
    aggregate = aggregate.sort_values(["Configuration", "MAE_rank_within_config", "MAE_mean"])

    ranking = (
        per_seed_df.groupby("Model", as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    ranking.columns = [
        "Model" if col[0] == "Model" else f"{col[0]}_{col[1]}"
        for col in ranking.columns.to_flat_index()
    ]
    ranking["overall_MAE_rank"] = ranking["MAE_mean"].rank(method="dense")
    ranking = ranking.sort_values(["overall_MAE_rank", "MAE_mean"])
    return aggregate, ranking


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIG["epochs"] = args.epochs

    print("=" * 72)
    print("Standalone Model Five-Seed Comparison")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Configurations: {args.configs}")
    print(f"Sentiment source: {args.sentiment_source}")
    print(f"Deterministic training: {DETERMINISTIC_TRAINING}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 72)

    set_global_seed(args.seeds[0])
    splits, preprocessor = prepare_splits(args.sentiment_source, args.refresh_stock_data)

    all_eval_rows = []
    all_training_rows = []
    per_seed_path = args.output_dir / "per_seed_results.csv"
    training_path = args.output_dir / "per_seed_training_summary.csv"

    for seed in args.seeds:
        for config_name in args.configs:
            eval_rows, training_rows = train_one_config(
                seed=seed,
                config_name=config_name,
                splits=splits,
                preprocessor=preprocessor,
                epochs=args.epochs,
                output_dir=args.output_dir,
                save_models=args.save_models,
            )
            all_eval_rows.extend(eval_rows)
            all_training_rows.extend(training_rows)

            pd.DataFrame(all_eval_rows).to_csv(per_seed_path, index=False)
            pd.DataFrame(all_training_rows).to_csv(training_path, index=False)

    per_seed_df = pd.DataFrame(all_eval_rows)
    training_df = pd.DataFrame(all_training_rows)
    aggregate_df, ranking_df = aggregate_results(per_seed_df)

    aggregate_path = args.output_dir / "aggregate_results.csv"
    ranking_path = args.output_dir / "model_ranking.csv"
    settings_path = args.output_dir / "experiment_settings.csv"

    aggregate_df.to_csv(aggregate_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    training_df.to_csv(training_path, index=False)
    pd.DataFrame(
        [
            {
                "seeds": " ".join(str(seed) for seed in args.seeds),
                "epochs": args.epochs,
                "configs": " ".join(args.configs),
                "sentiment_source": args.sentiment_source,
                "deterministic_training": DETERMINISTIC_TRAINING,
                "save_models": args.save_models,
            }
        ]
    ).to_csv(settings_path, index=False)

    print("\n" + "=" * 72)
    print("Aggregate Results: lower MAE is better")
    print("=" * 72)
    display_cols = [
        "Configuration",
        "Model",
        "Runs",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "MAPE_mean",
        "MAE_rank_within_config",
    ]
    print(aggregate_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  Per-seed results:      {per_seed_path}")
    print(f"  Training summaries:    {training_path}")
    print(f"  Aggregate comparison:  {aggregate_path}")
    print(f"  Overall model ranking: {ranking_path}")
    print(f"  Experiment settings:   {settings_path}")


if __name__ == "__main__":
    main()
