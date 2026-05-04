"""
Training script for multi-model regression comparison
Trains LSTM, GRU, Transformer, and XGBoost models for three configurations:
  1. technical            — price + technical indicators only
  2. sentiment            — sentiment features only
  3. technical+sentiment  — technical features concatenated with sentiment features
Uses returns-based prediction for better accuracy.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from pathlib import Path
import random
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.stock_data import load_stock_data
from src.data.sentiment_data import fetch_sentiment_data
from src.features.technical import calculate_all_indicators
from src.data.preprocessing import DataPreprocessor
from src.models.regression_models import MultiModelRegressor
from config import (
    DETERMINISTIC_TRAINING,
    MODELS_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SENTIMENT_CONFIG,
    TRAINING_CONFIG,
)


CONFIGS = ["technical", "sentiment", "technical_sentiment"]
RESULTS_DIR = MODELS_DIR / "result"
APP_MODEL_CONFIG = "technical_sentiment"
ROOT_PREPROCESSING_ARTIFACTS = [
    "preprocessing_metadata.pkl",
    "feature_scaler.pkl",
    "return_scaler.pkl",
]


def set_global_seed(seed: int = RANDOM_SEED, deterministic: bool = DETERMINISTIC_TRAINING):
    """Seed Python, NumPy, and PyTorch for repeatable comparison runs."""
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


def safe_name(value: str) -> str:
    """Return a filesystem-friendly name for result folders."""
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def build_features(splits, config_name):
    """Return X_train/X_val/X_test for a given configuration."""
    if config_name == "technical":
        return (
            splits['train']['X_price'],
            splits['val']['X_price'],
            splits['test']['X_price'],
        )

    if splits['train']['X_sentiment'] is None:
        raise RuntimeError(
            f"Sentiment features unavailable but '{config_name}' "
            "configuration was requested."
        )

    if config_name == "sentiment":
        return (
            splits['train']['X_sentiment'],
            splits['val']['X_sentiment'],
            splits['test']['X_sentiment'],
        )

    if config_name == "technical_sentiment":
        X_train = np.concatenate(
            [splits['train']['X_price'], splits['train']['X_sentiment']], axis=2
        )
        X_val = np.concatenate(
            [splits['val']['X_price'], splits['val']['X_sentiment']], axis=2
        )
        X_test = np.concatenate(
            [splits['test']['X_price'], splits['test']['X_sentiment']], axis=2
        )
        return X_train, X_val, X_test

    raise ValueError(f"Unknown configuration: {config_name}")


def validate_app_model_artifacts(source_dir: Path):
    """Confirm app-compatible comparison artifacts exist in one folder."""
    artifacts = [
        "multi_model_metadata.pkl",
        "lstm_best.pt",
        "gru_best.pt",
        "transformer_best.pt",
        "xgboost_best.pkl",
        "preprocessing_metadata.pkl",
        "feature_scaler.pkl",
        "return_scaler.pkl",
    ]

    missing = [artifact for artifact in artifacts if not (source_dir / artifact).exists()]
    if missing:
        raise RuntimeError(
            f"Missing app model artifacts in {source_dir}: {', '.join(missing)}"
        )

    print(f"\nApp comparison models are ready in: {source_dir}")


def backup_root_preprocessing_artifacts():
    """Keep the main training artifacts intact while comparison preprocessing runs."""
    import joblib

    backups = {}
    for artifact in ROOT_PREPROCESSING_ARTIFACTS:
        path = MODELS_DIR / artifact
        if path.exists():
            backups[artifact] = joblib.load(path)
    return backups


def restore_root_preprocessing_artifacts(backups):
    """Restore root preprocessing artifacts used by the main training pipeline."""
    import joblib

    for artifact, value in backups.items():
        joblib.dump(value, MODELS_DIR / artifact)


def save_preprocessing_artifacts(preprocessor: DataPreprocessor, save_dir: Path, config_name: str):
    """Save preprocessing artifacts beside comparison models."""
    import joblib

    if config_name == "technical":
        feature_columns = preprocessor.feature_columns
        sentiment_columns = []
    elif config_name == "sentiment":
        feature_columns = []
        sentiment_columns = preprocessor.sentiment_columns
    elif config_name == "technical_sentiment":
        feature_columns = preprocessor.feature_columns
        sentiment_columns = preprocessor.sentiment_columns
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

    metadata = {
        "feature_columns": feature_columns,
        "sentiment_columns": sentiment_columns,
        "sequence_length": preprocessor.sequence_length,
        "n_price_features": len(feature_columns),
        "n_sentiment_features": len(sentiment_columns),
        "comparison_configuration": config_name,
        "random_seed": RANDOM_SEED,
        "deterministic_training": DETERMINISTIC_TRAINING,
    }

    joblib.dump(metadata, save_dir / "preprocessing_metadata.pkl")
    joblib.dump(preprocessor.feature_scaler, save_dir / "feature_scaler.pkl")
    joblib.dump(preprocessor.return_scaler, save_dir / "return_scaler.pkl")


def run_config(config_name, splits, preprocessor):
    """Train and evaluate all baselines for a single feature configuration."""
    set_global_seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print(f"CONFIGURATION: {config_name}")
    print("=" * 60)

    X_train, X_val, X_test = build_features(splits, config_name)
    y_train = splits['train']['y_reg']
    y_val = splits['val']['y_reg']
    y_test = splits['test']['y_reg']
    close_prices_val = splits['val']['close_prices']
    close_prices_test = splits['test']['close_prices']

    print(f"Training data shape:   {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape:       {X_test.shape}")

    save_dir = MODELS_DIR / f"comparison_{config_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config_result_dir = RESULTS_DIR / safe_name(config_name)
    config_result_dir.mkdir(parents=True, exist_ok=True)

    multi_model = MultiModelRegressor(input_size=X_train.shape[2], random_seed=RANDOM_SEED)
    multi_model.train_all(
        X_train, y_train, X_val, y_val,
        epochs=TRAINING_CONFIG["epochs"],
        plot_history=False,
        return_scaler=preprocessor.return_scaler,
        val_close_prices=close_prices_val,
        checkpoint_dir=save_dir,
    )
    multi_model.histories["XGBoost"] = multi_model.models["XGBoost"].training_history()
    multi_model.plot_training_history(save_path=config_result_dir / "training_history.png")

    results = multi_model.evaluate_all(
        X_test, y_test,
        return_scaler=preprocessor.return_scaler,
        close_prices=close_prices_test,
    )

    actual_returns = preprocessor.return_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    actual_prices = close_prices_test * (1 + actual_returns)

    evaluation_rows = []
    training_rows = []
    for model_name, metrics in results.items():
        model_result_dir = config_result_dir / safe_name(model_name)
        model_result_dir.mkdir(parents=True, exist_ok=True)

        history = multi_model.histories.get(model_name, {})
        history_df = history_to_dataframe(model_name, history)
        history_df.to_csv(model_result_dir / "training_history.csv", index=False)

        training_summary = summarize_history(model_name, history)
        training_summary["Configuration"] = config_name
        training_summary["Model"] = model_name
        training_summary["Random Seed"] = RANDOM_SEED
        training_summary["Deterministic Training"] = DETERMINISTIC_TRAINING
        training_rows.append(training_summary)

        eval_row = {
            'Configuration': config_name,
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'MSE': metrics['MSE'],
            'Random Seed': RANDOM_SEED,
            'Deterministic Training': DETERMINISTIC_TRAINING,
        }
        evaluation_rows.append(eval_row)
        pd.DataFrame([eval_row]).to_csv(model_result_dir / "evaluation_results.csv", index=False)

        predictions_df = pd.DataFrame({
            "actual_return": actual_returns,
            "predicted_return": metrics["pred_returns"],
            "actual_price": actual_prices,
            "predicted_price": metrics["predictions"],
            "prediction_error": metrics["predictions"] - actual_prices,
            "abs_error": np.abs(metrics["predictions"] - actual_prices),
        })
        predictions_df.to_csv(model_result_dir / "test_predictions.csv", index=False)

        print_model_result(config_name, model_name, training_summary, eval_row, model_result_dir)

    multi_model.save_models(save_dir)
    save_preprocessing_artifacts(preprocessor, save_dir, config_name)

    pd.DataFrame(training_rows).to_csv(config_result_dir / "training_summary.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(config_result_dir / "evaluation_results.csv", index=False)

    return evaluation_rows, training_rows


def history_to_dataframe(model_name, history):
    """Normalize PyTorch and XGBoost histories into a CSV-friendly table."""
    if not history:
        return pd.DataFrame(columns=["epoch"])

    if model_name == "XGBoost":
        n_epochs = max(len(history.get("train_mae", [])), len(history.get("val_mae", [])))
        rows = []
        for idx in range(n_epochs):
            rows.append({
                "epoch": idx + 1,
                "train_mae_scaled_return": get_history_value(history, "train_mae", idx),
                "val_mae_scaled_return": get_history_value(history, "val_mae", idx),
            })
        return pd.DataFrame(rows)

    n_epochs = max(
        len(history.get("train_loss", [])),
        len(history.get("val_loss", [])),
        len(history.get("val_mae", [])),
        len(history.get("lr", [])),
    )
    rows = []
    for idx in range(n_epochs):
        rows.append({
            "epoch": idx + 1,
            "train_loss": get_history_value(history, "train_loss", idx),
            "val_loss": get_history_value(history, "val_loss", idx),
            "val_mae_dollars": get_history_value(history, "val_mae", idx),
            "learning_rate": get_history_value(history, "lr", idx),
        })
    return pd.DataFrame(rows)


def get_history_value(history, key, idx):
    values = history.get(key, [])
    return values[idx] if idx < len(values) else np.nan


def summarize_history(model_name, history):
    """Create one row describing train/validation behavior for a model."""
    summary = {
        "History Epochs": 0,
        "Best Epoch": np.nan,
        "Final Train Loss": np.nan,
        "Final Val Loss": np.nan,
        "Best Val MAE": np.nan,
        "Final Val MAE": np.nan,
        "Final Learning Rate": np.nan,
        "Final Train MAE Scaled Return": np.nan,
        "Final Val MAE Scaled Return": np.nan,
        "Best Val MAE Scaled Return": np.nan,
    }

    if not history:
        return summary

    if model_name == "XGBoost":
        val_mae = history.get("val_mae", [])
        train_mae = history.get("train_mae", [])
        summary["History Epochs"] = max(len(train_mae), len(val_mae))
        if val_mae:
            summary["Best Epoch"] = int(np.argmin(val_mae) + 1)
            summary["Best Val MAE Scaled Return"] = float(np.min(val_mae))
            summary["Final Val MAE Scaled Return"] = float(val_mae[-1])
        if train_mae:
            summary["Final Train MAE Scaled Return"] = float(train_mae[-1])
        return summary

    val_mae = history.get("val_mae", [])
    summary["History Epochs"] = len(history.get("train_loss", []))
    if history.get("train_loss"):
        summary["Final Train Loss"] = float(history["train_loss"][-1])
    if history.get("val_loss"):
        summary["Final Val Loss"] = float(history["val_loss"][-1])
    if val_mae:
        summary["Best Epoch"] = int(np.argmin(val_mae) + 1)
        summary["Best Val MAE"] = float(np.min(val_mae))
        summary["Final Val MAE"] = float(val_mae[-1])
    if history.get("lr"):
        summary["Final Learning Rate"] = float(history["lr"][-1])
    return summary


def print_model_result(config_name, model_name, training_summary, eval_row, model_result_dir):
    """Print concise train/validation and test evaluation output for one model."""
    print("\n" + "-" * 60)
    print(f"Stored results for {model_name} [{config_name}]")
    print("-" * 60)
    print(f"  Result folder: {model_result_dir}")
    print(f"  History epochs: {training_summary['History Epochs']}")
    if not pd.isna(training_summary["Best Epoch"]):
        print(f"  Best epoch: {int(training_summary['Best Epoch'])}")
    if not pd.isna(training_summary["Final Train Loss"]):
        print(f"  Final train loss: {training_summary['Final Train Loss']:.6f}")
    if not pd.isna(training_summary["Final Val Loss"]):
        print(f"  Final val loss:   {training_summary['Final Val Loss']:.6f}")
    if not pd.isna(training_summary["Best Val MAE"]):
        print(f"  Best val MAE:     ${training_summary['Best Val MAE']:.2f}")
    if not pd.isna(training_summary["Final Val MAE Scaled Return"]):
        print(f"  Final val MAE:    {training_summary['Final Val MAE Scaled Return']:.6f} scaled return")
    print(f"  Test RMSE:        ${eval_row['RMSE']:.2f}")
    print(f"  Test MAE:         ${eval_row['MAE']:.2f}")
    print(f"  Test MAPE:        {eval_row['MAPE']:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare LSTM/GRU/Transformer/XGBoost baselines."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_CONFIG["epochs"],
        help=f"Max training epochs (default: {TRAINING_CONFIG['epochs']}).",
    )
    return parser.parse_args()


def main():
    """Train and compare baseline models across feature configurations."""
    global RANDOM_SEED
    args = parse_args()
    RANDOM_SEED = args.seed
    TRAINING_CONFIG["epochs"] = args.epochs
    set_global_seed(RANDOM_SEED)

    print("=" * 60)
    print("Multi-Model Regression Training")
    print("LSTM vs GRU vs Transformer vs XGBoost")
    print("Configurations: technical | sentiment | technical+sentiment")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Deterministic training: {DETERMINISTIC_TRAINING}")
    print("=" * 60)

    # Step 1: Fetch and preprocess data once; reuse for both configurations.
    print("\n Step 1: Fetching and preprocessing data...")
    stock_df = load_stock_data()

    sentiment_source = SENTIMENT_CONFIG.get("source", "alpha_vantage")
    print(f"Sentiment source: {sentiment_source}")
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

    # Step 2: Run each configuration.
    all_eval_rows = []
    all_training_rows = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "deterministic_training": DETERMINISTIC_TRAINING,
        "batch_size": TRAINING_CONFIG["batch_size"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "epochs": TRAINING_CONFIG["epochs"],
        "train_split": TRAINING_CONFIG["train_split"],
        "val_split": TRAINING_CONFIG["val_split"],
        "test_split": TRAINING_CONFIG["test_split"],
    }]).to_csv(RESULTS_DIR / "experiment_settings.csv", index=False)

    for config_name in CONFIGS:
        eval_rows, training_rows = run_config(config_name, splits, preprocessor)
        all_eval_rows.extend(eval_rows)
        all_training_rows.extend(training_rows)

        if config_name == APP_MODEL_CONFIG:
            validate_app_model_artifacts(MODELS_DIR / f"comparison_{config_name}")

    # Step 3: Display and save combined comparison.
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    comparison_df = pd.DataFrame(all_eval_rows)
    display_df = comparison_df.copy()
    display_df['RMSE'] = display_df['RMSE'].map(lambda v: f"${v:.2f}")
    display_df['MAE'] = display_df['MAE'].map(lambda v: f"${v:.2f}")
    display_df['MAPE'] = display_df['MAPE'].map(lambda v: f"{v:.2f}%")
    print(display_df.to_string(index=False))

    best_idx = comparison_df['MAE'].idxmin()
    best = comparison_df.loc[best_idx]
    print(f"\nBest overall (lowest MAE): {best['Model']} "
          f"[{best['Configuration']}] — ${best['MAE']:.2f}")

    training_df = pd.DataFrame(all_training_rows)
    comparison_df.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    comparison_df.to_csv(RESULTS_DIR / 'evaluation_results.csv', index=False)
    training_df.to_csv(RESULTS_DIR / 'training_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nCombined evaluation CSV: {RESULTS_DIR / 'evaluation_results.csv'}")
    print(f"Combined training summary CSV: {RESULTS_DIR / 'training_summary.csv'}")
    print(f"Per-model folders: {RESULTS_DIR}")
    print("\nTo run the Streamlit app with model comparison:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
