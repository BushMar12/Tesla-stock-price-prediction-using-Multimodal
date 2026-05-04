"""
Audit sentiment feature coverage, temporal alignment, and dead columns.

Outputs:
    models/sentiment_alignment_report.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR, RAW_DATA_DIR, STOCK_SYMBOL
from src.data.stock_data import load_stock_data
from src.features.technical import add_target_variables


SENTIMENT_COLUMNS = [
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "news_count",
    "sentiment_compound_lag1",
    "sentiment_compound_lag2",
    "sentiment_compound_lag3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sentiment temporal alignment.")
    parser.add_argument(
        "--sentiment-path",
        type=Path,
        default=RAW_DATA_DIR / "sentiment_data.csv",
        help="Sentiment CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "sentiment_alignment_report.txt",
        help="Plain-text report output path.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-6,
        help="Variance threshold for constant-feature detection.",
    )
    return parser.parse_args()


def load_sentiment(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def coverage_summary(sentiment_df: pd.DataFrame) -> dict:
    rows = len(sentiment_df)
    if rows == 0:
        return {
            "rows": 0,
            "non_default_rows": 0,
            "non_default_fraction": 0.0,
            "date_min": "N/A",
            "date_max": "N/A",
        }

    non_default_mask = pd.Series(False, index=sentiment_df.index)
    if "news_count" in sentiment_df:
        non_default_mask |= sentiment_df["news_count"].fillna(0) > 0
    for col in [
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
    ]:
        if col in sentiment_df:
            non_default_mask |= sentiment_df[col].fillna(0).abs() > 1e-12

    return {
        "rows": rows,
        "non_default_rows": int(non_default_mask.sum()),
        "non_default_fraction": float(non_default_mask.mean()),
        "date_min": sentiment_df["date"].min().date().isoformat(),
        "date_max": sentiment_df["date"].max().date().isoformat(),
    }


def build_alignment_frame(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = load_stock_data(symbol=STOCK_SYMBOL)
    stock_df = add_target_variables(stock_df.copy(), horizon=1)
    stock_df = stock_df[["date", "Target_Return"]].copy()
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    merged = stock_df.merge(sentiment_df, on="date", how="left").sort_values("date")
    return merged.dropna(subset=["Target_Return"])


def lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_compound" not in df:
        return pd.DataFrame(columns=["sentiment_shift", "correlation"])

    rows = []
    for shift in [-3, -2, -1, 0, 1, 2, 3]:
        shifted = df["sentiment_compound"].shift(shift)
        valid = pd.concat([df["Target_Return"], shifted], axis=1).dropna()
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) > 2 else np.nan
        rows.append(
            {
                "sentiment_shift": shift,
                "interpretation": interpret_shift(shift),
                "correlation": corr,
            }
        )
    return pd.DataFrame(rows)


def interpret_shift(shift: int) -> str:
    if shift < 0:
        return f"future sentiment t+{abs(shift)}"
    if shift > 0:
        return f"past sentiment t-{shift}"
    return "same-day sentiment t"


def constant_features(sentiment_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    numeric_cols = [
        col for col in SENTIMENT_COLUMNS
        if col in sentiment_df.columns and pd.api.types.is_numeric_dtype(sentiment_df[col])
    ]
    rows = []
    for col in numeric_cols:
        variance = float(sentiment_df[col].fillna(0).var())
        rows.append(
            {
                "feature": col,
                "variance": variance,
                "is_constantish": variance < threshold,
            }
        )
    return pd.DataFrame(rows)


def verdict(coverage: dict, correlations: pd.DataFrame, constants: pd.DataFrame) -> str:
    dead_features = constants[constants["is_constantish"]]["feature"].tolist()
    future = correlations[correlations["sentiment_shift"] < 0]
    non_future = correlations[correlations["sentiment_shift"] >= 0]

    suspicious_future = False
    if not future.empty and not non_future.empty:
        future_best = future["correlation"].abs().max()
        non_future_best = non_future["correlation"].abs().max()
        suspicious_future = bool(future_best > non_future_best * 1.25 and future_best > 0.05)

    if coverage["non_default_fraction"] < 0.20:
        return "Verdict: fix sentiment coverage before relying on multimodal results."
    if suspicious_future:
        return "Verdict: audit timestamps; future sentiment correlates unusually strongly."
    if dead_features:
        return "Verdict: usable, but drop or ignore near-constant sentiment columns."
    return "Verdict: sentiment alignment looks usable for ablation validation."


def main() -> None:
    args = parse_args()
    sentiment_df = load_sentiment(args.sentiment_path)
    coverage = coverage_summary(sentiment_df)
    alignment_df = build_alignment_frame(sentiment_df)
    correlations = lag_correlations(alignment_df)
    constants = constant_features(sentiment_df, args.variance_threshold)

    lines = [
        "Sentiment Alignment Audit",
        "=========================",
        f"Sentiment file: {args.sentiment_path}",
        f"Rows: {coverage['rows']}",
        f"Date range: {coverage['date_min']} to {coverage['date_max']}",
        (
            "Non-default sentiment rows: "
            f"{coverage['non_default_rows']} "
            f"({coverage['non_default_fraction']:.1%})"
        ),
        "",
        "Correlation with Target_Return by sentiment timing:",
        correlations.to_string(index=False),
        "",
        "Constant-feature check:",
        constants.to_string(index=False),
        "",
        verdict(coverage, correlations, constants),
    ]

    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
