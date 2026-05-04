"""
Fetch and cache Alpha Vantage TSLA news sentiment separately from training.

This script is intentionally separate from train.py so sentiment data can be
prepared once, resumed after rate limits reset, and reused by later training
runs. It does not rotate API keys to bypass provider limits.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SENTIMENT_CONFIG, START_DATE, END_DATE
from src.data.stock_data import fetch_stock_data
from src.data.sentiment_data import fetch_alpha_vantage_news_sentiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch/cache Alpha Vantage TSLA news sentiment for training."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ALPHA_VANTAGE_API_KEY"),
        help="Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY.",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="Stock/sentiment start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=SENTIMENT_CONFIG.get("alpha_vantage_chunk_days", 365),
        help="Number of days per Alpha Vantage request.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=SENTIMENT_CONFIG.get("alpha_vantage_request_sleep", 12),
        help="Seconds to sleep between Alpha Vantage requests.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore final daily sentiment cache. Raw partial cache may still be used for resume.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing Alpha Vantage API key. Set ALPHA_VANTAGE_API_KEY or pass --api-key."
        )

    SENTIMENT_CONFIG["alpha_vantage_api_key"] = args.api_key
    SENTIMENT_CONFIG["alpha_vantage_chunk_days"] = args.chunk_days
    SENTIMENT_CONFIG["alpha_vantage_request_sleep"] = args.request_sleep
    SENTIMENT_CONFIG["alpha_vantage_use_cache"] = not args.force_refresh

    print("Preparing TSLA trading dates...")
    stock_df = fetch_stock_data(
        start_date=args.start_date,
        end_date=args.end_date,
        save=True,
    )

    print("Fetching/caching Alpha Vantage sentiment...")
    sentiment_df = fetch_alpha_vantage_news_sentiment(stock_df, save=True)

    print("\nAlpha Vantage sentiment cache ready.")
    print(f"Rows: {len(sentiment_df)}")
    print(f"Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print("Files written under data/raw/:")
    print("  alpha_vantage_tsla_news_raw.csv")
    print("  alpha_vantage_tsla_sentiment_data.csv")
    print("  sentiment_data.csv")


if __name__ == "__main__":
    main()
