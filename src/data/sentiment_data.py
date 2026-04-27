"""
Sentiment data collection and analysis
Uses free sources: RSS feeds, VADER sentiment, and optional FinBERT
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import requests
from bs4 import BeautifulSoup
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import STOCK_SYMBOL, RAW_DATA_DIR, SENTIMENT_CONFIG


SENTIMENT_COLUMNS = [
    'date', 'sentiment_compound', 'sentiment_positive',
    'sentiment_negative', 'sentiment_neutral', 'news_count',
    'sentiment_compound_lag1', 'sentiment_compound_lag2',
    'sentiment_compound_lag3'
]


def _add_sentiment_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged sentiment features used by the multimodal model."""
    df = df.sort_values('date').copy()
    for lag in [1, 2, 3]:
        df[f'sentiment_compound_lag{lag}'] = df['sentiment_compound'].shift(lag)
    return df.dropna(subset=[f'sentiment_compound_lag{lag}' for lag in [1, 2, 3]])


def _align_sentiment_to_stock_dates(stock_df: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Align daily sentiment rows to stock trading dates and fill missing dates as neutral."""
    stock_dates = pd.DataFrame({'date': pd.to_datetime(stock_df['date']).dt.date})
    daily = daily_sentiment.copy()
    daily['date'] = pd.to_datetime(daily['date']).dt.date

    merged = stock_dates.merge(daily, on='date', how='left')
    merged = merged.fillna({
        'sentiment_compound': 0.0,
        'sentiment_positive': 0.0,
        'sentiment_negative': 0.0,
        'sentiment_neutral': 1.0,
        'news_count': 0,
    })
    return _add_sentiment_lags(merged)[SENTIMENT_COLUMNS]


class SentimentAnalyzer:
    """Sentiment analysis using VADER and optionally FinBERT"""
    
    def __init__(self, use_finbert: bool = False):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT (requires more resources)
        """
        self.vader = SentimentIntensityAnalyzer()
        self.use_finbert = use_finbert
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        if use_finbert:
            self._load_finbert()
    
    def _load_finbert(self):
        """Load FinBERT model for financial sentiment analysis"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            model_name = SENTIMENT_CONFIG["finbert_model"]
            print(f"Loading FinBERT model: {model_name}...")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_model.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.finbert_model.to(self.device)
            
            print(f"FinBERT loaded on {self.device}")
        except Exception as e:
            print(f"Failed to load FinBERT: {e}")
            print("Falling back to VADER sentiment")
            self.use_finbert = False
    
    def analyze_vader(self, text: str) -> dict:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment scores
        """
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_finbert(self, text: str) -> dict:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment scores
        """
        import torch
        
        inputs = self.finbert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        # FinBERT labels: positive, negative, neutral
        return {
            'compound': probs[0] - probs[1],  # positive - negative
            'positive': probs[0],
            'negative': probs[1],
            'neutral': probs[2]
        }
    
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using the configured method.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment scores
        """
        if self.use_finbert and self.finbert_model is not None:
            return self.analyze_finbert(text)
        return self.analyze_vader(text)


class NewsCollector:
    """Collect news headlines from free RSS feeds"""
    
    # Free RSS feeds for Tesla/stock news
    RSS_FEEDS = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=Tesla+stock&hl=en-US&gl=US&ceid=US:en",
    ]
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer(use_finbert=False)  # Start with VADER
    
    def fetch_rss_headlines(self) -> list:
        """
        Fetch headlines from RSS feeds.
        
        Returns:
            List of dicts with headline info
        """
        all_headlines = []
        
        for feed_url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:SENTIMENT_CONFIG["max_headlines_per_day"]]:
                    headline = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'source': feed_url.split('/')[2],
                    }
                    
                    # Parse date
                    try:
                        if headline['published']:
                            headline['date'] = pd.to_datetime(headline['published']).date()
                        else:
                            headline['date'] = datetime.now().date()
                    except:
                        headline['date'] = datetime.now().date()
                    
                    all_headlines.append(headline)
                    
            except Exception as e:
                print(f"Error fetching from {feed_url}: {e}")
        
        return all_headlines
    
    def analyze_headlines(self, headlines: list) -> pd.DataFrame:
        """
        Analyze sentiment of headlines.
        
        Args:
            headlines: List of headline dicts
        
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for headline in headlines:
            text = f"{headline['title']} {headline['summary']}"
            sentiment = self.analyzer.analyze(text)
            
            results.append({
                'date': headline['date'],
                'title': headline['title'],
                'source': headline['source'],
                **sentiment
            })
        
        return pd.DataFrame(results)
    
    def get_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment by day.
        
        Args:
            df: DataFrame with individual headline sentiments
        
        Returns:
            DataFrame with daily aggregated sentiment
        """
        daily = df.groupby('date').agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'title': 'count'
        }).reset_index()
        
        daily.columns = ['date', 'sentiment_compound', 'sentiment_positive', 
                        'sentiment_negative', 'sentiment_neutral', 'news_count']
        
        return daily


def generate_synthetic_sentiment(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic sentiment data based on LAGGED price movements.
    Uses yesterday's returns and rolling trends to avoid same-day information leakage.
    
    Args:
        stock_df: DataFrame with stock price data
    
    Returns:
        DataFrame with synthetic sentiment scores
    """
    df = stock_df.copy()
    
    np.random.seed(42)
    
    # Use LAGGED information only - no same-day return leakage
    lagged_1d = df['returns'].shift(1).fillna(0)
    lagged_5d_mean = df['returns'].rolling(5).mean().shift(1).fillna(0)
    volatility_10d = df['returns'].rolling(10).std().shift(1).fillna(df['returns'].std())
    
    # Compound sentiment from lagged signals + independent noise
    df['sentiment_compound'] = np.tanh(
        0.25 * lagged_1d * 5 +                            # Yesterday's return
        0.25 * lagged_5d_mean * 8 +                        # Weekly trend (lagged)
        0.10 * np.tanh(-volatility_10d * 20) +             # High vol -> negative sentiment
        0.40 * np.random.normal(0, 0.25, len(df))          # Independent noise
    )
    
    # Decompose into positive/negative/neutral
    df['sentiment_positive'] = np.clip(df['sentiment_compound'] + 0.5, 0, 1) / 2
    df['sentiment_negative'] = np.clip(-df['sentiment_compound'] + 0.5, 0, 1) / 2
    df['sentiment_neutral'] = 1 - df['sentiment_positive'] - df['sentiment_negative']
    df['news_count'] = np.random.randint(1, 20, len(df))
    
    df = _add_sentiment_lags(df)
    return df[SENTIMENT_COLUMNS]


def _alpha_vantage_sentiment_from_article(article: dict) -> dict:
    """Extract TSLA-specific Alpha Vantage sentiment from a news item."""
    ticker_sentiments = article.get('ticker_sentiment', []) or []
    tsla_sentiment = None
    for item in ticker_sentiments:
        if item.get('ticker') == STOCK_SYMBOL:
            tsla_sentiment = item
            break

    if tsla_sentiment:
        compound = float(tsla_sentiment.get('ticker_sentiment_score', 0) or 0)
        relevance = float(tsla_sentiment.get('relevance_score', 0) or 0)
        label = tsla_sentiment.get('ticker_sentiment_label', '')
    else:
        compound = float(article.get('overall_sentiment_score', 0) or 0)
        relevance = 0.0
        label = article.get('overall_sentiment_label', '')

    return {
        'compound': compound,
        'positive': max(compound, 0.0),
        'negative': max(-compound, 0.0),
        'neutral': max(0.0, 1.0 - abs(compound)),
        'relevance': relevance,
        'label': label,
    }


def _parse_alpha_vantage_time(value: str):
    """Parse Alpha Vantage news timestamp format."""
    return pd.to_datetime(value, format='%Y%m%dT%H%M%S', errors='coerce')


def fetch_alpha_vantage_news_sentiment(stock_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Fetch real TSLA news sentiment from Alpha Vantage NEWS_SENTIMENT.

    Requires ALPHA_VANTAGE_API_KEY in the environment. Raw article-level
    records are cached to data/raw/alpha_vantage_tsla_news_raw.csv, and the
    daily aligned model features are cached to data/raw/alpha_vantage_tsla_sentiment_data.csv.
    """
    daily_cache_path = RAW_DATA_DIR / "alpha_vantage_tsla_sentiment_data.csv"
    raw_cache_path = RAW_DATA_DIR / "alpha_vantage_tsla_news_raw.csv"

    if SENTIMENT_CONFIG.get("alpha_vantage_use_cache", True) and daily_cache_path.exists():
        cached = pd.read_csv(daily_cache_path)
        cached['date'] = pd.to_datetime(cached['date']).dt.date
        print(f"Loaded cached Alpha Vantage sentiment from {daily_cache_path}")
        return cached[SENTIMENT_COLUMNS]

    api_key = SENTIMENT_CONFIG.get("alpha_vantage_api_key")
    if not api_key:
        raise ValueError(
            "Alpha Vantage sentiment requires ALPHA_VANTAGE_API_KEY. "
            "Set it in your shell before training."
        )

    stock_start = pd.to_datetime(stock_df['date']).min()
    stock_end = pd.to_datetime(stock_df['date']).max()
    configured_start = pd.to_datetime(SENTIMENT_CONFIG.get("alpha_vantage_start_date") or stock_start)
    start = max(stock_start, configured_start)
    end = stock_end

    chunk_days = int(SENTIMENT_CONFIG.get("alpha_vantage_chunk_days", 30))
    request_sleep = float(SENTIMENT_CONFIG.get("alpha_vantage_request_sleep", 12))
    limit = int(SENTIMENT_CONFIG.get("alpha_vantage_limit", 1000))

    records = []
    if SENTIMENT_CONFIG.get("alpha_vantage_use_cache", True) and raw_cache_path.exists():
        cached_raw = pd.read_csv(raw_cache_path)
        if not cached_raw.empty:
            cached_raw['date'] = pd.to_datetime(cached_raw['date']).dt.date
            cached_raw['published_at'] = pd.to_datetime(cached_raw['published_at'])
            records = cached_raw.to_dict('records')
            last_cached_date = pd.to_datetime(cached_raw['date']).max()
            if pd.notna(last_cached_date):
                resume_date = last_cached_date + pd.Timedelta(days=1)
                if resume_date > start:
                    print(f"Resuming Alpha Vantage fetch after cached date {last_cached_date.date()}")
                    start = resume_date

    current = start
    print(f"Fetching Alpha Vantage TSLA news sentiment from {start.date()} to {end.date()}...")

    def save_partial_raw():
        if save and records:
            partial = pd.DataFrame(records).drop_duplicates(subset=['url', 'title', 'published_at'])
            partial.to_csv(raw_cache_path, index=False)
            print(f"Partial Alpha Vantage raw cache saved to {raw_cache_path}")

    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": STOCK_SYMBOL,
            "time_from": current.strftime("%Y%m%dT0000"),
            "time_to": chunk_end.strftime("%Y%m%dT2359"),
            "sort": "EARLIEST",
            "limit": limit,
            "apikey": api_key,
        }
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if "Note" in payload:
            save_partial_raw()
            raise RuntimeError(
                "Alpha Vantage rate limit reached. Partial results were saved if any "
                f"requests succeeded. Try again after the daily quota resets. Details: {payload['Note']}"
            )
        if "Information" in payload:
            save_partial_raw()
            raise RuntimeError(
                "Alpha Vantage returned an informational/rate-limit response. "
                "Partial results were saved if any requests succeeded. Try again after "
                f"the daily quota resets. Details: {payload['Information']}"
            )
        if "Error Message" in payload:
            save_partial_raw()
            raise RuntimeError(f"Alpha Vantage error: {payload['Error Message']}")

        for article in payload.get("feed", []):
            published_at = _parse_alpha_vantage_time(article.get("time_published", ""))
            if pd.isna(published_at):
                continue
            sentiment = _alpha_vantage_sentiment_from_article(article)
            records.append({
                "date": published_at.date(),
                "published_at": published_at,
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "compound": sentiment["compound"],
                "positive": sentiment["positive"],
                "negative": sentiment["negative"],
                "neutral": sentiment["neutral"],
                "relevance": sentiment["relevance"],
                "label": sentiment["label"],
            })

        print(f"Fetched {len(payload.get('feed', []))} Alpha Vantage articles for {current.date()} to {chunk_end.date()}")
        save_partial_raw()
        current = chunk_end + pd.Timedelta(days=1)
        if current <= end and request_sleep > 0:
            time.sleep(request_sleep)

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        print("No Alpha Vantage articles returned; using neutral sentiment.")
        daily = pd.DataFrame(columns=[
            'date', 'sentiment_compound', 'sentiment_positive',
            'sentiment_negative', 'sentiment_neutral', 'news_count'
        ])
    else:
        raw_df = raw_df.drop_duplicates(subset=['url', 'title', 'published_at'])
        daily = raw_df.groupby('date').agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'title': 'count',
        }).reset_index()
        daily.columns = [
            'date', 'sentiment_compound', 'sentiment_positive',
            'sentiment_negative', 'sentiment_neutral', 'news_count'
        ]

    aligned = _align_sentiment_to_stock_dates(stock_df, daily)

    if save:
        if not raw_df.empty:
            raw_df.to_csv(raw_cache_path, index=False)
            print(f"Raw Alpha Vantage news saved to {raw_cache_path}")
        aligned.to_csv(daily_cache_path, index=False)
        print(f"Alpha Vantage sentiment data saved to {daily_cache_path}")

        # Keep the existing default sentiment_data.csv contract for training.
        aligned.to_csv(RAW_DATA_DIR / "sentiment_data.csv", index=False)
        print(f"Training sentiment data saved to {RAW_DATA_DIR / 'sentiment_data.csv'}")

    return aligned


def fetch_sentiment_data(
    stock_df: pd.DataFrame,
    use_real_data: bool = True,
    save: bool = True,
    source: str = None
) -> pd.DataFrame:
    """
    Fetch or generate sentiment data.
    
    Args:
        stock_df: DataFrame with stock data (for dates and synthetic generation)
        use_real_data: Whether to try fetching real news data
        save: Whether to save the data to disk
        source: Sentiment source: synthetic, rss, or alpha_vantage
    
    Returns:
        DataFrame with sentiment data
    """
    if source is None:
        source = SENTIMENT_CONFIG.get("source", "rss" if use_real_data else "synthetic")
    if use_real_data and source == "synthetic":
        source = "rss"

    if source == "alpha_vantage":
        return fetch_alpha_vantage_news_sentiment(stock_df, save=save)

    if source == "rss":
        try:
            print("Fetching news headlines...")
            collector = NewsCollector()
            headlines = collector.fetch_rss_headlines()
            
            if headlines:
                sentiment_df = collector.analyze_headlines(headlines)
                daily_sentiment = collector.get_daily_sentiment(sentiment_df)
                
                merged = _align_sentiment_to_stock_dates(stock_df, daily_sentiment)
                
                print(f"Fetched sentiment for {len(merged)} days")
                
                if save:
                    save_path = RAW_DATA_DIR / "sentiment_data.csv"
                    merged.to_csv(save_path, index=False)
                
                return merged
        except Exception as e:
            print(f"Error fetching RSS sentiment data: {e}")
            print("Falling back to synthetic sentiment...")
    elif source != "synthetic":
        raise ValueError(f"Unknown sentiment source: {source}")
    
    # Generate synthetic sentiment
    print("Generating synthetic sentiment data...")
    sentiment_df = generate_synthetic_sentiment(stock_df)
    
    if save:
        save_path = RAW_DATA_DIR / "sentiment_data.csv"
        sentiment_df.to_csv(save_path, index=False)
        print(f"Sentiment data saved to {save_path}")
    
    return sentiment_df


if __name__ == "__main__":
    from stock_data import fetch_stock_data
    
    # Fetch stock data first
    stock_df = fetch_stock_data()
    
    # Fetch sentiment data
    sentiment_df = fetch_sentiment_data(stock_df, use_real_data=True)
    
    print(sentiment_df.head())
    print(f"\nSentiment data shape: {sentiment_df.shape}")
