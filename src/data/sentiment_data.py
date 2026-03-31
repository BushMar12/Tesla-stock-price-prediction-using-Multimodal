"""
Sentiment data collection and analysis
Uses free sources: RSS feeds, VADER sentiment, and optional FinBERT
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import STOCK_SYMBOL, RAW_DATA_DIR, SENTIMENT_CONFIG


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
    Generate synthetic sentiment data based on price movements.
    This is used when real sentiment data is not available.
    
    Args:
        stock_df: DataFrame with stock price data
    
    Returns:
        DataFrame with synthetic sentiment scores
    """
    df = stock_df.copy()
    
    # Generate sentiment based on returns with some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(df))
    
    # Compound sentiment correlates with returns
    df['sentiment_compound'] = np.tanh(df['returns'] * 10 + noise)
    
    # Decompose into positive/negative/neutral
    df['sentiment_positive'] = np.clip(df['sentiment_compound'] + 0.5, 0, 1) / 2
    df['sentiment_negative'] = np.clip(-df['sentiment_compound'] + 0.5, 0, 1) / 2
    df['sentiment_neutral'] = 1 - df['sentiment_positive'] - df['sentiment_negative']
    df['news_count'] = np.random.randint(1, 20, len(df))
    
    # Add some lagged effects
    for lag in [1, 2, 3]:
        df[f'sentiment_compound_lag{lag}'] = df['sentiment_compound'].shift(lag)
    
    return df[['date', 'sentiment_compound', 'sentiment_positive', 
               'sentiment_negative', 'sentiment_neutral', 'news_count',
               'sentiment_compound_lag1', 'sentiment_compound_lag2', 
               'sentiment_compound_lag3']].dropna()


def fetch_sentiment_data(
    stock_df: pd.DataFrame,
    use_real_data: bool = True,
    save: bool = True
) -> pd.DataFrame:
    """
    Fetch or generate sentiment data.
    
    Args:
        stock_df: DataFrame with stock data (for dates and synthetic generation)
        use_real_data: Whether to try fetching real news data
        save: Whether to save the data to disk
    
    Returns:
        DataFrame with sentiment data
    """
    if use_real_data:
        try:
            print("Fetching news headlines...")
            collector = NewsCollector()
            headlines = collector.fetch_rss_headlines()
            
            if headlines:
                sentiment_df = collector.analyze_headlines(headlines)
                daily_sentiment = collector.get_daily_sentiment(sentiment_df)
                
                # Merge with stock dates
                stock_dates = pd.DataFrame({'date': stock_df['date'].dt.date})
                merged = stock_dates.merge(daily_sentiment, on='date', how='left')
                
                # Fill missing days with neutral sentiment
                merged = merged.fillna({
                    'sentiment_compound': 0,
                    'sentiment_positive': 0.33,
                    'sentiment_negative': 0.33,
                    'sentiment_neutral': 0.34,
                    'news_count': 0
                })
                
                print(f"Fetched sentiment for {len(merged)} days")
                
                if save:
                    save_path = RAW_DATA_DIR / "sentiment_data.csv"
                    merged.to_csv(save_path, index=False)
                
                return merged
        except Exception as e:
            print(f"Error fetching real sentiment data: {e}")
            print("Falling back to synthetic sentiment...")
    
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
