"""
Tesla Stock Price Prediction - Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import STREAMLIT_CONFIG, STOCK_SYMBOL, MODELS_DIR
from src.data.stock_data import fetch_stock_data, get_latest_data, load_stock_data
from src.data.sentiment_data import fetch_sentiment_data, SentimentAnalyzer
from src.features.technical import calculate_all_indicators, add_macd, add_stochastic
from src.data.preprocessing import DataPreprocessor


# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout']
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-up {
        color: #00c853;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff1744;
        font-weight: bold;
    }
    .prediction-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """Load the trained model"""
    try:
        from src.utils.helpers import load_trained_model
        model, metadata = load_trained_model()
        return model, metadata, True
    except Exception as e:
        st.warning(f"Could not load trained model: {e}")
        st.info("Running in demo mode with simulated predictions.")
        return None, None, False


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Create an interactive candlestick chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('TSLA Price', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['SMA_20'], 
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['SMA_50'], 
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig


def calculate_dashboard_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate model features plus display-only indicators used by the UI."""
    indicator_df = calculate_all_indicators(df, add_targets=False)
    indicator_df = add_macd(indicator_df)
    indicator_df = add_stochastic(indicator_df)
    return indicator_df


def create_technical_chart(df: pd.DataFrame, indicator: str) -> go.Figure:
    """Create chart for technical indicator"""
    fig = go.Figure()
    
    if indicator == 'RSI':
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['RSI_14'], name='RSI 14',
                      line=dict(color='purple'))
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title='RSI (14)', yaxis_range=[0, 100])
    
    elif indicator == 'MACD':
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MACD'], name='MACD',
                      line=dict(color='blue'))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MACD_Signal'], name='Signal',
                      line=dict(color='orange'))
        )
        fig.add_trace(
            go.Bar(x=df['date'], y=df['MACD_Hist'], name='Histogram',
                  marker_color='gray', opacity=0.5)
        )
        fig.update_layout(title='MACD')
    
    elif indicator == 'Bollinger Bands':
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['close'], name='Close',
                      line=dict(color='blue'))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['BB_Upper'], name='Upper Band',
                      line=dict(color='red', dash='dash'))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['BB_Lower'], name='Lower Band',
                      line=dict(color='green', dash='dash'),
                      fill='tonexty', fillcolor='rgba(0,100,80,0.1)')
        )
        fig.update_layout(title='Bollinger Bands')
    
    elif indicator == 'Stochastic':
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['Stoch_K'], name='%K',
                      line=dict(color='blue'))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['Stoch_D'], name='%D',
                      line=dict(color='orange'))
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.add_hline(y=20, line_dash="dash", line_color="green")
        fig.update_layout(title='Stochastic Oscillator', yaxis_range=[0, 100])
    
    fig.update_layout(height=300, template='plotly_white')
    return fig


def create_sentiment_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """Create sentiment visualization"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=sentiment_df['date'],
            y=sentiment_df['sentiment_compound'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='purple'),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Sentiment Over Time',
        yaxis_title='Compound Sentiment',
        yaxis_range=[-1, 1],
        height=300,
        template='plotly_white'
    )
    
    return fig


def load_multi_models():
    """Load multi-model comparison models"""
    try:
        from src.models.regression_models import MultiModelRegressor
        import joblib

        comparison_dir = MODELS_DIR / 'comparison_technical_sentiment'
        model_dir = (
            comparison_dir
            if (comparison_dir / 'multi_model_metadata.pkl').exists()
            else MODELS_DIR
        )

        metadata_path = model_dir / 'multi_model_metadata.pkl'
        if not metadata_path.exists():
            return None
        
        metadata = joblib.load(metadata_path)
        multi_model = MultiModelRegressor(
            input_size=metadata['input_size'],
            sequence_length=metadata['sequence_length']
        )
        # Do not load the pickled XGBoost artifact in Streamlit. If the
        # installed xgboost native library differs from the one used when the
        # pickle was created, unpickling can segfault the Python process.
        multi_model.load_models(model_dir, load_xgboost=False)
        multi_model.artifact_dir = model_dir
        return multi_model
    except Exception as e:
        return None


def get_multi_model_predictions(multi_model, df: pd.DataFrame, sentiment_df: pd.DataFrame):
    """Get predictions from all models using returns-based approach"""
    try:
        import joblib

        artifact_dir = getattr(multi_model, 'artifact_dir', MODELS_DIR)
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df, add_targets=False)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        merged = preprocessor.merge_data(df, sentiment_df, df_with_indicators)
        
        # Load metadata to get exactly the features the model was trained on
        metadata = joblib.load(artifact_dir / 'preprocessing_metadata.pkl')
        preprocessor.feature_columns = metadata['feature_columns']
        preprocessor.sentiment_columns = metadata['sentiment_columns']
        preprocessor.sequence_length = metadata['sequence_length']
        
        # Ensure all required features exist in merged
        all_required = preprocessor.feature_columns + preprocessor.sentiment_columns
        for col in all_required:
            if col not in merged.columns:
                merged[col] = 0.0
                
        merged = preprocessor.clean_data(merged)
        
        # Load scalers
        feature_scaler = joblib.load(artifact_dir / 'feature_scaler.pkl')
        return_scaler = joblib.load(artifact_dir / 'return_scaler.pkl')
        
        all_features = preprocessor.feature_columns + preprocessor.sentiment_columns
        merged[all_features] = feature_scaler.transform(merged[all_features])
        
        # Get last sequence
        sequence_length = preprocessor.sequence_length
        X_all = merged[all_features].values
        X = X_all[-sequence_length:].reshape(1, sequence_length, -1)
        
        # Get current (today's) close price for reconstructing predicted price
        current_price = df['close'].iloc[-1]
        
        # Get predictions from all models (returns predicted prices)
        predictions = multi_model.predict_all(
            X, 
            return_scaler=return_scaler,
            current_price=current_price
        )
        
        return predictions, current_price
    except Exception as e:
        st.error(f"Multi-model prediction error: {e}")
        return None, None


def make_prediction(model, preprocessor, df: pd.DataFrame, sentiment_df: pd.DataFrame):
    """Make next-day prediction using the model with returns-based approach."""
    try:
        from src.features.technical import calculate_all_indicators
        import joblib

        df_with_indicators = calculate_all_indicators(df, add_targets=False)
        merged = preprocessor.merge_data(df, sentiment_df, df_with_indicators)

        metadata = joblib.load(MODELS_DIR / 'preprocessing_metadata.pkl')
        preprocessor.feature_columns = metadata['feature_columns']
        preprocessor.sentiment_columns = metadata['sentiment_columns']
        preprocessor.sequence_length = metadata['sequence_length']

        all_required = preprocessor.feature_columns + preprocessor.sentiment_columns
        for col in all_required:
            if col not in merged.columns:
                merged[col] = 0.0

        merged = preprocessor.clean_data(merged)

        feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
        return_scaler = joblib.load(MODELS_DIR / 'return_scaler.pkl')

        all_features = preprocessor.feature_columns + preprocessor.sentiment_columns
        merged[all_features] = feature_scaler.transform(merged[all_features])

        n_price_features = len(preprocessor.feature_columns)
        sequence_length = preprocessor.sequence_length

        X_all = merged[all_features].values
        X_price = X_all[-sequence_length:, :n_price_features]
        X_sentiment = X_all[-sequence_length:, n_price_features:]

        X_price = torch.FloatTensor(X_price).unsqueeze(0)
        X_sentiment = torch.FloatTensor(X_sentiment).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(X_price, X_sentiment)
            predicted_return_scaled = outputs['regression'].item()

        predicted_return = return_scaler.inverse_transform([[predicted_return_scaled]])[0, 0]
        current_price = df['close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        predicted_change = predicted_return * 100

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'predicted_change': predicted_change,
            'predicted_return': predicted_return,
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def simulate_prediction(df: pd.DataFrame):
    """Simulate prediction when model is not available."""
    recent_returns = df['close'].pct_change().tail(5).mean()
    current_price = df['close'].iloc[-1]
    predicted_change = float(np.clip(recent_returns * 100, -3, 3))
    predicted_price = current_price * (1 + predicted_change / 100)

    return {
        'predicted_price': predicted_price,
        'current_price': current_price,
        'predicted_change': predicted_change,
        'predicted_return': predicted_change / 100,
    }


def load_dashboard_stock_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load stock data for the dashboard, falling back to the cached CSV."""
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    try:
        df = fetch_stock_data(start_date=start, end_date=end, save=False)
        if df.empty:
            raise ValueError("Live stock data fetch returned no rows.")
    except Exception as fetch_error:
        st.warning(f"Live stock data fetch failed; using cached data instead. Details: {fetch_error}")
        df = load_stock_data()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

    if df.empty:
        raise ValueError("No stock rows are available for the selected date range.")
    if len(df) < 2:
        raise ValueError("Select a wider date range with at least two trading days.")

    return df


def get_comparison_metrics_path(multi_model) -> Path:
    """Return the saved evaluation CSV matching the loaded comparison artifact."""
    artifact_name = getattr(multi_model, 'artifact_dir', Path()).name

    if artifact_name == 'comparison_technical_sentiment':
        result_path = MODELS_DIR / 'result' / 'technical_sentiment' / 'evaluation_results.csv'
    elif artifact_name == 'comparison_technical':
        result_path = MODELS_DIR / 'result' / 'technical' / 'evaluation_results.csv'
    else:
        result_path = MODELS_DIR / 'model_comparison.csv'

    return result_path if result_path.exists() else MODELS_DIR / 'model_comparison.csv'


def load_model_comparison_results() -> pd.DataFrame:
    """Load saved technical and technical+sentiment model-comparison results."""
    result_paths = [
        MODELS_DIR / 'result' / 'technical' / 'evaluation_results.csv',
        MODELS_DIR / 'result' / 'technical_sentiment' / 'evaluation_results.csv',
    ]

    frames = []
    for path in result_paths:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        fallback = MODELS_DIR / 'model_comparison.csv'
        return pd.read_csv(fallback) if fallback.exists() else pd.DataFrame()

    comparison_df = pd.concat(frames, ignore_index=True)
    comparison_df['Configuration'] = comparison_df['Configuration'].replace({
        'technical': 'Technical',
        'technical_sentiment': 'Technical + Sentiment',
    })
    return comparison_df


def create_comparison_metric_chart(comparison_df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a grouped chart comparing model metrics across configurations."""
    fig = px.bar(
        comparison_df,
        x='Model',
        y=metric,
        color='Configuration',
        barmode='group',
        title=f'{metric} by Model and Feature Configuration',
        labels={metric: metric, 'Model': 'Model', 'Configuration': 'Configuration'},
    )
    fig.update_layout(height=430, template='plotly_white')
    return fig


def load_shap_feature_importance() -> pd.DataFrame | None:
    """Load saved SHAP feature-importance rankings, if available."""
    shap_path = MODELS_DIR / 'shap_feature_importance.csv'
    if not shap_path.exists():
        return None

    shap_df = pd.read_csv(shap_path)
    required_columns = {'feature', 'stream', 'modality', 'mean_abs_shap'}
    if not required_columns.issubset(shap_df.columns):
        missing = sorted(required_columns - set(shap_df.columns))
        raise ValueError(f"SHAP feature importance file is missing columns: {missing}")

    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['rank'] = np.arange(1, len(shap_df) + 1)
    total_importance = shap_df['mean_abs_shap'].sum()
    shap_df['importance_pct'] = (
        shap_df['mean_abs_shap'] / total_importance * 100 if total_importance else 0
    )
    return shap_df


def create_shap_importance_chart(shap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create a horizontal SHAP feature-importance chart."""
    top_features = shap_df.head(top_n).sort_values('mean_abs_shap', ascending=True)
    fig = px.bar(
        top_features,
        x='mean_abs_shap',
        y='feature',
        color='modality',
        orientation='h',
        title=f'Top {min(top_n, len(shap_df))} Features by Mean Absolute SHAP Value',
        labels={
            'mean_abs_shap': 'Mean |SHAP value|',
            'feature': 'Feature',
            'modality': 'Modality',
        },
    )
    fig.update_layout(height=520, template='plotly_white')
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">Tesla Stock Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown("### Multimodal Deep Learning for Stock Price Prediction")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2021, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )
    
    # Technical indicator selection
    indicator = st.sidebar.selectbox(
        "Technical Indicator",
        ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic']
    )
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_dashboard_stock_data(start_date, end_date)
            df_with_indicators = calculate_dashboard_indicators(df)
            sentiment_df = fetch_sentiment_data(
                df,
                use_real_data=STREAMLIT_CONFIG.get("sentiment_source") != "synthetic",
                save=False,
                source=STREAMLIT_CONFIG.get("sentiment_source", "synthetic"),
            )
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Load model
    model, metadata, model_loaded = load_model()
    
    # Load multi-model comparison if available
    multi_model = load_multi_models()
    
    # Main content
    tab1, tab2, tab3, tab4, tab6, tab7 = st.tabs(
        ["Price Chart", "Technical Analysis", "Sentiment", "Next-Day Prediction", "Model Comparison", "SHAP Feature Importance"]
    )
    
    with tab1:
        st.subheader("TSLA Stock Price")
        
        # Key metrics
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = latest['close'] - prev['close']
        change_pct = (change / prev['close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Close Price", f"${latest['close']:.2f}", 
                     f"{change:+.2f} ({change_pct:+.2f}%)")
        with col2:
            st.metric("Open", f"${latest['open']:.2f}")
        with col3:
            st.metric("High", f"${latest['high']:.2f}")
        with col4:
            st.metric("Low", f"${latest['low']:.2f}")
        
        # Price chart
        fig = create_price_chart(df_with_indicators)
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Technical Indicators")
        
        # Selected indicator chart
        fig = create_technical_chart(df_with_indicators, indicator)
        st.plotly_chart(fig, width='stretch')
        
        # Technical summary
        st.subheader("Technical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = df_with_indicators['RSI_14'].iloc[-1]
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal)
        
        with col2:
            macd = df_with_indicators['MACD'].iloc[-1]
            signal = df_with_indicators['MACD_Signal'].iloc[-1]
            macd_signal = "Bullish" if macd > signal else "Bearish"
            st.metric("MACD", f"{macd:.2f}", macd_signal)
        
        with col3:
            bb_pos = df_with_indicators['BB_Position'].iloc[-1]
            bb_signal = "High" if bb_pos > 0.8 else "Low" if bb_pos < 0.2 else "Middle"
            st.metric("BB Position", f"{bb_pos:.2f}", bb_signal)
        
        with col4:
            stoch_k = df_with_indicators['Stoch_K'].iloc[-1]
            stoch_signal = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
            st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_signal)
    
    with tab3:
        st.subheader("Market Sentiment Analysis")
        
        # Sentiment chart
        fig = create_sentiment_chart(sentiment_df)
        st.plotly_chart(fig, width='stretch')
        
        # Current sentiment
        current_sentiment = sentiment_df['sentiment_compound'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_label = "Positive" if current_sentiment > 0.1 else "Negative" if current_sentiment < -0.1 else "Neutral"
            sentiment_color = "green" if current_sentiment > 0.1 else "red" if current_sentiment < -0.1 else "gray"
            st.metric("Current Sentiment", sentiment_label, f"{current_sentiment:.3f}")
        
        with col2:
            avg_sentiment = sentiment_df['sentiment_compound'].tail(7).mean()
            st.metric("7-Day Avg Sentiment", f"{avg_sentiment:.3f}")
        
        with col3:
            sentiment_trend = sentiment_df['sentiment_compound'].diff().tail(7).mean()
            trend_label = "Improving" if sentiment_trend > 0 else "Declining"
            st.metric("Sentiment Trend", trend_label, f"{sentiment_trend:+.4f}")
        
        # Sentiment explanation
        st.info("""
        **Sentiment Analysis** uses natural language processing to analyze news headlines 
        and social media mentions about Tesla. The compound score ranges from -1 (very negative) 
        to +1 (very positive). This multimodal approach combines sentiment with price data 
        for more accurate predictions.
        """)
    
    with tab4:
        st.subheader("Stock Price Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            The multimodal model combines:
            - **Time-series data**: Historical prices and technical indicators
            - **Sentiment data**: News and social media sentiment
            - **Deep Learning**: Transformer sequence encoder + temporal sentiment fusion
            """)
        
        with col2:
            predict_button = st.button("Make Next-Day Prediction", width='stretch')
        
        if predict_button:
            with st.spinner("Analyzing data and making prediction..."):
                if model_loaded and model is not None:
                    preprocessor = DataPreprocessor()
                    prediction = make_prediction(model, preprocessor, df, sentiment_df)
                else:
                    prediction = simulate_prediction(df)
                st.session_state['prediction'] = prediction
                
        if 'prediction' in st.session_state and st.session_state['prediction']:
            prediction = st.session_state['prediction']
            st.markdown("---")
            
            # Price Prediction (Regression)
            st.subheader("Price Prediction (Regression)")
            
            col1, col2, col3 = st.columns(3)
            
            current_price = prediction['current_price']
            predicted_price = prediction['predicted_price']
            predicted_change = prediction['predicted_change']
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric(
                    "Predicted Price (Next Day)", 
                    f"${predicted_price:.2f}",
                    f"{predicted_change:+.2f}%"
                )
            
            with col3:
                price_diff = predicted_price - current_price
                change_label = "Gain" if price_diff > 0 else "Loss" if price_diff < 0 else "No Change"
                st.metric(
                    "Expected Change",
                    f"{change_label}: ${abs(price_diff):.2f}",
                    f"{price_diff:+.2f} ({predicted_change:+.2f}%)",
                    delta_color="normal"
                )
            
            st.warning("""
            **Disclaimer**: This prediction is for educational purposes only. 
            Stock markets are inherently unpredictable, and this tool should not 
            be used for actual trading decisions.
            """)
        
        # Model info
        with st.expander("Model Architecture"):
            st.markdown("""
            **Multimodal Fusion Architecture:**
            
            1. **Time-Series Encoder** (Transformer)
               - Input: 60-day sequence of price & technical features
               - Input projection, positional encoding, and Transformer encoder layers
               - Attention pooling highlights important time steps
               
            2. **Sentiment Encoder** (Temporal CNN)
               - Input: 60-day sequence of sentiment features
               - Multi-scale convolutions (3, 5, 7 kernels)
               
            3. **Cross-Modal Multi-Head Attention**
               - Lets the time-series stream attend to sentiment context
               - Combines price patterns with market sentiment signals
               
            4. **Fusion MLP**
               - Combines encoded representations
               - Next-day return regression head
            """)

    with tab6:
        st.subheader("Model Comparison: LSTM vs GRU vs Transformer vs XGBoost")
        
        if multi_model is None:
            st.warning(
                "Multi-model comparison not available. Run `python model_comparison.py` first to train LSTM, GRU, Transformer, and XGBoost baselines."
            )
            st.code("python model_comparison.py", language="bash")
        else:
            live_model_names = [name for name, trained in multi_model.trained.items() if trained]
            st.success(f"Live models loaded: {', '.join(live_model_names)}")
            if not multi_model.trained.get('XGBoost', False):
                st.info(
                    "XGBoost metrics are shown from saved results, but live XGBoost prediction is disabled in Streamlit to avoid native-library pickle crashes."
                )
            
            comparison_df = load_model_comparison_results()
            if not comparison_df.empty:
                st.subheader("Test Set Performance Metrics")
                st.dataframe(
                    comparison_df,
                    width='stretch',
                    hide_index=True
                )

                metric_options = [
                    metric for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
                    if metric in comparison_df.columns
                ]
                selected_metric = st.selectbox(
                    "Metric",
                    metric_options,
                    index=metric_options.index('MAE') if 'MAE' in metric_options else 0,
                )
                fig = create_comparison_metric_chart(comparison_df, selected_metric)
                st.plotly_chart(fig, width='stretch')

                if {'Technical', 'Technical + Sentiment'}.issubset(set(comparison_df['Configuration'])):
                    st.subheader("Technical + Sentiment Delta vs Technical")
                    delta_metrics = [
                        metric for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
                        if metric in comparison_df.columns
                    ]
                    delta_df = comparison_df.pivot(
                        index='Model',
                        columns='Configuration',
                        values=delta_metrics,
                    )
                    delta_rows = []
                    for model_name in delta_df.index:
                        row = {'Model': model_name}
                        for metric in delta_metrics:
                            row[f'{metric} Delta'] = (
                                delta_df.loc[model_name, (metric, 'Technical + Sentiment')]
                                - delta_df.loc[model_name, (metric, 'Technical')]
                            )
                        delta_rows.append(row)
                    st.dataframe(pd.DataFrame(delta_rows), width='stretch', hide_index=True)
            else:
                st.warning("Saved model-comparison metrics were not found.")
            
            st.markdown("---")
            
            # Live predictions from all models
            st.subheader("Live Predictions from All Models")
            
            predict_btn = st.button("Get Predictions from All Models", width='stretch')
            
            if predict_btn:
                with st.spinner(f"Getting predictions from {', '.join(live_model_names)}..."):
                    predictions, current_price = get_multi_model_predictions(
                        multi_model, df, sentiment_df
                    )
                    
                    if predictions and current_price:
                        st.markdown(f"**Current Price:** ${current_price:.2f}")
                        st.markdown("---")
                        
                        # Create prediction comparison table
                        pred_data = []
                        for model_name, pred_price in predictions.items():
                            change = ((pred_price - current_price) / current_price) * 100
                            pred_data.append({
                                'Model': model_name,
                                'Predicted Price': f"${pred_price:.2f}",
                                'Change': f"{change:+.2f}%",
                                'Direction': 'Up' if change > 0 else 'Down' if change < 0 else 'Neutral'
                            })
                        
                        pred_df = pd.DataFrame(pred_data)
                        
                        # Display as columns
                        cols = st.columns(len(predictions))
                        for i, (model_name, pred_price) in enumerate(predictions.items()):
                            change = ((pred_price - current_price) / current_price) * 100
                            with cols[i]:
                                st.metric(
                                    model_name,
                                    f"${pred_price:.2f}",
                                    f"{change:+.2f}%"
                                )
                        
                        st.markdown("---")
                        st.subheader("Prediction Summary Table")
                        st.dataframe(pred_df, width='stretch', hide_index=True)
                        
                        # Calculate ensemble prediction (average)
                        ensemble_price = np.mean(list(predictions.values()))
                        ensemble_change = ((ensemble_price - current_price) / current_price) * 100
                        
                        st.markdown("---")
                        st.subheader("Ensemble Prediction (Average)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Ensemble Prediction", f"${ensemble_price:.2f}", f"{ensemble_change:+.2f}%")
                        with col3:
                            direction = 'Up' if ensemble_change > 0.5 else 'Down' if ensemble_change < -0.5 else 'Neutral'
                            st.metric("Direction", direction)
            
            # Model descriptions
            with st.expander("Model Descriptions"):
                st.markdown("""
                | Model | Description | Strengths |
                |-------|-------------|-----------|
                | **LSTM** | Long Short-Term Memory neural network | Captures long-term dependencies, handles vanishing gradients |
                | **GRU** | Gated Recurrent Unit neural network | Faster training, fewer parameters than LSTM |
                | **Transformer** | Self-attention encoder with positional encoding | Captures long-range patterns, parallelizable |
                | **XGBoost** | Gradient Boosting ensemble method | Handles non-linear patterns, robust to outliers |
                
                All models use the same input features:
                - 60-day price history with technical indicators + market context (SPY, VIX)
                - Sentiment features from news analysis
                - Calendar features (day of week, month, quarter-end)
                """)
    
    with tab7:
        st.subheader("SHAP Feature Importance")

        try:
            shap_df = load_shap_feature_importance()
        except Exception as e:
            st.error(f"Could not load SHAP feature importance: {e}")
            shap_df = None

        if shap_df is None:
            st.warning("SHAP feature-importance artifacts are not available yet.")
            st.code(".venv/bin/python scripts/shap_feature_importance.py", language="bash")
        else:
            top_n = st.slider(
                "Number of features to display",
                min_value=5,
                max_value=min(25, len(shap_df)),
                value=min(15, len(shap_df)),
            )

            fig = create_shap_importance_chart(shap_df, top_n=top_n)
            st.plotly_chart(fig, width='stretch')

            modality_df = (
                shap_df.groupby('modality', as_index=False)
                .agg(mean_abs_shap=('mean_abs_shap', 'sum'))
                .sort_values('mean_abs_shap', ascending=False)
            )
            total_modality_importance = modality_df['mean_abs_shap'].sum()
            modality_df['importance_pct'] = (
                modality_df['mean_abs_shap'] / total_modality_importance * 100
                if total_modality_importance else 0
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Features**")
                st.dataframe(
                    shap_df[['rank', 'feature', 'stream', 'modality', 'mean_abs_shap', 'importance_pct']].head(top_n),
                    width='stretch',
                    hide_index=True,
                )

            with col2:
                st.markdown("**Importance by Modality**")
                st.dataframe(
                    modality_df,
                    width='stretch',
                    hide_index=True,
                )

            st.markdown("---")
            image_col1, image_col2 = st.columns(2)
            shap_bar_path = MODELS_DIR / 'shap_feature_importance.png'
            shap_heatmap_path = MODELS_DIR / 'shap_temporal_heatmap.png'

            with image_col1:
                if shap_bar_path.exists():
                    st.image(str(shap_bar_path), caption="Saved SHAP feature-importance chart")
                else:
                    st.info("Saved SHAP feature-importance image not found.")

            with image_col2:
                if shap_heatmap_path.exists():
                    st.image(str(shap_heatmap_path), caption="Saved SHAP temporal heatmap")
                else:
                    st.info("Saved SHAP temporal heatmap not found.")

            st.info(
                "SHAP values explain the trained next-day return model. Higher mean absolute SHAP values indicate features that had a larger average influence on model output magnitude."
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit | Data from Yahoo Finance | "
        "Neural Networks & Fuzzy Logic Project"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
