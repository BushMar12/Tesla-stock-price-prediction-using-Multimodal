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

from config import STREAMLIT_CONFIG, STOCK_SYMBOL, MODELS_DIR, PREDICTION_HORIZONS
from src.data.stock_data import fetch_stock_data, get_latest_data
from src.data.sentiment_data import fetch_sentiment_data, SentimentAnalyzer
from src.features.technical import calculate_all_indicators
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
        
        metadata_path = MODELS_DIR / 'multi_model_metadata.pkl'
        if not metadata_path.exists():
            return None
        
        metadata = joblib.load(metadata_path)
        multi_model = MultiModelRegressor(
            input_size=metadata['input_size'],
            sequence_length=metadata['sequence_length']
        )
        multi_model.load_models(MODELS_DIR)
        return multi_model
    except Exception as e:
        return None


def get_multi_model_predictions(multi_model, df: pd.DataFrame, sentiment_df: pd.DataFrame):
    """Get predictions from all models using returns-based approach"""
    try:
        import joblib
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df, add_targets=False)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        merged = preprocessor.merge_data(df, sentiment_df, df_with_indicators)
        preprocessor.select_features(merged)
        merged = preprocessor.clean_data(merged)
        
        # Load scalers
        feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
        return_scaler = joblib.load(MODELS_DIR / 'return_scaler.pkl')
        
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
    """Make prediction using the model with returns-based approach (single + multi-day)"""
    try:
        from src.features.technical import calculate_all_indicators
        import joblib
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df, add_targets=False)
        
        # Merge with sentiment
        merged = preprocessor.merge_data(df, sentiment_df, df_with_indicators)
        
        # Select features
        preprocessor.select_features(merged)
        
        # Clean and transform
        merged = preprocessor.clean_data(merged)
        
        # Load scalers
        feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
        return_scaler = joblib.load(MODELS_DIR / 'return_scaler.pkl')
        
        # Load multi-day scalers if available
        multi_return_scalers = None
        multi_scaler_path = MODELS_DIR / 'multi_return_scalers.pkl'
        if multi_scaler_path.exists():
            multi_return_scalers = joblib.load(multi_scaler_path)
        
        all_features = preprocessor.feature_columns + preprocessor.sentiment_columns
        merged[all_features] = feature_scaler.transform(merged[all_features])
        
        # Get last sequence
        n_price_features = len(preprocessor.feature_columns)
        sequence_length = preprocessor.sequence_length
        
        X_all = merged[all_features].values
        
        X_price = X_all[-sequence_length:, :n_price_features]
        X_sentiment = X_all[-sequence_length:, n_price_features:]
        
        # Convert to tensors
        X_price = torch.FloatTensor(X_price).unsqueeze(0)
        X_sentiment = torch.FloatTensor(X_sentiment).unsqueeze(0)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(X_price, X_sentiment)
            
            # Classification output
            direction_probs = torch.softmax(outputs['classification'], dim=-1)
            direction = torch.argmax(direction_probs, dim=-1).item()
            
            # Single-day regression output
            predicted_return_scaled = outputs['regression'].item()
            
            # Multi-day regression output
            multi_day_scaled = outputs['multi_regression'].cpu().numpy()[0]
        
        # Inverse transform to actual return
        predicted_return = return_scaler.inverse_transform([[predicted_return_scaled]])[0, 0]
        
        # Get current price and reconstruct predicted price
        current_price = df['close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        predicted_change = predicted_return * 100
        
        # Multi-day forecasts
        multi_day_forecasts = []
        for i, horizon in enumerate(PREDICTION_HORIZONS):
            if multi_return_scalers and horizon in multi_return_scalers:
                ret = multi_return_scalers[horizon].inverse_transform([[multi_day_scaled[i]]])[0, 0]
            else:
                ret = return_scaler.inverse_transform([[multi_day_scaled[i]]])[0, 0]
            price = current_price * (1 + ret)
            multi_day_forecasts.append({
                'horizon': horizon,
                'predicted_price': price,
                'predicted_return': ret,
                'predicted_change': ret * 100
            })
            
        return {
            'direction': direction,
            'direction_probs': direction_probs.numpy()[0],
            'confidence': direction_probs.max().item(),
            'predicted_price': predicted_price,
            'current_price': current_price,
            'predicted_change': predicted_change,
            'predicted_return': predicted_return,
            'multi_day': multi_day_forecasts
        }
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def simulate_prediction(df: pd.DataFrame):
    """Simulate prediction when model is not available"""
    # Use recent momentum for simulation
    recent_returns = df['close'].pct_change().tail(5).mean()
    current_price = df['close'].iloc[-1]
    
    if recent_returns > 0:
        direction = 1  # Up
        probs = [0.3, 0.7]
        predicted_change = np.random.uniform(0.5, 3)
    else:
        direction = 0  # Down
        probs = [0.7, 0.3]
        predicted_change = np.random.uniform(-3, -0.5)
    
    predicted_price = current_price * (1 + predicted_change / 100)
    
    return {
        'direction': direction,
        'direction_probs': np.array(probs),
        'confidence': max(probs),
        'predicted_price': predicted_price,
        'current_price': current_price,
        'predicted_change': predicted_change
    }


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">Tesla Stock Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown("### Multimodal Deep Learning for Stock Price Prediction")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2010, 6, 29)
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
            df = fetch_stock_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                save=False
            )
            df_with_indicators = calculate_all_indicators(df, add_targets=False)
            sentiment_df = fetch_sentiment_data(
                df,
                use_real_data=STREAMLIT_CONFIG["sentiment_use_real_data"],
                save=False,
            )
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Load model
    model, metadata, model_loaded = load_model()
    
    # Load multi-model comparison if available
    multi_model = load_multi_models()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Price Chart", "📊 Technical Analysis", "💭 Sentiment", "🔮 Prediction", "🏆 Model Comparison"]
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
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Technical Indicators")
        
        # Selected indicator chart
        fig = create_technical_chart(df_with_indicators, indicator)
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.subheader("🔮 Stock Price Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            The multimodal model combines:
            - **Time-series data**: Historical prices and technical indicators
            - **Sentiment data**: News and social media sentiment
            - **Deep Learning**: LSTM with attention + Cross-modal fusion
            """)
        
        with col2:
            predict_button = st.button("🎯 Make Prediction", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing data and making prediction..."):
                if model_loaded and model is not None:
                    preprocessor = DataPreprocessor()
                    prediction = make_prediction(model, preprocessor, df, sentiment_df)
                else:
                    prediction = simulate_prediction(df)
                
                if prediction:
                    st.markdown("---")
                    
                    # Price Prediction (Regression)
                    st.subheader("💰 Price Prediction (Regression)")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    current_price = prediction['current_price']
                    predicted_price = prediction['predicted_price']
                    predicted_change = prediction['predicted_change']
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        change_color = "green" if predicted_change > 0 else "red" if predicted_change < 0 else "gray"
                        st.metric(
                            "Predicted Price (Next Day)", 
                            f"${predicted_price:.2f}",
                            f"{predicted_change:+.2f}%"
                        )
                    
                    with col3:
                        price_diff = predicted_price - current_price
                        st.metric(
                            "Expected Change",
                            f"${abs(price_diff):.2f}",
                            "Gain" if price_diff > 0 else "Loss" if price_diff < 0 else "No Change"
                        )
                    
                    # Multi-Day Forecast
                    if 'multi_day' in prediction and prediction['multi_day']:
                        st.markdown("---")
                        st.subheader("📅 Multi-Day Forecast")
                        
                        # Display as metric columns
                        forecast_cols = st.columns(len(prediction['multi_day']))
                        for i, f in enumerate(prediction['multi_day']):
                            with forecast_cols[i]:
                                st.metric(
                                    f"{f['horizon']}-Day Ahead",
                                    f"${f['predicted_price']:.2f}",
                                    f"{f['predicted_change']:+.2f}%"
                                )
                        
                        # Multi-day forecast line chart
                        forecast_dates = [0] + [f['horizon'] for f in prediction['multi_day']]
                        forecast_prices = [current_price] + [f['predicted_price'] for f in prediction['multi_day']]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast_dates, y=forecast_prices,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=10)
                        ))
                        fig.add_hline(y=current_price, line_dash="dash", line_color="gray",
                                      annotation_text="Current Price")
                        fig.update_layout(
                            title='Multi-Day Price Forecast',
                            xaxis_title='Days Ahead',
                            yaxis_title='Predicted Price ($)',
                            height=350,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Direction Prediction (Classification)
                    st.subheader("📊 Direction Prediction (Classification)")
                    
                    direction_labels = ['📉 DOWN', '📈 UP']
                    
                    direction = prediction['direction']
                    confidence = prediction['confidence']
                    probs = prediction['direction_probs']
                    
                    # Display prediction
                    st.markdown(f"### Predicted Direction: **{direction_labels[direction]}**")
                    st.markdown(f"Confidence: **{confidence*100:.1f}%**")
                    
                    # Probability bars
                    st.subheader("Direction Probabilities")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Down", f"{probs[0]*100:.1f}%")
                        st.progress(float(probs[0]))
                    
                    with col2:
                        st.metric("Up", f"{probs[1]*100:.1f}%")
                        st.progress(float(probs[1]))
                    
                    # Warning
                    st.warning("""
                    ⚠️ **Disclaimer**: This prediction is for educational purposes only. 
                    Stock markets are inherently unpredictable, and this tool should not 
                    be used for actual trading decisions.
                    """)
        
        # Model info
        with st.expander("Model Architecture"):
            st.markdown("""
            **Multimodal Fusion Architecture:**
            
            1. **Time-Series Encoder** (Bidirectional LSTM)
               - Input: 60-day sequence of price & technical features
               - Attention mechanism for important time steps
               
            2. **Sentiment Encoder** (Temporal CNN)
               - Input: 60-day sequence of sentiment features
               - Multi-scale convolutions (3, 5, 7 kernels)
               
            3. **Cross-Modal Attention**
               - Learns relationships between price patterns and sentiment
               
            4. **Fusion Layer**
               - Combines encoded representations
               - Dual output heads for regression & classification
            """)
    
    with tab5:
        st.subheader("🏆 Model Comparison: LSTM vs GRU vs Transformer vs XGBoost")
        
        if multi_model is None:
            st.warning(
                "Multi-model comparison not available. Run `python model_comparison.py` first to train LSTM, GRU, and XGBoost baselines."
            )
            st.code("python model_comparison.py", language="bash")
        else:
            st.success("All models loaded successfully!")
            
            # Display saved metrics if available
            comparison_file = MODELS_DIR / 'model_comparison.csv'
            if comparison_file.exists():
                st.subheader("📊 Test Set Performance Metrics")
                comparison_df = pd.read_csv(comparison_file)
                
                # Style the dataframe
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Create bar chart for comparison
                if multi_model.metrics:
                    metrics_data = []
                    for model_name, metrics in multi_model.metrics.items():
                        metrics_data.append({
                            'Model': model_name,
                            'RMSE': metrics['RMSE'],
                            'MAE': metrics['MAE']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='RMSE',
                        x=metrics_df['Model'],
                        y=metrics_df['RMSE'],
                        marker_color='indianred'
                    ))
                    fig.add_trace(go.Bar(
                        name='MAE',
                        x=metrics_df['Model'],
                        y=metrics_df['MAE'],
                        marker_color='lightsalmon'
                    ))
                    fig.update_layout(
                        title='Model Error Comparison',
                        barmode='group',
                        yaxis_title='Error ($)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Live predictions from all models
            st.subheader("🔮 Live Predictions from All Models")
            
            predict_btn = st.button("🎯 Get Predictions from All Models", use_container_width=True)
            
            if predict_btn:
                with st.spinner("Getting predictions from LSTM, GRU, and XGBoost..."):
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
                                'Direction': '📈 Up' if change > 0 else '📉 Down' if change < 0 else '➡️ Neutral'
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
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        # Calculate ensemble prediction (average)
                        ensemble_price = np.mean(list(predictions.values()))
                        ensemble_change = ((ensemble_price - current_price) / current_price) * 100
                        
                        st.markdown("---")
                        st.subheader("🎯 Ensemble Prediction (Average)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Ensemble Prediction", f"${ensemble_price:.2f}", f"{ensemble_change:+.2f}%")
                        with col3:
                            direction = '📈 Up' if ensemble_change > 0.5 else '📉 Down' if ensemble_change < -0.5 else '➡️ Neutral'
                            st.metric("Direction", direction)
            
            # Model descriptions
            with st.expander("📚 Model Descriptions"):
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
