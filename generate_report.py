"""
Generate a comprehensive PDF report for the Tesla Stock Price Prediction project.
"""
import sys
import subprocess

# Ensure fpdf2 is installed
try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

from pathlib import Path
import os


class ProjectReport(FPDF):
    """Custom PDF class with headers and footers"""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Tesla Stock Price Prediction Using Multimodal Deep Learning", align="C")
        self.ln(4)
        self.set_draw_color(41, 82, 152)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── helpers ──────────────────────────────────────────────
    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(41, 82, 152)
        self.cell(0, 12, f"{num}.  {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(60, 60, 60)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def sub_sub_title(self, title):
        self.set_font("Helvetica", "BI", 11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(indent)
        self.cell(5, 6, "-")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # header row
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(41, 82, 152)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(230, 238, 248)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, str(val), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(4)


def build_report(output_path: str):
    pdf = ProjectReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ══════════════════════════════════════════════════════════
    #  COVER PAGE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 82, 152)
    pdf.multi_cell(0, 14, "Tesla Stock Price Prediction\nUsing Multimodal Deep Learning", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "49275 Neural Networks and Fuzzy Logic", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "University of Technology Sydney (UTS)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 8, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_draw_color(41, 82, 152)
    pdf.set_line_width(1)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())

    # ══════════════════════════════════════════════════════════
    #  TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(41, 82, 152)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    toc = [
        ("1", "Executive Summary"),
        ("2", "Introduction & Motivation"),
        ("3", "Data Sources & Feature Engineering"),
        ("4", "Model Architectures"),
        ("5", "Training Methodology"),
        ("6", "Multi-Day Prediction"),
        ("7", "Evaluation Metrics & Baseline Results"),
        ("8", "System Architecture & Pipeline"),
        ("9", "Technology Stack"),
        ("10", "Streamlit Dashboard"),
        ("11", "Conclusion & Future Work"),
        ("12", "References"),
    ]
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(30, 30, 30)
    for num, title in toc:
        pdf.cell(10, 8, num + ".")
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════
    #  1. EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("1", "Executive Summary")
    pdf.body_text(
        "This project implements a comprehensive stock price prediction system for Tesla (TSLA) "
        "that combines time-series analysis with textual sentiment data using a multimodal deep "
        "learning fusion model built in PyTorch. Rather than predicting raw prices directly, the "
        "system adopts a returns-based approach: it predicts daily percentage returns and "
        "reconstructs actual prices via the formula  predicted_price = close * (1 + predicted_return)."
    )
    pdf.body_text(
        "The project delivers four model variants for comparison: a Multimodal Fusion model "
        "(BiLSTM + Temporal CNN + Cross-Modal Multi-Head Attention), standalone Bidirectional LSTM, "
        "standalone Bidirectional GRU, a Transformer Encoder baseline, and an XGBoost gradient-boosting "
        "regressor. All models are evaluated on both regression accuracy (RMSE, MAE, MAPE) and "
        "directional classification accuracy (Up/Down)."
    )
    pdf.body_text(
        "Multi-day prediction is natively supported: the fusion model simultaneously predicts returns "
        "for 1-day, 3-day, 5-day, and 7-day horizons using a dedicated multi-output regression head. "
        "An interactive Streamlit dashboard provides real-time visualization of predictions, technical "
        "indicators, sentiment analysis, and model comparison metrics."
    )

    # ══════════════════════════════════════════════════════════
    #  2. INTRODUCTION & MOTIVATION
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("2", "Introduction & Motivation")
    pdf.body_text(
        "Stock price prediction is one of the most challenging tasks in financial machine learning "
        "due to the inherently noisy, non-stationary, and multi-factor nature of financial markets. "
        "Traditional models rely exclusively on historical price data, but recent research has shown "
        "that incorporating alternative data sources such as news sentiment significantly improves "
        "predictive performance."
    )
    pdf.body_text(
        "Tesla (TSLA) is chosen as the target security due to its high volatility, strong retail "
        "investor interest, and heavy media coverage, making it an ideal candidate for sentiment-"
        "augmented prediction. The stock has experienced extreme price swings driven as much by "
        "social media discussion and CEO statements as by fundamental financial metrics."
    )
    pdf.body_text(
        "This project explores whether a multimodal architecture, one that jointly learns from "
        "time-series price patterns and textual sentiment signals, can outperform unimodal baselines "
        "(LSTM, GRU, Transformer, XGBoost) trained on price features alone."
    )

    # ══════════════════════════════════════════════════════════
    #  3. DATA SOURCES & FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("3", "Data Sources & Feature Engineering")

    pdf.sub_title("3.1  Price Data (OHLCV)")
    pdf.body_text(
        "Historical daily OHLCV (Open, High, Low, Close, Volume) data for Tesla is fetched via "
        "the Yahoo Finance API (yfinance) from June 29, 2010 (IPO date) to the present day. "
        "Derived features include daily returns, log returns, price range, and price range percentage."
    )

    pdf.sub_title("3.2  Technical Indicators")
    pdf.body_text("A rich set of technical indicators is computed to capture market dynamics:")
    pdf.bullet("Moving Averages: SMA (5, 10, 20, 50, 100, 200), EMA (12, 26, 50)")
    pdf.bullet("Momentum: RSI-14, Stochastic Oscillator (%K, %D), Momentum (5, 10, 20), ROC (10, 20)")
    pdf.bullet("Trend: MACD, MACD Signal, MACD Histogram")
    pdf.bullet("Volatility: Bollinger Bands (Upper, Lower, Width, Position), ATR-14, Parkinson Vol, Rolling Vol (5, 10, 20)")
    pdf.bullet("Volume: OBV, OBV SMA, VWAP")
    pdf.bullet("Price Patterns: Candlestick body, upper/lower shadows, gap detection")

    pdf.sub_title("3.3  Market Context Features (New)")
    pdf.body_text(
        "To provide broader market context, two additional data sources are integrated:"
    )
    pdf.bullet("S&P 500 (SPY): Daily close price and returns, enabling computation of Tesla's alpha (TSLA return minus SPY return).")
    pdf.bullet("VIX (Volatility Index): The market's expectation of 30-day volatility, acting as a fear gauge.")

    pdf.sub_title("3.4  Calendar Features (New)")
    pdf.body_text("Cyclical and categorical calendar features are added:")
    pdf.bullet("Day of week (normalized to [0, 1])")
    pdf.bullet("Month encoded as sin/cos for cyclical continuity")
    pdf.bullet("Binary flags for month-end and quarter-end trading sessions")

    pdf.sub_title("3.5  Sentiment Data")
    pdf.body_text(
        "News sentiment analysis is performed using VADER (Valence Aware Dictionary and sEntiment "
        "Reasoner) and optionally FinBERT (a BERT model fine-tuned on financial text). "
        "When real-time RSS feed data is available, headlines from Yahoo Finance and Google News "
        "are scored. For historical periods without real data, a synthetic sentiment generator "
        "produces sentiment signals derived from lagged price returns, rolling trends, and "
        "volatility, combined with independent noise to avoid information leakage."
    )
    pdf.body_text("Eight sentiment features are produced per day:")
    pdf.bullet("sentiment_compound, sentiment_positive, sentiment_negative, sentiment_neutral")
    pdf.bullet("news_count, sentiment_compound_lag1, sentiment_compound_lag2, sentiment_compound_lag3")

    # ══════════════════════════════════════════════════════════
    #  4. MODEL ARCHITECTURES
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("4", "Model Architectures")

    pdf.sub_title("4.1  Multimodal Fusion Model (Primary)")
    pdf.body_text(
        "The primary architecture is a multimodal fusion network consisting of three major components:"
    )

    pdf.sub_sub_title("Time-Series Encoder")
    pdf.body_text(
        "A 2-layer Bidirectional LSTM with 128 hidden units per direction processes 60-day "
        "sequences of price and technical features. An input projection layer maps raw features "
        "to the hidden dimension. A learned attention mechanism produces a weighted context vector "
        "from the full LSTM output sequence, allowing the model to focus on important time steps."
    )

    pdf.sub_sub_title("Sentiment Encoder")
    pdf.body_text(
        "A Temporal CNN encoder with multi-scale 1D convolutions (kernel sizes 3, 5, and 7) "
        "extracts local sentiment patterns. Global average pooling compresses temporal information "
        "into a fixed-length vector. This architecture captures short-term and medium-term "
        "sentiment trends simultaneously."
    )

    pdf.sub_sub_title("Cross-Modal Attention")
    pdf.body_text(
        "A 4-head Multi-Head Attention layer computes cross-modal interactions. The full LSTM "
        "output sequence serves as queries while the sentiment encoding provides keys and values. "
        "This allows each time step in the price sequence to attend to the relevant sentiment "
        "context. A residual connection preserves the original time-series information."
    )

    pdf.sub_sub_title("Fusion & Output Heads")
    pdf.body_text(
        "The attended time-series encoding and sentiment encoding are concatenated and passed "
        "through a two-layer fusion MLP (256 -> 128 units with LayerNorm, ReLU, and dropout). "
        "Three output heads produce:"
    )
    pdf.bullet("Single-day regression: Predicts next-day return (1 output)")
    pdf.bullet("Multi-day regression: Predicts returns for 1, 3, 5, and 7 days ahead (4 outputs)")
    pdf.bullet("Direction classification: Predicts Up/Down movement (2-class softmax)")

    pdf.sub_title("4.2  Baseline Models")

    pdf.sub_sub_title("Bidirectional LSTM Regressor")
    pdf.body_text(
        "A standalone 2-layer Bidirectional LSTM with 128 hidden units, input projection, "
        "and a 2-layer dense head. Uses the concatenation of the last forward hidden state "
        "and first backward hidden state for prediction."
    )

    pdf.sub_sub_title("Bidirectional GRU Regressor")
    pdf.body_text(
        "Architecturally identical to the LSTM regressor but uses Gated Recurrent Units, "
        "which have fewer parameters (no cell state) and often train faster."
    )

    pdf.sub_sub_title("Transformer Encoder Regressor (New)")
    pdf.body_text(
        "A 2-layer Transformer Encoder with 4 attention heads, 128 model dimension, and 256 "
        "feed-forward dimension. Sinusoidal positional encodings inject temporal ordering. "
        "The last token embedding is used for regression output. This model captures long-range "
        "dependencies without recurrence."
    )

    pdf.sub_sub_title("XGBoost Regressor")
    pdf.body_text(
        "An XGBRegressor with 400 estimators, max depth 4, learning rate 0.05, and 0.8 "
        "subsample rate. Sequences are flattened to (batch, seq_len * features) for input. "
        "This tree-based model serves as a strong non-neural baseline."
    )

    # ══════════════════════════════════════════════════════════
    #  5. TRAINING METHODOLOGY
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("5", "Training Methodology")

    pdf.sub_title("5.1  Data Splitting")
    pdf.body_text(
        "Data is split chronologically (no shuffling) into 80% training, 10% validation, "
        "and 10% test sets. Chronological splitting is critical for time-series to prevent "
        "look-ahead bias."
    )

    pdf.sub_title("5.2  Feature Scaling")
    pdf.body_text(
        "StandardScaler is applied to input features and MinMaxScaler to target returns. "
        "Critically, scalers are fitted exclusively on the training set to prevent data leakage. "
        "The fitted scalers are then applied to validation and test sets via transform-only."
    )

    pdf.sub_title("5.3  Loss Functions")
    pdf.body_text("The fusion model uses a combined loss:")
    pdf.bullet("Huber Loss (SmoothL1) for single-day regression (weight: 0.5)")
    pdf.bullet("Cross-Entropy Loss for direction classification (weight: 0.5)")
    pdf.bullet("Huber Loss for multi-day regression (weight: 0.3)")
    pdf.body_text(
        "Huber Loss is chosen over MSE for its robustness to outlier price movements, "
        "which are common in Tesla's highly volatile stock."
    )

    pdf.sub_title("5.4  Optimizer & Scheduler")
    pdf.body_text(
        "AdamW optimizer with weight decay 1e-5 and a OneCycleLR scheduler (max_lr = 10x "
        "initial lr) provide fast convergence with an automatic warmup phase and cosine "
        "annealing. Gradient clipping at max_norm=1.0 prevents exploding gradients."
    )

    pdf.sub_title("5.5  Early Stopping")
    pdf.body_text(
        "Training runs for up to 200 epochs with early stopping (patience = 30 epochs). "
        "The best model checkpoint (lowest validation loss) is saved. All baseline models "
        "use identical training settings for fair comparison."
    )

    pdf.sub_title("5.6  Training Configuration Summary")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Sequence Length", "60 days"],
            ["Batch Size", "32"],
            ["Initial Learning Rate", "1e-4"],
            ["Max Epochs", "200"],
            ["Early Stopping Patience", "30"],
            ["Optimizer", "AdamW (weight_decay=1e-5)"],
            ["Scheduler", "OneCycleLR"],
            ["Loss (Regression)", "Huber / SmoothL1"],
            ["Loss (Classification)", "CrossEntropy"],
            ["Gradient Clipping", "max_norm=1.0"],
            ["Train/Val/Test Split", "80% / 10% / 10%"],
        ],
        col_widths=[80, 110],
    )

    # ══════════════════════════════════════════════════════════
    #  6. MULTI-DAY PREDICTION
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("6", "Multi-Day Prediction")
    pdf.body_text(
        "A key enhancement of this project is the ability to forecast stock returns over "
        "multiple horizons simultaneously. The model predicts returns for 1, 3, 5, and 7 "
        "trading days ahead using a single forward pass."
    )

    pdf.sub_title("6.1  Multi-Output Architecture")
    pdf.body_text(
        "The fusion model's shared trunk (time-series encoder + sentiment encoder + fusion MLP) "
        "produces a common latent representation. A dedicated multi-output regression head maps "
        "this representation to 4 outputs, one per horizon. Each horizon has its own MinMaxScaler "
        "fitted on the training set to handle different return distributions."
    )

    pdf.sub_title("6.2  Multi-Day Target Generation")
    pdf.body_text(
        "For each training sample at time t, four target returns are computed:"
    )
    pdf.bullet("1-day return: (close[t+1] - close[t]) / close[t]")
    pdf.bullet("3-day return: (close[t+3] - close[t]) / close[t]")
    pdf.bullet("5-day return: (close[t+5] - close[t]) / close[t]")
    pdf.bullet("7-day return: (close[t+7] - close[t]) / close[t]")
    pdf.body_text(
        "The multi-day loss is averaged over all valid (non-NaN) horizon targets and added to "
        "the total loss with a weight of 0.3."
    )

    pdf.sub_title("6.3  Price Reconstruction")
    pdf.body_text(
        "At inference time, predicted returns are inverse-transformed using their respective "
        "per-horizon scalers, and actual prices are reconstructed:"
    )
    pdf.code_block("  predicted_price_Nd = current_close * (1 + predicted_return_Nd)")

    # ══════════════════════════════════════════════════════════
    #  7. EVALUATION METRICS & BASELINE RESULTS
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("7", "Evaluation Metrics & Baseline Results")

    pdf.sub_title("7.1  Metrics")
    pdf.bullet("RMSE (Root Mean Square Error): Measures average prediction error in dollars.")
    pdf.bullet("MAE (Mean Absolute Error): Average absolute error in dollars.")
    pdf.bullet("MAPE (Mean Absolute Percentage Error): Scale-independent percentage error.")
    pdf.bullet("Direction Accuracy: Percentage of correctly predicted Up/Down movements.")

    pdf.sub_title("7.2  Previous Baseline Results (Before Upgrades)")
    pdf.body_text("The following results were obtained before the improvements described in this report:")
    pdf.add_table(
        ["Model", "RMSE", "MAE", "MAPE", "Dir. Accuracy"],
        [
            ["LSTM", "$13.51", "$10.51", "3.08%", "49.9%"],
            ["GRU", "$14.56", "$11.06", "3.25%", "49.1%"],
            ["XGBoost", "$14.71", "$11.51", "3.31%", "49.1%"],
        ],
        col_widths=[35, 35, 35, 35, 50],
    )
    pdf.body_text(
        "Direction accuracy of approximately 50% indicates the previous models were essentially "
        "performing at chance level for directional prediction. This was partly due to data "
        "leakage in the scalers inflating regression metrics while masking classification weakness."
    )

    pdf.sub_title("7.3  Expected Improvements")
    pdf.body_text("The following changes are expected to improve performance:")
    pdf.bullet("Fixing scaler leakage will produce more honest and potentially improved metrics.")
    pdf.bullet("Decoupled synthetic sentiment provides an independent signal source.")
    pdf.bullet("Market context (SPY, VIX) gives broader market awareness.")
    pdf.bullet("Proper cross-modal attention enables meaningful multimodal learning.")
    pdf.bullet("OneCycleLR and Huber loss improve training dynamics and robustness.")
    pdf.bullet("Transformer baseline may capture long-range dependencies better than RNNs.")
    pdf.body_text(
        "To realize these improvements, retrain all models using: python train.py and python model_comparison.py"
    )

    # ══════════════════════════════════════════════════════════
    #  8. SYSTEM ARCHITECTURE & PIPELINE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("8", "System Architecture & Pipeline")

    pdf.sub_title("8.1  Project Structure")
    pdf.code_block(
        "  Project Root/\n"
        "  |-- app/\n"
        "  |   +-- streamlit_app.py        # Interactive web dashboard\n"
        "  |-- data/\n"
        "  |   |-- raw/                    # Raw stock & sentiment data\n"
        "  |   +-- processed/              # Preprocessed sequences\n"
        "  |-- models/                     # Saved model checkpoints & scalers\n"
        "  |-- notebooks/\n"
        "  |   +-- exploration.ipynb       # Data exploration & analysis\n"
        "  |-- src/\n"
        "  |   |-- data/\n"
        "  |   |   |-- stock_data.py       # Yahoo Finance data fetcher\n"
        "  |   |   |-- sentiment_data.py   # Sentiment data generator\n"
        "  |   |   +-- preprocessing.py    # Data preprocessing pipeline\n"
        "  |   |-- features/\n"
        "  |   |   +-- technical.py        # Technical indicators + market context\n"
        "  |   |-- models/\n"
        "  |   |   |-- fusion.py           # Multimodal fusion + MHA attention\n"
        "  |   |   |-- time_series.py      # Time-series LSTM encoder\n"
        "  |   |   |-- text_encoder.py     # Sentiment encoders\n"
        "  |   |   |-- regression_models.py # LSTM, GRU, Transformer, XGBoost\n"
        "  |   |   +-- trainer.py          # Training loop (OneCycleLR)\n"
        "  |   +-- utils/                  # Helper functions\n"
        "  |-- config.py                   # Configuration settings\n"
        "  |-- train.py                    # Train multimodal fusion model\n"
        "  |-- model_comparison.py         # Train & compare all baselines\n"
        "  |-- predict.py                  # CLI prediction (multi-day)\n"
        "  +-- requirements.txt            # Dependencies"
    )

    pdf.sub_title("8.2  Execution Pipeline")
    pdf.body_text("The system operates in four stages:")
    pdf.bullet("Stage 1 - Data Acquisition: Fetch OHLCV data from Yahoo Finance, download SPY and VIX, fetch or generate sentiment data.")
    pdf.bullet("Stage 2 - Feature Engineering: Calculate 50+ technical indicators, market context, and calendar features. Generate multi-day target returns.")
    pdf.bullet("Stage 3 - Training: Fit scalers on training set only, create 60-day sequences, train models with OneCycleLR and early stopping.")
    pdf.bullet("Stage 4 - Inference: Load best checkpoint, preprocess new data, predict single-day and multi-day returns, reconstruct prices.")

    pdf.sub_title("8.3  GPU Support")
    pdf.body_text("Automatic device selection: CUDA (NVIDIA) > MPS (Apple Silicon M1/M2/M3/M4) > CPU.")

    # ══════════════════════════════════════════════════════════
    #  9. TECHNOLOGY STACK
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("9", "Technology Stack")
    pdf.add_table(
        ["Category", "Technologies"],
        [
            ["Deep Learning", "PyTorch, Transformers (HuggingFace)"],
            ["Machine Learning", "Scikit-learn, XGBoost"],
            ["Data Processing", "Pandas, NumPy, yfinance"],
            ["Visualization", "Matplotlib, Seaborn, Plotly"],
            ["Web Application", "Streamlit"],
            ["NLP / Sentiment", "NLTK, VADER, FinBERT"],
            ["Utilities", "tqdm, joblib, feedparser, BeautifulSoup"],
        ],
        col_widths=[60, 130],
    )

    # ══════════════════════════════════════════════════════════
    #  10. STREAMLIT DASHBOARD
    # ══════════════════════════════════════════════════════════
    pdf.section_title("10", "Streamlit Dashboard")
    pdf.body_text("The interactive web dashboard (streamlit run app/streamlit_app.py) provides five tabs:")
    pdf.bullet("Price Chart: Interactive candlestick chart with SMA overlays and volume bars.")
    pdf.bullet("Technical Analysis: RSI, MACD, Bollinger Bands, and Stochastic Oscillator charts with buy/sell signal interpretation.")
    pdf.bullet("Sentiment: Time-series visualization of compound sentiment score with 7-day average and trend indicators.")
    pdf.bullet("Prediction: One-click prediction showing next-day price, multi-day forecast chart (1/3/5/7 days), direction probability bars, and confidence score.")
    pdf.bullet("Model Comparison: Side-by-side RMSE/MAE/MAPE/Direction Accuracy for LSTM, GRU, Transformer, and XGBoost with bar chart visualization and live ensemble prediction.")

    # ══════════════════════════════════════════════════════════
    #  11. CONCLUSION & FUTURE WORK
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("11", "Conclusion & Future Work")
    pdf.body_text(
        "This project demonstrates a complete end-to-end pipeline for multimodal stock price "
        "prediction, integrating time-series analysis with sentiment data. The multimodal "
        "fusion architecture with cross-modal multi-head attention represents a principled "
        "approach to combining heterogeneous data sources."
    )
    pdf.body_text(
        "Key contributions include: (1) a returns-based prediction approach for stationarity, "
        "(2) multi-day prediction via multi-output regression heads, (3) proper data leakage "
        "prevention via train-only scaler fitting, (4) market context features (SPY, VIX) for "
        "broader awareness, and (5) fair model comparison with identical training regimes."
    )
    pdf.sub_title("Future Work")
    pdf.bullet("Integrate Optuna for automated hyperparameter optimization.")
    pdf.bullet("Incorporate options market data (Put/Call ratio) as additional sentiment signals.")
    pdf.bullet("Implement a formal backtesting module with Sharpe ratio and max drawdown metrics.")
    pdf.bullet("Deploy as a Dockerized microservice with FastAPI backend for production inference.")
    pdf.bullet("Explore Temporal Fusion Transformers (TFT) for interpretable multi-horizon forecasting.")
    pdf.bullet("Use real historical sentiment datasets from Kaggle for stronger multimodal learning.")

    # ══════════════════════════════════════════════════════════
    #  12. REFERENCES
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("12", "References")
    refs = [
        "[1] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.",
        "[2] Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078.",
        "[3] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS).",
        "[4] Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv:1603.02754.",
        "[5] Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.",
        "[6] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.",
        "[7] Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting.",
        "[8] Smith, L.N. (2019). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv:1708.07120.",
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    for ref in refs:
        pdf.multi_cell(0, 5, ref)
        pdf.ln(3)

    # ── Save ─────────────────────────────────────────────────
    pdf.output(output_path)
    print(f"\n{'='*60}")
    print(f"PDF report saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    output = str(
        Path(__file__).parent / "Tesla_Stock_Prediction_Report.pdf"
    )
    build_report(output)
