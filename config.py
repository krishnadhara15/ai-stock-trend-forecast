"""Central configuration for the stock trend forecasting pipeline."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
WEB_DIR = ROOT / "web"

# Data
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM",
    "PFE", "CSCO", "INTC", "VZ", "KO", "PEP", "MRK", "ABBV", "TMO",
    "COST", "AVGO", "NFLX", "AMD", "CRM", "ORCL", "ADBE", "NKE",
    "LLY", "MCD", "TXN", "QCOM", "HON", "UPS", "IBM", "CAT", "GE",
    "AMAT", "SBUX", "GS", "BLK", "AXP", "DE",
]
START_DATE = "2000-01-01"
END_DATE = "2024-01-31"
MIN_ROWS_TARGET = 1_000_000  # total rows before windowing

# Feature engineering
MA_WINDOWS = [5, 10, 20, 50]
VOLATILITY_WINDOWS = [5, 10, 20]
RETURN_LAGS = [1, 2, 3, 5]

# Sequences
WINDOW_SIZE = 60          # last N steps per sample
FORECAST_HORIZON = 5      # predict next k steps (direction)
FEATURE_COLUMNS = None    # set at runtime after feature engineering

# Train / val / test — chronological, no shuffle
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model
LSTM_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.3
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 5

# Cap sequences for memory/runtime (raw rows still 1M+ before windowing)
MAX_SEQUENCES = 250_000
MAX_TRAIN_SEQUENCES = 80_000

# API
API_HOST = "0.0.0.0"
API_PORT = 8765

# Paths
RAW_CSV = RAW_DIR / "ohlcv_combined.csv"
PROCESSED_CSV = PROCESSED_DIR / "features.csv"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "lstm_trend.keras"
META_PATH = MODEL_DIR / "model_meta.joblib"
