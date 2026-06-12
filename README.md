# AI-Powered Stock Market Trend Forecasting

LSTM-based system that ingests long OHLCV history, engineers features, trains a sequence model, and serves multi-step **direction** forecasts (up/down) through a REST API to a web UI.

> **Disclaimer:** Directional accuracy on held-out chronological data — **not** trading profit. Not financial advice.

## Architecture

```
Raw OHLCV (1M+ rows)
  → clean & order chronologically
  → feature engineering (returns, MAs, volatility)
  → scale (fit on train only)
  → sliding windows (60 × features)
  → stacked LSTM + dense head
  → REST API → web app
```

## Quick Start

```bash
cd "untitled folder 5"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (ingest → features → LSTM → evaluate)
python scripts/train_model.py

# Serve API + web UI at http://localhost:8765
python scripts/run_api.py
```

Force re-download/re-ingest:

```bash
python scripts/train_model.py --force-ingest
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health |
| GET | `/tickers` | Available symbols |
| POST | `/predict` | `{"ticker": "AAPL"}` → multi-step forecast |
| GET | `/metrics` | Last training evaluation |
| GET | `/` | Web dashboard |

### Example

```bash
curl -X POST http://localhost:8765/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

## Model Details

- **Input:** Last 60 timesteps × 24 engineered features (OHLCV, returns, MAs, volatility)
- **Output:** 5-step binary direction (sigmoid → up/down + confidence)
- **Loss:** Binary cross-entropy
- **Regularization:** Dropout, batch normalization, LR scheduling, early stopping
- **Split:** Chronological 70/15/15 — no shuffle (prevents leakage)
- **Scaler:** Fit on training windows only

## Interview Talking Points

| Question | Answer |
|----------|--------|
| Why LSTM over ARIMA? | Nonlinear patterns across many features; always benchmark classical baselines first |
| Data split? | Chronological walk-forward — random shuffle leaks future info |
| Loss function? | Binary cross-entropy for direction |
| Data leakage risks? | Shuffled splits, scalers on full data, features using future prices |
| Would you trade on it? | No — costs, regime change, near-efficient markets; engineering exercise |
| Improvements? | Transformers, exogenous features, probabilistic intervals |

## Project Layout

```
config.py              # hyperparameters & paths
src/data/              # ingest, clean, features, sequences
src/model/             # LSTM build, train, evaluate
src/api/app.py         # FastAPI service
web/                   # dashboard
scripts/               # train & serve entrypoints
```
