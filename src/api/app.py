"""REST API: accept latest window, return multi-step trend predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tensorflow import keras

import config
from src.data.clean import clean
from src.data.features import engineer_features
from src.data.sequences import build_sequences


class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    window_size: Optional[int] = Field(None, ge=10, le=120)


class StepPrediction(BaseModel):
    step: int
    direction: str
    confidence: float


class PredictionResponse(BaseModel):
    ticker: str
    window_end_date: str
    predictions: List[StepPrediction]
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: Optional[int]


_state: dict[str, Any] = {}


def _load_artifacts() -> None:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}. Run train first.")
    _state["model"] = keras.models.load_model(config.MODEL_PATH)
    _state["scaler"] = joblib.load(config.SCALER_PATH)
    _state["meta"] = joblib.load(config.META_PATH)
    if config.PROCESSED_CSV.exists():
        _state["featured_df"] = pd.read_csv(config.PROCESSED_CSV, parse_dates=["date"])
    else:
        _state["featured_df"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_artifacts()
    except FileNotFoundError:
        _state["model"] = None
    yield


app = FastAPI(
    title="Stock Trend Forecast API",
    description="LSTM multi-step directional trend predictions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    meta = _state.get("meta")
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
        feature_count=meta["n_features"] if meta else None,
    )


@app.get("/tickers")
def list_tickers() -> dict:
    df = _state.get("featured_df")
    if df is None:
        raise HTTPException(503, "Processed data unavailable. Train the model first.")
    tickers = sorted(df["ticker"].unique().tolist())
    return {"tickers": tickers, "count": len(tickers)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    model = _state.get("model")
    meta = _state.get("meta")
    scaler = _state.get("scaler")
    df = _state.get("featured_df")

    if model is None or meta is None or scaler is None or df is None:
        raise HTTPException(503, "Model not ready. Run: python scripts/train_model.py")

    ticker = req.ticker.upper()
    subset = df[df["ticker"] == ticker]
    if subset.empty:
        raise HTTPException(404, f"No data for ticker {ticker}")

    window = req.window_size or meta["window_size"]
    feature_cols = meta["feature_columns"]

    if len(subset) < window:
        raise HTTPException(400, f"Need at least {window} rows for {ticker}, have {len(subset)}")

    recent = subset.tail(window)
    X, _, dates = build_sequences(recent, feature_cols, window=window, horizon=meta["forecast_horizon"])
    if len(X) == 0:
        raise HTTPException(400, "Could not build sequence from provided data")

    X_last = X[-1:]
    n, w, f = X_last.shape
    X_scaled = scaler.transform(X_last.reshape(n * w, f)).reshape(n, w, f).astype(np.float32)

    probs = model.predict(X_scaled, verbose=0)[0]
    predictions = []
    for i, p in enumerate(probs):
        direction = "up" if p >= 0.5 else "down"
        confidence = float(p if p >= 0.5 else 1.0 - p)
        predictions.append(StepPrediction(step=i + 1, direction=direction, confidence=round(confidence, 4)))

    return PredictionResponse(
        ticker=ticker,
        window_end_date=str(pd.Timestamp(dates[-1]).date()),
        predictions=predictions,
        model_version="lstm_trend_v1",
        disclaimer="Directional forecast only — not financial advice. Not profit accuracy.",
    )


@app.get("/metrics")
def get_metrics() -> dict:
    """Return last training metrics if available."""
    metrics_path = config.MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "No metrics file. Train the model first.")
    import json
    return json.loads(metrics_path.read_text())


# Serve web UI
if config.WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index = config.WEB_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(404, "Web UI not found")
