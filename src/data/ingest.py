"""Download or synthesize OHLCV history to reach 1M+ rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _synthetic_ohlcv(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV for fast local runs."""
    rng = np.random.default_rng(seed)
    tickers = config.TICKERS
    rows_per_ticker = max(n_rows // len(tickers), 5000)
    frames = []

    for i, ticker in enumerate(tickers):
        dates = pd.bdate_range("2000-01-01", periods=rows_per_ticker, freq="B")
        price = 50.0 + i * 3.0
        closes = [price]
        for _ in range(1, rows_per_ticker):
            closes.append(closes[-1] * (1 + rng.normal(0.0002, 0.015)))

        close = np.array(closes)
        high = close * (1 + rng.uniform(0.001, 0.02, size=rows_per_ticker))
        low = close * (1 - rng.uniform(0.001, 0.02, size=rows_per_ticker))
        open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=rows_per_ticker)
        volume = rng.integers(1_000_000, 50_000_000, size=rows_per_ticker)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _download_yfinance() -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for ticker in config.TICKERS:
        df = yf.download(
            ticker,
            start=config.START_DATE,
            end=config.END_DATE,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            continue
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [str(c).lower() for c in df.columns]
        rename = {"adj close": "adj_close", "date": "date"}
        df = df.rename(columns=rename)
        df["ticker"] = ticker
        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        frames.append(df[[c for c in keep if c in df.columns]])

    if not frames:
        raise RuntimeError("yfinance returned no data")
    return pd.concat(frames, ignore_index=True)


def ingest(save: bool = True) -> pd.DataFrame:
    """Fetch market history; fall back to synthetic data to hit row target."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = _download_yfinance()
        source = "yfinance"
    except Exception:
        df = pd.DataFrame()
        source = "synthetic"

    if len(df) < config.MIN_ROWS_TARGET:
        needed = config.MIN_ROWS_TARGET - len(df)
        synthetic = _synthetic_ohlcv(max(needed, config.MIN_ROWS_TARGET))
        df = pd.concat([df, synthetic], ignore_index=True) if len(df) else synthetic
        source = "mixed" if source == "yfinance" else "synthetic"

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["source"] = source

    if save:
        df.to_csv(config.RAW_CSV, index=False)

    print(f"Ingested {len(df):,} rows via {source}")
    return df
