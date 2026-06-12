"""Clean raw OHLCV: types, ordering, gaps."""

from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0) & (df["close"] > 0)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Forward-fill small gaps within each ticker (no cross-ticker leakage)
    filled = []
    for _, grp in df.groupby("ticker", sort=False):
        filled.append(grp.ffill().bfill())
    df = pd.concat(filled, ignore_index=True)

    return df
