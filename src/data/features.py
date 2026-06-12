"""Automated feature engineering: returns, MAs, volatility."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

import config


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)

    for lag in config.RETURN_LAGS:
        df[f"return_{lag}"] = g["close"].pct_change(lag)

    df["log_return_1"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))

    for w in config.MA_WINDOWS:
        df[f"ma_{w}"] = g["close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f"close_to_ma_{w}"] = df["close"] / df[f"ma_{w}"] - 1.0

    for w in config.VOLATILITY_WINDOWS:
        df[f"volatility_{w}"] = g["close"].transform(
            lambda s: s.pct_change().rolling(w, min_periods=w).std()
        )

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["volume_log"] = np.log1p(df["volume"])
    df["volume_ma_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_return_1",
        *[f"return_{lag}" for lag in config.RETURN_LAGS],
        *[f"ma_{w}" for w in config.MA_WINDOWS],
        *[f"close_to_ma_{w}" for w in config.MA_WINDOWS],
        *[f"volatility_{w}" for w in config.VOLATILITY_WINDOWS],
        "hl_range", "oc_change", "volume_log", "volume_ratio",
    ]

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Multi-step direction labels: 1 = up, 0 = down for each future step
    for step in range(1, config.FORECAST_HORIZON + 1):
        df[f"direction_t+{step}"] = g["close"].transform(
            lambda s: (s.shift(-step) > s).astype(np.float32)
        )

    label_cols = [f"direction_t+{s}" for s in range(1, config.FORECAST_HORIZON + 1)]
    df = df.dropna(subset=label_cols).reset_index(drop=True)

    return df, feature_cols
