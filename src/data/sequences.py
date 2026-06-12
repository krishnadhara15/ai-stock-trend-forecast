"""Window time series into (samples, window, features) tensors."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: Optional[int] = None,
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = window or config.WINDOW_SIZE
    horizon = horizon or config.FORECAST_HORIZON
    label_cols = [f"direction_t+{s}" for s in range(1, horizon + 1)]

    X_list, y_list, date_list = [], [], []

    for _, group in df.groupby("ticker"):
        feats = group[feature_cols].values.astype(np.float32)
        labels = group[label_cols].values.astype(np.float32)
        dates = group["date"].values

        for i in range(window - 1, len(group)):
            X_list.append(feats[i - window + 1 : i + 1])
            y_list.append(labels[i])
            date_list.append(dates[i])

    X = np.stack(X_list)
    y = np.stack(y_list)
    dates = np.array(date_list)
    return X, y, dates


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
) -> Dict[str, np.ndarray]:
    order = np.argsort(dates)
    X, y = X[order], y[order]
    dates = dates[order]

    n = len(X)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "dates_train": dates[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "dates_val": dates[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "dates_test": dates[val_end:],
    }
