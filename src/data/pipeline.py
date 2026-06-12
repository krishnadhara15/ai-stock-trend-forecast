"""End-to-end data pipeline: ingest → clean → features → sequences."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from src.data.clean import clean
from src.data.features import engineer_features
from src.data.ingest import ingest
from src.data.sequences import build_sequences, chronological_split


def run_data_pipeline(force_ingest: bool = False) -> dict:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if force_ingest or not config.RAW_CSV.exists():
        raw = ingest(save=True)
    else:
        raw = pd.read_csv(config.RAW_CSV, parse_dates=["date"])

    cleaned = clean(raw)
    featured, feature_cols = engineer_features(cleaned)
    config.FEATURE_COLUMNS = feature_cols
    featured.to_csv(config.PROCESSED_CSV, index=False)

    X, y, dates = build_sequences(featured, feature_cols)

    if config.MAX_SEQUENCES and len(X) > config.MAX_SEQUENCES:
        idx = np.linspace(0, len(X) - 1, config.MAX_SEQUENCES, dtype=int)
        X, y, dates = X[idx], y[idx], dates[idx]
        print(f"Subsampled to {len(X):,} sequences (chronological) for memory/runtime")

    splits = chronological_split(X, y, dates)

    # Fit scaler on training windows only — prevents leakage
    n_train, window, n_feat = splits["X_train"].shape
    scaler = StandardScaler()
    X_train_flat = splits["X_train"].reshape(n_train * window, n_feat)
    scaler.fit(X_train_flat)

    def scale_split(arr: np.ndarray) -> np.ndarray:
        n, w, f = arr.shape
        scaled = scaler.transform(arr.reshape(n * w, f))
        return scaled.reshape(n, w, f).astype(np.float32)

    splits["X_train"] = scale_split(splits["X_train"])
    splits["X_val"] = scale_split(splits["X_val"])
    splits["X_test"] = scale_split(splits["X_test"])

    joblib.dump(scaler, config.SCALER_PATH)
    meta = {
        "feature_columns": feature_cols,
        "window_size": config.WINDOW_SIZE,
        "forecast_horizon": config.FORECAST_HORIZON,
        "n_features": len(feature_cols),
        "total_rows_raw": len(raw),
        "total_rows_processed": len(featured),
        "n_sequences": len(X),
    }
    joblib.dump(meta, config.META_PATH)

    print(f"Pipeline complete: {meta['total_rows_raw']:,} raw → {meta['n_sequences']:,} sequences")
    return {**splits, "meta": meta, "scaler": scaler, "featured_df": featured}
