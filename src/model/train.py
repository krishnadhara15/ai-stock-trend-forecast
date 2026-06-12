"""Train LSTM with early stopping, LR schedule, chronological validation."""

from __future__ import annotations

import numpy as np
from tensorflow import keras

import config
from src.data.pipeline import run_data_pipeline
from src.model.build import build_lstm_model, learning_rate_schedule
from src.model.evaluate import evaluate_model, format_report


def train_model(force_ingest: bool = False) -> dict:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = run_data_pipeline(force_ingest=force_ingest)
    meta = data["meta"]

    X_train, y_train = data["X_train"], data["y_train"]
    if config.MAX_TRAIN_SEQUENCES and len(X_train) > config.MAX_TRAIN_SEQUENCES:
        idx = np.linspace(0, len(X_train) - 1, config.MAX_TRAIN_SEQUENCES, dtype=int)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"Subsampled train set to {len(X_train):,} sequences for runtime")

    model = build_lstm_model(
        window_size=meta["window_size"],
        n_features=meta["n_features"],
        horizon=meta["forecast_horizon"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        learning_rate_schedule(),
    ]

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # chronological — no leakage
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    print(format_report(metrics))

    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    return {"history": history.history, "metrics": metrics, "meta": meta}
