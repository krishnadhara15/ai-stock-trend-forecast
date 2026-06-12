"""LSTM architecture: stacked LSTM + dense head with regularization."""

from __future__ import annotations

from typing import List, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_lstm_model(
    window_size: int,
    n_features: int,
    horizon: int,
    lstm_units: Optional[List[int]] = None,
    dropout: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> keras.Model:
    lstm_units = lstm_units or config.LSTM_UNITS
    dropout = dropout if dropout is not None else config.DROPOUT_RATE
    learning_rate = learning_rate or config.LEARNING_RATE

    inputs = keras.Input(shape=(window_size, n_features), name="sequence_input")
    x = inputs

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout / 2, name="dropout_head")(x)
    outputs = layers.Dense(horizon, activation="sigmoid", name="direction_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_trend_forecaster")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=True, num_labels=horizon),
        ],
    )
    return model


def learning_rate_schedule():
    """Reduce LR on plateau — wired via callback in trainer."""
    return keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
