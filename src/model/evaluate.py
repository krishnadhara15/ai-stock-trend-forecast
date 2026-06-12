"""Chronological evaluation: directional accuracy and baseline comparison."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-step and aggregate up/down accuracy."""
    pred_binary = (y_pred >= threshold).astype(np.float32)
    per_step = (pred_binary == y_true).mean(axis=0)
    aggregate = (pred_binary == y_true).mean()
    return {
        "aggregate": float(aggregate),
        "per_step": {f"t+{i+1}": float(v) for i, v in enumerate(per_step)},
    }


def baseline_accuracy(y_true: np.ndarray) -> dict:
    """Naive baseline: always predict majority class per step."""
    majority = (y_true.mean(axis=0) >= 0.5).astype(np.float32)
    preds = np.tile(majority, (len(y_true), 1))
    return directional_accuracy(y_true, preds)


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    dir_acc = directional_accuracy(y_test, y_prob)
    baseline = baseline_accuracy(y_test)

    improvement = dir_acc["aggregate"] - baseline["aggregate"]
    return {
        "loss": float(loss),
        "keras_accuracy": float(acc),
        "auc": float(auc),
        "directional_accuracy": dir_acc,
        "baseline_accuracy": baseline,
        "improvement_vs_baseline": float(improvement),
        "improvement_pct_points": float(improvement * 100),
    }


def format_report(metrics: dict) -> str:
    da = metrics["directional_accuracy"]["aggregate"] * 100
    base = metrics["baseline_accuracy"]["aggregate"] * 100
    imp = metrics["improvement_pct_points"]
    lines = [
        "=== Evaluation (held-out chronological test) ===",
        f"Directional accuracy: {da:.2f}%  (up/down — not profit)",
        f"Baseline accuracy:      {base:.2f}%",
        f"Improvement:            +{imp:.2f} percentage points",
        f"Loss (BCE):             {metrics['loss']:.4f}",
        f"AUC:                    {metrics['auc']:.4f}",
        "Per-step accuracy:",
    ]
    for step, val in metrics["directional_accuracy"]["per_step"].items():
        lines.append(f"  {step}: {val * 100:.2f}%")
    return "\n".join(lines)
