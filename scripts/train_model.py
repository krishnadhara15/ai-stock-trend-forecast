#!/usr/bin/env python3
"""Run full pipeline and train LSTM model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.model.train import train_model


def main():
    force = "--force-ingest" in sys.argv
    result = train_model(force_ingest=force)
    metrics_path = config.MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
