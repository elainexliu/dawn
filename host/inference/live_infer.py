"""
live_infer.py — phase 2 stub for on-laptop live inference.

NOT yet functional.  Placeholder for the inference pipeline that will run
after the sensing/labeling/training loop is validated.

Architecture:
- Imports packet parsing directly from acquisition.receiver (no duplication).
- Buffers a rolling window of IMU+PPG samples.
- Calls pipeline.features.compute_features (same function used at training time).
- Normalizes with the saved scaler (pipeline.normalization.load_and_transform).
- Calls model.predict_proba → threshold → trigger signal (stubbed as no-op).

TENS/MNS actuation is intentionally not implemented here.
See CLAUDE.md — actuation is gated behind sham-controlled self-experiment.
"""
import argparse
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from host.pipeline.features import compute_features
from host.pipeline.normalization import load_and_transform

# Re-use packet parsing from receiver — do NOT re-implement
from host.acquisition.receiver import (
    PKT_TYPE_IMU_PPG,
    FIELDS_IMU_PPG,
    _parse_imu_ppg,
    FMT_IMU_PPG,
    IMU_PPG_LEN,
    PKT_START,
)

import serial

# Configuration
WINDOW_MS   = 2000
FS_IMU_PPG  = 50   # Hz
WINDOW_SIZE = int(WINDOW_MS / 1000 * FS_IMU_PPG)  # samples
THRESHOLD   = 0.5  # prediction probability threshold


def _trigger(prob: float) -> None:
    """Stub — replace with MNS actuation when hardware arrives."""
    pass


def run(port: str, baud: int, model_path: str, scaler_path: str) -> None:
    model  = joblib.load(model_path)
    buffer: deque[dict] = deque(maxlen=WINDOW_SIZE)

    print(f"Loaded model from {model_path}")
    print(f"Connecting to {port} @ {baud} ...")

    with serial.Serial(port, baud, timeout=0.02) as ser:
        try:
            while True:
                byte = ser.read(1)
                if not byte or byte[0] != PKT_START:
                    continue
                type_byte = ser.read(1)
                if not type_byte or type_byte[0] != PKT_TYPE_IMU_PPG:
                    # Skip EMG packets in inference — not used for classification
                    continue
                rest = ser.read(IMU_PPG_LEN - 2)
                raw = byte + type_byte + rest
                pkt = _parse_imu_ppg(raw)
                if pkt is None:
                    continue

                buffer.append(pkt)

                if len(buffer) < WINDOW_SIZE:
                    continue

                window_df = pd.DataFrame(list(buffer)).set_index("timestamp_ms")
                feats     = compute_features(window_df)
                X         = np.array(list(feats.values()), dtype=float).reshape(1, -1)
                X_scaled  = load_and_transform(X, scaler_path)
                prob      = model.predict_proba(X_scaled)[0, 1]

                if prob >= THRESHOLD:
                    print(f"[{pkt['timestamp_ms']}] BFRB predicted  (p={prob:.2f})")
                    _trigger(prob)

        except KeyboardInterrupt:
            print("Stopped.")


def _parse_args():
    p = argparse.ArgumentParser(description="Dawn live inference (phase 2 stub)")
    p.add_argument("--port",   required=True)
    p.add_argument("--baud",   type=int, default=921600)
    p.add_argument("--model",  default="models/best.model.joblib")
    p.add_argument("--scaler", default="models/best.scaler.joblib")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.port, args.baud, args.model, args.scaler)
