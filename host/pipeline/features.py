"""
features.py — single source of truth for feature extraction.

IMPORTANT: This module is imported by both training (offline) and inference
(online/live_infer.py).  Do not duplicate or shadow this logic elsewhere.

Feature set (per channel, where applicable):
  Time-domain:  mean, std, min, max, rms, zero_crossing_rate
  Freq-domain:  dominant_freq, power_0_5hz, power_5_15hz

Channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_ir, ppg_red
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nominal IMU+PPG sample rate (50 Hz)
_FS = 50.0

# Accel scaling: raw → g
_ACCEL_SCALE = 1.0 / 16384.0
# Gyro scaling: raw → °/s
_GYRO_SCALE  = 1.0 / 131.0


def compute_features(window_df: pd.DataFrame) -> dict[str, float]:
    """Extract features from one IMU+PPG window.

    Args:
        window_df: DataFrame with columns including at least:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_ir, ppg_red.
            Index is timestamp_ms (used only for length check).

    Returns:
        Flat dict mapping feature name → float value.
    """
    feats: dict[str, float] = {}

    # Scale raw values before feature extraction
    signals: dict[str, np.ndarray] = {
        "accel_x": window_df["accel_x"].to_numpy(dtype=float) * _ACCEL_SCALE,
        "accel_y": window_df["accel_y"].to_numpy(dtype=float) * _ACCEL_SCALE,
        "accel_z": window_df["accel_z"].to_numpy(dtype=float) * _ACCEL_SCALE,
        "gyro_x":  window_df["gyro_x"].to_numpy(dtype=float) * _GYRO_SCALE,
        "gyro_y":  window_df["gyro_y"].to_numpy(dtype=float) * _GYRO_SCALE,
        "gyro_z":  window_df["gyro_z"].to_numpy(dtype=float) * _GYRO_SCALE,
        "ppg_ir":  window_df["ppg_ir"].to_numpy(dtype=float),
        "ppg_red": window_df["ppg_red"].to_numpy(dtype=float),
    }

    for name, sig in signals.items():
        feats.update(_time_domain(sig, name))
        feats.update(_freq_domain(sig, name, fs=_FS))

    return feats


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _time_domain(sig: np.ndarray, prefix: str) -> dict[str, float]:
    n = len(sig)
    rms = float(np.sqrt(np.mean(sig ** 2))) if n > 0 else 0.0
    if n > 1:
        zcr = float(np.sum(np.diff(np.sign(sig)) != 0)) / (n - 1)
    else:
        zcr = 0.0
    return {
        f"{prefix}_mean": float(np.mean(sig)),
        f"{prefix}_std":  float(np.std(sig)),
        f"{prefix}_min":  float(np.min(sig)),
        f"{prefix}_max":  float(np.max(sig)),
        f"{prefix}_rms":  rms,
        f"{prefix}_zcr":  zcr,
    }


def _freq_domain(sig: np.ndarray, prefix: str, fs: float) -> dict[str, float]:
    n = len(sig)
    if n < 4:
        return {
            f"{prefix}_dominant_freq": 0.0,
            f"{prefix}_power_0_5hz":   0.0,
            f"{prefix}_power_5_15hz":  0.0,
        }

    fft_vals  = np.abs(np.fft.rfft(sig)) ** 2
    freqs     = np.fft.rfftfreq(n, d=1.0 / fs)
    df        = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    dominant_freq = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) else 0.0
    power_0_5   = float(np.sum(fft_vals[(freqs >= 0)   & (freqs < 5)])  * df)
    power_5_15  = float(np.sum(fft_vals[(freqs >= 5)   & (freqs < 15)]) * df)

    return {
        f"{prefix}_dominant_freq": dominant_freq,
        f"{prefix}_power_0_5hz":   power_0_5,
        f"{prefix}_power_5_15hz":  power_5_15,
    }
