"""
features.py - single source of truth for feature extraction.

Shared by training and inference - don't duplicate this logic elsewhere.

Feature set:
  accel_x/y/z, gyro_x/y/z - generic time-domain (mean, std, min, max, rms,
    zero_crossing_rate) + freq-domain (dominant_freq, power_0_5hz, power_5_15hz)
  ppg_ir - dedicated cardiac features via beat detection: ppg_hr_mean/std/min/max
    (bpm) and ppg_rmssd (ms; NaN unless the window is >= 5 minutes - see
    _ppg_hr_features). ppg_red is currently unused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# Nominal IMU+PPG sample rate (50 Hz)
_FS = 50.0

# Accel scaling: raw -> g
_ACCEL_SCALE = 1.0 / 16384.0
# Gyro scaling: raw -> deg/s
_GYRO_SCALE  = 1.0 / 131.0

# Cardiac pulse band for PPG bandpass filtering (~42-210 bpm)
_PPG_BAND_LOW_HZ  = 0.7
_PPG_BAND_HIGH_HZ = 3.5
# Minimum peak spacing, set by a 220bpm ceiling - avoids double-counting a
# single beat as two peaks.
_PPG_MAX_BPM = 220
# RMSSD needs several minutes of beats to mean anything physiologically;
# below this, return NaN rather than a number computed on too little data.
_RMSSD_MIN_WINDOW_MS = 300_000  # 5 min


def compute_features(window_df: pd.DataFrame) -> dict[str, float]:
    """Extract features from one IMU+PPG window.

    Args:
        window_df: DataFrame with columns including at least:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_ir.
            Index is timestamp_ms (used for length/duration checks).

    Returns:
        Flat dict mapping feature name -> float value.
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
    }

    for name, sig in signals.items():
        feats.update(_time_domain(sig, name))
        feats.update(_freq_domain(sig, name, fs=_FS))

    window_ms = float(window_df.index.max() - window_df.index.min()) if len(window_df) > 1 else 0.0
    ppg_ir = window_df["ppg_ir"].to_numpy(dtype=float)
    feats.update(_ppg_hr_features(ppg_ir, fs=_FS, window_ms=window_ms))

    return feats


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


def _ppg_hr_features(ppg_ir: np.ndarray, fs: float, window_ms: float) -> dict[str, float]:
    """Beat-detect on the cardiac band of the PPG IR channel; derive HR + RMSSD.

    ppg_red is intentionally not used here - no current feature need for it.
    """
    empty = {
        "ppg_hr_mean": float("nan"),
        "ppg_hr_std":  float("nan"),
        "ppg_hr_min":  float("nan"),
        "ppg_hr_max":  float("nan"),
        "ppg_rmssd":   float("nan"),
    }

    n = len(ppg_ir)
    if n < 4:
        return empty

    nyq  = fs / 2.0
    low  = _PPG_BAND_LOW_HZ / nyq
    high = min(_PPG_BAND_HIGH_HZ / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    try:
        filtered = filtfilt(b, a, ppg_ir)
    except ValueError:
        # filtfilt needs more samples than its padlen - window too short to filter.
        return empty

    min_distance = max(1, int(fs * 60.0 / _PPG_MAX_BPM))
    peaks, _ = find_peaks(filtered, distance=min_distance)
    if len(peaks) < 2:
        return empty

    ibi_ms = np.diff(peaks) / fs * 1000.0
    hr_bpm = 60000.0 / ibi_ms

    # RMSSD needs several minutes of beats to mean anything; current windows
    # are shorter, so this comes back NaN for now.
    rmssd = float("nan")
    if window_ms >= _RMSSD_MIN_WINDOW_MS and len(ibi_ms) >= 2:
        rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))

    return {
        "ppg_hr_mean": float(np.mean(hr_bpm)),
        "ppg_hr_std":  float(np.std(hr_bpm)),
        "ppg_hr_min":  float(np.min(hr_bpm)),
        "ppg_hr_max":  float(np.max(hr_bpm)),
        "ppg_rmssd":   rmssd,
    }
