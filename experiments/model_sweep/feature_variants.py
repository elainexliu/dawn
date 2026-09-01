"""
feature_variants.py - composable feature-extraction functions for four
feature-set variants (A/B/C/D below).

All four reuse host/pipeline/ and host/training/ (compute_features,
extract_windows, collect_windows, windows_to_features, and a few private
loaders) without modifying anything there.

  A. Baseline: compute_features()'s accel/gyro output unchanged (54
     features: mean/std/min/max/rms/zcr/dominant_freq/power_0_5hz/
     power_5_15hz per axis), single 60s window. The FFT-derived stats
     were excluded in experiments/informed_prior/ only because they don't
     transfer cross-dataset at a different sampling rate - that concern
     doesn't apply here (personal data only), so they're included normally.
  B. Multi-horizon: (A)'s feature computation applied independently at
     three window LENGTHS (30s/60s/180s), all ending at the same
     anticipation point (event_start - buffer_ms), concatenated into one
     162-feature vector per sample. Column names prefixed w30_/w60_/w180_.
  C. Enriched single-window: (A) + jerk (d(accel)/dt) time-domain stats
     (mean/std/min/max/rms/zcr per axis, 18 features), 60s window, 72 total.
  D. Enriched multi-horizon: (B) + jerk at each of the three window
     lengths, 216 features total (72 x 3, w30_/w60_/w180_ prefixed).

Window validity/exclusion-zone logic is always driven by the LARGEST
window length in play (180s for B/D, 60s for A/C) - a window ending at a
given point that's valid at the largest length is automatically valid
(same anchor, non-overlapping, adequate coverage) at any shorter length
ending at that same point, since a shorter window is a strict suffix of
the longer one already vetted by extract_windows().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from host.pipeline.features import _ACCEL_SCALE, _FS, _time_domain, compute_features
from host.pipeline.segmentation import extract_windows
from host.training.build_dataset import (
    DEFAULT_BUFFER_MS,
    RAW_DIR,
    _clean_nans,
    _load_imu_ppg_packets,
    _load_session_day,
    _markers_to_labels,
    _min_samples_for_window,
    collect_windows,
    windows_to_features,
)

HORIZON_LENGTHS_MS = [30_000, 60_000, 180_000]
ACCEL_GYRO_PREFIXES = ("accel_", "gyro_")


def _select_accel_gyro(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith(ACCEL_GYRO_PREFIXES)]


def _jerk_features(window_df: pd.DataFrame, fs: float) -> dict[str, float]:
    """d(accel)/dt time-domain stats - same 6 stats as compute_features'
    _time_domain, same helper function, applied to a differenced signal.
    accel columns are scaled to g first (matching compute_features'
    internal convention) before differencing.
    """
    feats: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        accel_g = window_df[f"accel_{axis}"].to_numpy(dtype=float) * _ACCEL_SCALE
        jerk = np.diff(accel_g) * fs if len(accel_g) > 1 else np.array([0.0])
        feats.update(_time_domain(jerk, f"jerk_{axis}"))
    return feats


def feature_set_A(window_df: pd.DataFrame) -> dict[str, float]:
    feats = compute_features(window_df)
    return {k: v for k, v in feats.items() if k.startswith(ACCEL_GYRO_PREFIXES)}


def feature_set_C(window_df: pd.DataFrame) -> dict[str, float]:
    feats = feature_set_A(window_df)
    feats.update(_jerk_features(window_df, fs=_FS))
    return feats


def feature_set_B(multi_window_dfs: dict[int, pd.DataFrame]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for length_ms in HORIZON_LENGTHS_MS:
        sub = feature_set_A(multi_window_dfs[length_ms])
        feats.update({f"w{length_ms // 1000}_{k}": v for k, v in sub.items()})
    return feats


def feature_set_D(multi_window_dfs: dict[int, pd.DataFrame]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for length_ms in HORIZON_LENGTHS_MS:
        sub = feature_set_C(multi_window_dfs[length_ms])
        feats.update({f"w{length_ms // 1000}_{k}": v for k, v in sub.items()})
    return feats


def collect_multi_window_anchors(
    raw_dir: Path = RAW_DIR,
    buffer_ms: int = DEFAULT_BUFFER_MS,
    lengths_ms: list[int] = HORIZON_LENGTHS_MS,
    neg_step_ms: int | None = None,
) -> list[tuple[dict[int, pd.DataFrame], int, str, int]]:
    """Returns (multi_window_dfs, label, day_str, end_ms) per valid anchor.
    Vetting (exclusion zones, negative sampling, coverage) is driven by the
    largest length via extract_windows() - reused unmodified - then the
    same imu_df is re-sliced at the other lengths ending at the same point.
    """
    largest_ms = max(lengths_ms)
    neg_step_ms = neg_step_ms or largest_ms
    bin_paths = sorted(Path(raw_dir).glob("*.bin"))
    results = []

    for bin_path in bin_paths:
        session_id = bin_path.stem
        markers_path = Path(raw_dir) / f"{session_id}.markers.csv"
        meta_path = Path(raw_dir) / f"{session_id}.meta.json"
        day_str = _load_session_day(meta_path, session_id)
        labels_df = _markers_to_labels(markers_path, buffer_ms, largest_ms)

        packets = _load_imu_ppg_packets(bin_path)
        if not packets:
            continue
        imu_df = pd.DataFrame(packets).set_index("timestamp_ms").sort_index()
        imu_df._day_str = day_str

        windows = extract_windows(imu_df, labels_df, window_ms=largest_ms, step_ms=neg_step_ms)
        for w in windows:
            if w.df.empty:
                continue
            end_ms = int(w.df.index.max()) + 1
            multi = {}
            ok = True
            for L in lengths_ms:
                sub = imu_df.loc[end_ms - L: end_ms - 1]
                if len(sub) < _min_samples_for_window(L):
                    ok = False
                    break
                multi[L] = sub
            if ok:
                results.append((multi, w.label, day_str, end_ms))

    return results


def _extract_xyz(windows, feature_fn, min_samples) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Windows -> (X, y, groups, feature_names, end_timestamps). Mirrors
    host/training/build_dataset.py:windows_to_features's NaN-cleaning
    (reused via _clean_nans) but with a pluggable feature function."""
    rows, labels, groups, timestamps = [], [], [], []
    feature_names = None
    for w in windows:
        if len(w.df) < min_samples:
            continue
        feats = feature_fn(w.df)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        rows.append([feats[k] for k in feature_names])
        labels.append(w.label)
        groups.append(w.day_str)
        timestamps.append(int(w.df.index.max()) + 1 if len(w.df) else -1)

    if not rows:
        raise RuntimeError("No windows produced.")
    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names, np.array(timestamps)


def _extract_xyz_multiwindow(anchors, feature_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    rows, labels, groups, timestamps = [], [], [], []
    feature_names = None
    for multi, label, day_str, end_ms in anchors:
        feats = feature_fn(multi)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        rows.append([feats[k] for k in feature_names])
        labels.append(label)
        groups.append(day_str)
        timestamps.append(end_ms)

    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names, np.array(timestamps)


def build_A():
    windows = collect_windows()
    min_samples = _min_samples_for_window(60_000)
    return _extract_xyz(windows, feature_set_A, min_samples)


def build_C():
    windows = collect_windows()
    min_samples = _min_samples_for_window(60_000)
    return _extract_xyz(windows, feature_set_C, min_samples)


def build_B():
    anchors = collect_multi_window_anchors(lengths_ms=HORIZON_LENGTHS_MS)
    return _extract_xyz_multiwindow(anchors, feature_set_B)


def build_D():
    anchors = collect_multi_window_anchors(lengths_ms=HORIZON_LENGTHS_MS)
    return _extract_xyz_multiwindow(anchors, feature_set_D)


FEATURE_SET_BUILDERS = {"A": build_A, "B": build_B, "C": build_C, "D": build_D}


if __name__ == "__main__":
    for name, builder in FEATURE_SET_BUILDERS.items():
        X, y, groups, feature_names, ts = builder()
        print(f"{name}: X={X.shape}  positive_rate={y.mean():.2%}  days={sorted(set(groups))}")
