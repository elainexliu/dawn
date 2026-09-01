"""
sample_weighting.py - sample weighting and negative-sampling variants,
layered onto whichever model/feature-set wins the main sweep (this module
provides the building blocks; run_sweep.py decides what to apply them to
once that ranking is known).

Two variants:
  1. Downweight "dirty" positives - a positive window whose lookback period
     contains ANOTHER marker (a different event) besides the one that
     defines it. Detected per-SESSION, not per-day: markers.csv timestamps
     are ESP32 millis() since that session's own power-on, not comparable
     across different sessions even on the same calendar day (a day's
     "group" in this project can span multiple independent sessions/clock
     domains - see host/acquisition/receiver.py). Getting this cross-
     session distinction right is why this isn't just a filter on the
     already-built (X, y, groups) arrays; it re-walks sessions directly.
  2. All available negatives (no 1:1 count-matched subsampling), with
     class_weight="balanced" to handle the resulting imbalance, vs. the
     current strict 1:1 subsampling. Uses
     host.pipeline.segmentation._sample_negative_windows directly (bypassing
     extract_windows' hardcoded count-matching to positives) requesting an
     effectively unbounded n_windows, so it returns every valid negative
     candidate instead of a count-matched sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_sweep.feature_variants import feature_set_A
from host.pipeline.segmentation import Window, _sample_negative_windows
from host.training.build_dataset import (
    DEFAULT_BUFFER_MS,
    RAW_DIR,
    _clean_nans,
    _load_imu_ppg_packets,
    _load_session_day,
    _markers_to_labels,
    _min_samples_for_window,
)

X_WINDOW_MS = 60_000
DIRTY_WEIGHT = 0.5
DIRTY_EXCLUDE_OWN_MARKER_TOLERANCE_MS = 1_000  # avoid flagging the defining marker as "another" one


def build_A_with_dirty_tags(x_window_ms: int = X_WINDOW_MS, buffer_ms: int = DEFAULT_BUFFER_MS):
    """Feature set A, plus a same-length boolean 'dirty' array: True for
    positive windows whose lookback period contains a marker other than
    the one defining them. Negatives are always False (not applicable)."""
    bin_paths = sorted(RAW_DIR.glob("*.bin"))
    min_samples = _min_samples_for_window(x_window_ms)
    rows, labels, groups, dirty = [], [], [], []
    feature_names = None

    for bin_path in bin_paths:
        session_id = bin_path.stem
        markers_path = RAW_DIR / f"{session_id}.markers.csv"
        meta_path = RAW_DIR / f"{session_id}.meta.json"
        day_str = _load_session_day(meta_path, session_id)

        marker_ts = np.array([])
        if markers_path.exists():
            markers_df = pd.read_csv(markers_path)
            if not markers_df.empty:
                marker_ts = markers_df["timestamp_ms"].to_numpy()

        labels_df = _markers_to_labels(markers_path, buffer_ms, x_window_ms)
        packets = _load_imu_ppg_packets(bin_path)
        if not packets:
            continue
        imu_df = pd.DataFrame(packets).set_index("timestamp_ms").sort_index()
        imu_df._day_str = day_str

        from host.pipeline.segmentation import extract_windows
        windows = extract_windows(imu_df, labels_df, window_ms=x_window_ms, step_ms=x_window_ms)

        for w in windows:
            if len(w.df) < min_samples:
                continue
            feats = feature_set_A(w.df)
            if feature_names is None:
                feature_names = sorted(feats.keys())
            rows.append([feats[k] for k in feature_names])
            labels.append(w.label)
            groups.append(day_str)

            if w.label == 1 and len(marker_ts):
                start_ms, end_ms = int(w.df.index.min()), int(w.df.index.max()) + 1
                owning_marker_ts = end_ms + buffer_ms  # approx location of the marker this window anticipates
                in_window = (marker_ts >= start_ms) & (marker_ts < end_ms)
                not_owning = np.abs(marker_ts - owning_marker_ts) > DIRTY_EXCLUDE_OWN_MARKER_TOLERANCE_MS
                dirty.append(bool(np.any(in_window & not_owning)))
            else:
                dirty.append(False)

    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    dirty_arr = np.array(dirty)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names, dirty_arr


def dirty_sample_weights(y: np.ndarray, dirty: np.ndarray, dirty_weight: float = DIRTY_WEIGHT) -> np.ndarray:
    return np.where((y == 1) & dirty, dirty_weight, 1.0)


def build_A_all_negatives(x_window_ms: int = X_WINDOW_MS, buffer_ms: int = DEFAULT_BUFFER_MS):
    """Feature set A, but with EVERY valid negative candidate (no 1:1
    count-matching to positives) - for comparing against class-weighted LR
    instead of subsampling."""
    bin_paths = sorted(RAW_DIR.glob("*.bin"))
    min_samples = _min_samples_for_window(x_window_ms)
    rows, labels, groups = [], [], []
    feature_names = None

    for bin_path in bin_paths:
        session_id = bin_path.stem
        markers_path = RAW_DIR / f"{session_id}.markers.csv"
        meta_path = RAW_DIR / f"{session_id}.meta.json"
        day_str = _load_session_day(meta_path, session_id)
        labels_df = _markers_to_labels(markers_path, buffer_ms, x_window_ms)

        packets = _load_imu_ppg_packets(bin_path)
        if not packets:
            continue
        imu_df = pd.DataFrame(packets).set_index("timestamp_ms").sort_index()
        imu_df._day_str = day_str

        positive_intervals = labels_df[labels_df["label"] == 1]
        pos_windows = []
        for _, row in positive_intervals.iterrows():
            start_ms, end_ms = int(row["start_ms"]), int(row["end_ms"])
            chunk = imu_df.loc[start_ms:end_ms - 1]
            if not chunk.empty:
                pos_windows.append(Window(df=chunk.copy(), label=1, day_str=day_str))

        # Request an effectively unbounded number of negatives - _sample_negative_windows
        # caps at however many valid non-overlapping candidates actually exist.
        neg_windows = _sample_negative_windows(
            imu_df, positive_intervals, window_ms=x_window_ms, step_ms=x_window_ms,
            n_windows=10**6, day_str=day_str, seed=42,
        )

        for w in pos_windows + neg_windows:
            if len(w.df) < min_samples:
                continue
            feats = feature_set_A(w.df)
            if feature_names is None:
                feature_names = sorted(feats.keys())
            rows.append([feats[k] for k in feature_names])
            labels.append(w.label)
            groups.append(day_str)

    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names
