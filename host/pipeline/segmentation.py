"""
segmentation.py - build anticipatory windows from IMU+PPG data and marker-
derived event intervals.

Single source: used by training (offline) and will be used by inference (online).

Positive windows are NOT "does this window overlap the event" (that trains a
detector, not a predictor). Each positive window is the single fixed-length
lookback window ending *before* the event, as already computed by the caller
into labels_df's [start_ms, end_ms) intervals (see
host/training/build_dataset.py:_markers_to_labels) - the model is meant to
recognize the pre-event state, not the event itself.

Negative windows are sampled separately from stretches that don't overlap any
positive interval, count-matched to the number of positives.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class Window(NamedTuple):
    df: pd.DataFrame    # rows from the IMU+PPG DataFrame within [start_ms, end_ms)
    label: int          # 0 or 1
    day_str: str        # e.g. "2026-07-07"  - used for day-based CV splits
    horizon: int = 0    # 0 = nearest lookback window to the event; >0 = further
                        # back in time (see build_dataset._markers_to_labels'
                        # n_lookback_shifts) - lets callers check whether
                        # predictive signal holds up further from the event


def extract_windows(
    imu_ppg_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    window_ms: int = 60000,
    step_ms: int = 60000,
    seed: int = 42,
) -> list[Window]:
    """Build one window per positive labels_df interval, plus a count-matched
    sample of negative windows that avoid every positive interval.

    Args:
        imu_ppg_df: DataFrame indexed by timestamp_ms (ESP32 clock).
                    Must have been sorted ascending before passing in.
        labels_df:  DataFrame with columns [start_ms, end_ms, label]. Each
                    label==1 row is already the exact pre-event lookback
                    interval (window_ms long) - this function does not slide
                    or search for positives, it just slices them out.
        window_ms:  Expected length of each positive interval, and the length
                    of every sampled negative window.
        step_ms:    Stride used only to tile candidate negative windows.
        seed:       RNG seed for reproducible negative sampling.

    Returns:
        List of Window namedtuples - all positives first, then negatives.
    """
    if imu_ppg_df.empty:
        return []

    # Derive day_str from the meta.json attached to the session at load time.
    # We can't know the session date here, so callers should attach it.
    day_str = str(getattr(imu_ppg_df, "_day_str", "unknown"))

    positive_intervals = labels_df[labels_df["label"] == 1]

    windows: list[Window] = []
    for _, row in positive_intervals.iterrows():
        start_ms, end_ms = int(row["start_ms"]), int(row["end_ms"])
        chunk = imu_ppg_df.loc[start_ms:end_ms - 1]
        if chunk.empty:
            continue
        horizon = int(row["horizon"]) if "horizon" in row and not pd.isna(row["horizon"]) else 0
        windows.append(Window(df=chunk.copy(), label=1, day_str=day_str, horizon=horizon))

    negatives = _sample_negative_windows(
        imu_ppg_df, positive_intervals,
        window_ms=window_ms, step_ms=step_ms,
        n_windows=len(windows), day_str=day_str, seed=seed,
    )
    windows.extend(negatives)

    return windows


def _sample_negative_windows(
    imu_ppg_df: pd.DataFrame,
    positive_intervals: pd.DataFrame,
    window_ms: int,
    step_ms: int,
    n_windows: int,
    day_str: str,
    seed: int,
) -> list[Window]:
    """Sample up to n_windows windows of length window_ms that don't overlap
    any positive interval's [start_ms, end_ms) region ("far enough from all
    markers"), tiled at step_ms and chosen at random for reproducibility.
    """
    if n_windows <= 0:
        return []

    t_min = int(imu_ppg_df.index.min())
    t_max = int(imu_ppg_df.index.max())
    exclusions = list(zip(positive_intervals["start_ms"].astype(int),
                          positive_intervals["end_ms"].astype(int)))

    candidates: list[int] = []
    t_start = t_min
    while t_start + window_ms <= t_max:
        t_end = t_start + window_ms
        if not any(t_end > ex_start and t_start < ex_end for ex_start, ex_end in exclusions):
            candidates.append(t_start)
        t_start += step_ms

    if not candidates:
        return []

    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(n_windows, len(candidates)), replace=False)

    windows: list[Window] = []
    for t_start in sorted(int(c) for c in chosen):
        chunk = imu_ppg_df.loc[t_start:t_start + window_ms - 1]
        if chunk.empty:
            continue
        windows.append(Window(df=chunk.copy(), label=0, day_str=day_str))
    return windows
