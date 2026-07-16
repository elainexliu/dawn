"""
segmentation.py — align labels to IMU+PPG timeline and extract fixed windows.

Single source: used by training (offline) and will be used by inference (online).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class Window(NamedTuple):
    df: pd.DataFrame    # rows from the IMU+PPG DataFrame within [start_ms, end_ms)
    label: int          # 0 or 1
    day_str: str        # e.g. "2026-07-07"  — used for day-based CV splits


def extract_windows(
    imu_ppg_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    window_ms: int = 2000,
    step_ms: int = 500,
) -> list[Window]:
    """Slide a window over the IMU+PPG data and assign binary labels.

    Args:
        imu_ppg_df: DataFrame indexed by timestamp_ms (ESP32 clock).
                    Must have been sorted ascending before passing in.
        labels_df:  DataFrame with columns [start_ms, end_ms, label].
                    All timestamps are ESP32 millis() values.
        window_ms:  Window length in milliseconds.
        step_ms:    Stride between successive windows.

    Returns:
        List of Window namedtuples.
    """
    if imu_ppg_df.empty or labels_df.empty:
        return []

    # Derive day_str from the meta.json attached to the session at load time.
    # We can't know the session date here, so callers should attach it.
    # For now we fall back to a placeholder; training.train will set it properly.
    day_str = str(getattr(imu_ppg_df, "_day_str", "unknown"))

    t_min = int(imu_ppg_df.index.min())
    t_max = int(imu_ppg_df.index.max())

    windows: list[Window] = []
    t_start = t_min

    while t_start + window_ms <= t_max:
        t_end = t_start + window_ms
        chunk = imu_ppg_df.loc[t_start:t_end - 1]

        # Assign label: 1 if any labeled event overlaps this window
        label = _assign_label(t_start, t_end, labels_df)
        windows.append(Window(df=chunk.copy(), label=label, day_str=day_str))
        t_start += step_ms

    return windows


def _assign_label(win_start: int, win_end: int, labels_df: pd.DataFrame) -> int:
    """Return 1 if any positive label window overlaps [win_start, win_end)."""
    positive = labels_df[labels_df["label"] == 1]
    overlap = positive[
        (positive["end_ms"] > win_start) & (positive["start_ms"] < win_end)
    ]
    return 1 if not overlap.empty else 0
