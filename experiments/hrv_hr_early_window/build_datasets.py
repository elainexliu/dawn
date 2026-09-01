"""
build_datasets.py - assembles the 4 feature configs for this experiment.

Feature set A (IMU-only baseline) reuses collect_windows() +
windows_to_features() exactly as they exist in host/training/build_dataset.py
(60s window, matching the production pipeline).

For the HR-enriched configs, each anchor needs a 180s window (to derive the
clean early segment from) ending at the SAME point as its 60s IMU window.
collect_windows() only supports one window length per call, so this module
collects both lengths per anchor directly, reusing the same lower-level
building blocks (extract_windows and build_dataset.py's private loaders)
that collect_windows() is built from - same technique model_sweep's
multi-horizon feature sets use, reimplemented here to keep this experiment
self-contained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hrv_hr_early_window.clean_segment_features import clean_segment_cardiac_features
from host.pipeline.features import compute_features
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

X_WINDOW_MS = 60_000    # matches the production IMU baseline
LARGE_WINDOW_MS = 180_000  # the lookback window the clean early segment is drawn from

ACCEL_GYRO_PREFIXES = ("accel_", "gyro_")


def _select_accel_gyro(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith(ACCEL_GYRO_PREFIXES)]


def build_feature_set_A() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Standard 60s-window IMU baseline - exact reuse, no modification."""
    windows = collect_windows()  # defaults: x_window_ms=60000, matching production
    min_samples = _min_samples_for_window(X_WINDOW_MS)
    X, y, groups, names = windows_to_features(windows, min_samples)
    idx = _select_accel_gyro(names)
    return X[:, idx], y, groups, [names[i] for i in idx]


def collect_dual_window_anchors(
    raw_dir: Path = RAW_DIR,
    buffer_ms: int = DEFAULT_BUFFER_MS,
    small_ms: int = X_WINDOW_MS,
    large_ms: int = LARGE_WINDOW_MS,
) -> list[dict]:
    """For every anchor valid at the LARGE window length, also slice the
    SAME imu_df at the small window length ending at the same point.
    Vetting (exclusion zones, negative sampling, coverage) is driven by the
    large length via extract_windows(), same as model_sweep's approach.

    Returns a list of dicts: {small_df, large_df, label, day_str, end_ms}.
    """
    bin_paths = sorted(Path(raw_dir).glob("*.bin"))
    results = []

    for bin_path in bin_paths:
        session_id = bin_path.stem
        markers_path = Path(raw_dir) / f"{session_id}.markers.csv"
        meta_path = Path(raw_dir) / f"{session_id}.meta.json"
        day_str = _load_session_day(meta_path, session_id)
        labels_df = _markers_to_labels(markers_path, buffer_ms, large_ms)

        packets = _load_imu_ppg_packets(bin_path)
        if not packets:
            continue
        imu_df = pd.DataFrame(packets).set_index("timestamp_ms").sort_index()
        imu_df._day_str = day_str

        windows = extract_windows(imu_df, labels_df, window_ms=large_ms, step_ms=large_ms)
        for w in windows:
            if w.df.empty:
                continue
            end_ms = int(w.df.index.max()) + 1
            small_df = imu_df.loc[end_ms - small_ms: end_ms - 1]
            if len(small_df) < _min_samples_for_window(small_ms):
                continue
            results.append({
                "small_df": small_df, "large_df": w.df, "label": w.label,
                "day_str": day_str, "end_ms": end_ms,
            })

    return results


def build_all_configs():
    """Returns:
      X_A_full, y_A_full, groups_A_full, names_A_full  - full-size reference baseline
      restricted: dict of {config_name: (X, y, groups, names)} on the COMMON subset
        of anchors that have valid 60s+180s data AND pass HR/HRV quality gates
      quality_report: dict summarizing the funnel of anchors through each gate
    """
    print("Building feature set A (full reference, 60s window, standard pipeline) ...")
    X_a_full, y_a_full, groups_a_full, names_a_full = build_feature_set_A()
    print(f"  X_A full reference: {X_a_full.shape}  positive_rate={y_a_full.mean():.2%}")

    print("\nCollecting dual-window (60s + 180s) anchors ...")
    anchors = collect_dual_window_anchors()
    print(f"  {len(anchors)} anchors have both valid 60s and 180s windows")

    imu_rows, imu_names = [], None
    hr_trend_rows, hrv_rows, labels, groups, ends = [], [], [], [], []
    quality_log = []

    for a in anchors:
        imu_feats_full = compute_features(a["small_df"])
        imu_feats = {k: v for k, v in imu_feats_full.items() if k.startswith(ACCEL_GYRO_PREFIXES)}
        if imu_names is None:
            imu_names = sorted(imu_feats.keys())
        imu_rows.append([imu_feats[k] for k in imu_names])

        result = clean_segment_cardiac_features(a["large_df"], a["end_ms"])
        hr_trend_rows.append([result["features"]["hr_trend_mean"], result["features"]["hr_trend_slope"]])
        hrv_rows.append([result["features"]["rmssd_clean"], result["features"]["sdnn_clean"]])
        labels.append(a["label"])
        groups.append(a["day_str"])
        ends.append(a["end_ms"])
        quality_log.append(result["quality"])

    X_imu = np.array(imu_rows, dtype=float)
    X_imu, imu_names = _clean_nans(X_imu, imu_names)
    X_hr_trend = np.array(hr_trend_rows, dtype=float)
    X_hrv = np.array(hrv_rows, dtype=float)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)

    trend_ok = np.array([q["trend_ok"] for q in quality_log])
    hrv_ok = np.array([q["hrv_ok"] for q in quality_log])
    plausible = np.array([q["plausible_hr"] for q in quality_log])

    quality_report = {
        "n_dual_window_anchors": len(anchors),
        "n_plausible_hr": int(plausible.sum()),
        "n_trend_ok": int(trend_ok.sum()),
        "n_hrv_ok": int(hrv_ok.sum()),
        "n_both_ok": int((trend_ok & hrv_ok).sum()),
        "n_full_reference_A": int(X_a_full.shape[0]),
    }

    # Common subset for the fair comparison: anchors passing BOTH quality
    # gates, so all four configs are evaluated on the identical population.
    common_mask = trend_ok & hrv_ok
    n_common = int(common_mask.sum())
    print(f"\nQuality funnel: {len(anchors)} dual-window anchors -> "
          f"{quality_report['n_plausible_hr']} plausible HR -> "
          f"{quality_report['n_trend_ok']} trend-OK, {quality_report['n_hrv_ok']} HRV-OK -> "
          f"{n_common} pass BOTH (common subset used below)")

    configs = {
        "A_only":            X_imu[common_mask],
        "A_plus_hr_trend":   np.hstack([X_imu[common_mask], X_hr_trend[common_mask]]),
        "A_plus_hrv":        np.hstack([X_imu[common_mask], X_hrv[common_mask]]),
        "A_plus_both":       np.hstack([X_imu[common_mask], X_hr_trend[common_mask], X_hrv[common_mask]]),
    }
    y_common = y[common_mask]
    groups_common = groups_arr[common_mask]

    restricted = {name: (X, y_common, groups_common) for name, X in configs.items()}

    return (X_a_full, y_a_full, groups_a_full, names_a_full), restricted, quality_report, quality_log


if __name__ == "__main__":
    (X_a, y_a, g_a, n_a), restricted, quality_report, quality_log = build_all_configs()
    print("\nFull reference A:", X_a.shape)
    for name, (X, y, g) in restricted.items():
        print(f"{name}: {X.shape}  positive_rate={y.mean():.2%}  days={sorted(set(g))}")
    print("\nQuality report:", quality_report)
