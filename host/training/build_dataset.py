"""
build_dataset.py - turn raw logged sessions into data/processed/features.npz.

For each data/raw/<session_id>.bin, pairs it with <session_id>.markers.csv
(keypress event timestamps) and <session_id>.meta.json (session date, for
day-based CV grouping). Each marker timestamp is expanded into a single
*anticipatory* lookback interval - [marker - buffer_ms - x_window_ms,
marker - buffer_ms) - labeled positive, so the model learns to recognize the
pre-event state rather than the event itself. Negative windows are sampled
separately from stretches that don't overlap any positive interval. See
host.pipeline.segmentation / host.pipeline.features for the shared window
extraction / feature extraction logic.

This is a v0 label source: keypress timing is self-reported, not EMG- or
human-confirmed. See host/labeling/emg_candidate_flagger.py + review_tool.py
for the more precise (but manual) alternative once there's time to review.

Usage:
    python -m host.training.build_dataset
"""
from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from host.pipeline.features import compute_features
from host.pipeline.segmentation import extract_windows

RAW_DIR  = Path("data/raw")
OUT_PATH = Path("data/processed/features.npz")

IMU_PPG_LEN    = 37
PKT_START      = 0xAA
PKT_TYPE_IMU_PPG = 0x01
FMT_IMU_PPG    = "<BBBIhhhhhhIIffBB"
FIELDS_IMU_PPG = (
    "start", "type", "version", "timestamp_ms",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "ppg_ir", "ppg_red",
    "ambient_c", "object_c",
    "checksum", "end",
)

# Anticipatory positive window: [marker - buffer_ms - x_window_ms, marker - buffer_ms).
# buffer_ms absorbs self-report timing imprecision (the keypress lags the
# actual urge/onset by an unknown amount) so the lookback window doesn't
# bleed into the event itself. x_window_ms is the lookback length used for
# both positive and negative windows.
DEFAULT_BUFFER_MS   = 7000
DEFAULT_X_WINDOW_MS = 60000

# Stride for tiling candidate negative windows (see segmentation.extract_windows).
NEG_STEP_MS = DEFAULT_X_WINDOW_MS

# Skip windows with too few samples to produce meaningful features (e.g.
# session start/end edges, dropped-packet gaps). Expressed as a coverage
# fraction of nominal samples rather than a fixed count, since window_ms
# is configurable and a fixed count stops meaning the same thing.
MIN_WINDOW_COVERAGE = 0.5
_NOMINAL_FS_HZ = 50.0


def _min_samples_for_window(window_ms: int) -> int:
    return int(window_ms / 1000.0 * _NOMINAL_FS_HZ * MIN_WINDOW_COVERAGE)


def _load_imu_ppg_packets(bin_path: Path) -> list[dict]:
    """Scan a raw .bin file and return all valid IMU+PPG+Thermal packets.

    Pre-filters candidate start positions with numpy (only ~1/256 of bytes
    equal PKT_START) instead of stepping through every byte in Python -
    matters once sessions run tens of MB.
    """
    buf = bin_path.read_bytes()
    n = len(buf)
    data = np.frombuffer(buf, dtype=np.uint8)
    packets = []
    next_allowed = 0
    for i in np.flatnonzero(data == PKT_START):
        i = int(i)
        if i < next_allowed or i + IMU_PPG_LEN > n:
            continue
        if data[i + 1] != PKT_TYPE_IMU_PPG:
            continue
        chunk = buf[i:i + IMU_PPG_LEN]
        cs = 0
        for b in chunk[1:IMU_PPG_LEN - 2]:
            cs ^= b
        if chunk[IMU_PPG_LEN - 2] != cs or chunk[IMU_PPG_LEN - 1] != 0xBB:
            continue
        vals = struct.unpack(FMT_IMU_PPG, chunk)
        packets.append(dict(zip(FIELDS_IMU_PPG, vals)))
        next_allowed = i + IMU_PPG_LEN
    return packets


def _load_session_day(meta_path: Path, session_id: str) -> str:
    """Return the calendar day (YYYY-MM-DD) a session was recorded on."""
    if not meta_path.exists():
        return "unknown"
    meta = json.loads(meta_path.read_text())
    start_utc = meta.get("start_utc")
    if not start_utc:
        return "unknown"
    return datetime.fromisoformat(start_utc).date().isoformat()


def _markers_to_labels(
    markers_path: Path,
    buffer_ms: int,
    x_window_ms: int,
    n_lookback_shifts: int = 1,
    shift_step_ms: int | None = None,
) -> pd.DataFrame:
    """Expand each keypress marker into its anticipatory lookback interval(s).

    window = [marker - buffer_ms - x_window_ms, marker - buffer_ms)

    This is the pre-event state the model should learn, not the event
    itself - see module docstring.

    n_lookback_shifts > 1 generates additional windows further back from the
    same marker (k=1, 2, ... each shift_step_ms earlier), tagged with a
    "horizon" column (k) so callers can check whether predictive signal
    holds up further from the event. Two distinct uses depending on
    shift_step_ms:
      - shift_step_ms < x_window_ms (e.g. the default, x_window_ms // 3):
        OVERLAPPING windows - re-using already-recorded data at adjacent
        pre-event offsets, not fabricating new signal, but correlated with
        each other, not independent samples.
      - shift_step_ms == x_window_ms: NON-overlapping, back-to-back windows
        tiling further into the past - genuinely distinct stretches of
        recorded time, and a real way to ask "how far ahead can we predict".
    Default of 1 reproduces the original single-window-per-marker behavior
    exactly, with horizon=0 for every row.
    """
    if shift_step_ms is None:
        shift_step_ms = x_window_ms // 3

    columns = ["start_ms", "end_ms", "label", "horizon"]
    if not markers_path.exists():
        return pd.DataFrame(columns=columns)

    markers = pd.read_csv(markers_path)
    if markers.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for ts in markers["timestamp_ms"]:
        for k in range(n_lookback_shifts):
            end_ms = ts - buffer_ms - k * shift_step_ms
            rows.append({"start_ms": end_ms - x_window_ms, "end_ms": end_ms, "label": 1, "horizon": k})
    return pd.DataFrame(rows, columns=columns)


def collect_windows(
    raw_dir: Path = RAW_DIR,
    buffer_ms: int = DEFAULT_BUFFER_MS,
    x_window_ms: int = DEFAULT_X_WINDOW_MS,
    neg_step_ms: int = NEG_STEP_MS,
    n_lookback_shifts: int = 1,
    shift_step_ms: int | None = None,
) -> list:
    """Parse every session and return the raw Window list, pre-feature-extraction.

    Exposed separately from build_dataset() so callers that want to augment
    raw signal (jitter/rotation etc.) before feature extraction can work with
    windows directly instead of only the flat feature table.
    """
    bin_paths = sorted(raw_dir.glob("*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No .bin sessions found under {raw_dir}")

    all_windows = []
    for bin_path in bin_paths:
        session_id   = bin_path.stem
        markers_path = raw_dir / f"{session_id}.markers.csv"
        meta_path    = raw_dir / f"{session_id}.meta.json"

        day_str = _load_session_day(meta_path, session_id)
        labels_df = _markers_to_labels(
            markers_path, buffer_ms, x_window_ms,
            n_lookback_shifts=n_lookback_shifts, shift_step_ms=shift_step_ms,
        )

        print(f"Loading {bin_path.name} (day={day_str}) ...")
        packets = _load_imu_ppg_packets(bin_path)
        if not packets:
            print("  no IMU+PPG packets found, skipping")
            continue

        imu_df = pd.DataFrame(packets).set_index("timestamp_ms").sort_index()
        imu_df._day_str = day_str  # read by segmentation.extract_windows

        windows = extract_windows(imu_df, labels_df, window_ms=x_window_ms, step_ms=neg_step_ms)
        n_positive = sum(w.label for w in windows)
        print(f"  {len(packets)} packets  ->  {len(windows)} windows"
              f"  ({n_positive} positive, {len(windows) - n_positive} negative)")
        all_windows.extend(windows)

    return all_windows


def windows_to_features(windows: list, min_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run compute_features over a Window list, applying the same
    min-samples coverage filter and NaN cleaning build_dataset() uses."""
    feature_names: list[str] | None = None
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for w in windows:
        if len(w.df) < min_samples:
            continue
        feats = compute_features(w.df)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        rows.append(np.array([feats[name] for name in feature_names]))
        labels.append(w.label)
        groups.append(w.day_str)

    if not rows:
        raise RuntimeError("No windows produced - check raw data / markers.")

    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names


def build_dataset(
    raw_dir: Path = RAW_DIR,
    buffer_ms: int = DEFAULT_BUFFER_MS,
    x_window_ms: int = DEFAULT_X_WINDOW_MS,
    neg_step_ms: int = NEG_STEP_MS,
    n_lookback_shifts: int = 1,
    shift_step_ms: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    windows = collect_windows(
        raw_dir, buffer_ms, x_window_ms, neg_step_ms,
        n_lookback_shifts=n_lookback_shifts, shift_step_ms=shift_step_ms,
    )
    min_samples = _min_samples_for_window(x_window_ms)
    X, y, groups_arr, feature_names = windows_to_features(windows, min_samples)

    print(f"\nTotal: X={X.shape}  positive rate={y.mean():.2%}  days={sorted(set(groups_arr))}")
    return X, y, groups_arr, feature_names


def _clean_nans(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop all-NaN feature columns, median-impute any remaining NaNs.

    ppg_rmssd is currently NaN for every window (windows are far shorter
    than the 5-min floor in features._ppg_hr_features) - that column would
    have zero variance anyway, so drop rather than impute it. ppg_hr_* can
    occasionally be NaN too when beat detection fails on a motion-corrupted
    window; impute those with the column median rather than dropping rows,
    since dropping would disproportionately remove positive (motion-heavy)
    windows and shrink an already tiny positive class further.
    """
    nan_frac = np.isnan(X).mean(axis=0)
    all_nan = nan_frac >= 1.0
    if all_nan.any():
        dropped = [feature_names[i] for i in np.flatnonzero(all_nan)]
        print(f"  dropping all-NaN feature column(s): {dropped}")
        X = X[:, ~all_nan]
        feature_names = [f for f, keep in zip(feature_names, ~all_nan) if keep]

    remaining = np.isnan(X)
    if remaining.any():
        n_windows = int(remaining.any(axis=1).sum())
        print(f"  imputing {int(remaining.sum())} remaining NaN value(s)"
              f" across {n_windows} window(s) with per-column median")
        col_medians = np.nanmedian(X, axis=0)
        rows_idx, cols_idx = np.where(remaining)
        X[rows_idx, cols_idx] = col_medians[cols_idx]

    return X, feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-ms", type=int, default=DEFAULT_BUFFER_MS,
                        help="ms gap before each marker excluded from the lookback window")
    parser.add_argument("--x-window-ms", type=int, default=DEFAULT_X_WINDOW_MS,
                        help="length (ms) of each positive/negative lookback window")
    parser.add_argument("--neg-step-ms", type=int, default=NEG_STEP_MS,
                        help="stride (ms) used to tile candidate negative windows")
    parser.add_argument("--n-lookback-shifts", type=int, default=1,
                        help="overlapping lookback windows per marker (1 = off, original behavior)")
    parser.add_argument("--shift-step-ms", type=int, default=None,
                        help="spacing between shifted lookback windows (default: x_window_ms // 3)")
    args = parser.parse_args()

    X, y, groups, feature_names = build_dataset(
        buffer_ms=args.buffer_ms, x_window_ms=args.x_window_ms,
        neg_step_ms=args.neg_step_ms,
        n_lookback_shifts=args.n_lookback_shifts, shift_step_ms=args.shift_step_ms,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, X=X, y=y, groups=groups, feature_names=np.array(feature_names))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
