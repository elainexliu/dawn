"""
public_features.py - parse the public BFRBAnticipationDataset (Searle et
al., MobileHCI 2021, github.com/Bhorda/BFRBAnticipationDataset) into
feature vectors using the SAME feature-computation logic as our own
pipeline (host/pipeline/features.py), so the resulting population model's
coefficients live in the same feature space as our personal model's.

Data facts below were verified by directly inspecting the raw CSVs and the
dataset's own pipeline scripts (public_dataset/BFRB_Detection_Data/pipeline/
*.py) - not assumed from the README:

  - Accelerometer is in m/s^2 (mean magnitude ~9.9, matches gravity) ->
    converted to g here (/9.80665) to match our _ACCEL_SCALE convention.
  - Gyroscope is already in deg/s (max ~1189; rad/s would be physically
    impossible for a wrist) -> no conversion applied.
  - Sample rate is ~10 Hz (median inter-sample gap 100ms), NOT our 50 Hz.
    This matters: power_5_15hz needs frequency content up to 15Hz, but the
    Nyquist limit at 10Hz sampling is only 5Hz. That feature is structurally
    unmeasurable here - see NOTE below on how this is handled, not hidden.
  - timestamps.csv event times use MM.SS-as-decimal notation, not true
    decimal minutes (e.g. "7.52" = 7 min 52 sec, not 7.52 min) - the
    conversion formula below is copied exactly from their own
    1+_WindowSplit.py / 1-_WindowSplit.py to parse this correctly.
  - Their own windows span [event_start - x, event_start + y] (bleeding
    slightly into the event). We use y=0 (window ends exactly at the
    verified onset) - their onsets are video-verified ground truth, unlike
    our self-reported keypress timing that buffer_ms=7000 exists to guard
    against. There's no self-report lag here to buffer for.
  - Behavior types (skin picking, face touching, fidgeting, skin biting,
    hand scratching, leg scratching, nail biting) are pooled into one
    binary label, matching how our own pipeline treats behavior types
    (splitting by type didn't help on our data either, in an earlier check).

host/pipeline/features.py hardcodes fs=50Hz internally and isn't
parameterized, and it's shared with the training pipeline so I didn't want
to touch it just for this. Instead this imports the same underlying
_time_domain / _freq_domain helpers and calls them with the correct fs per
dataset. See _compute_imu_features() and the parity check in
verify_parity(), which confirms this reproduces compute_features()'s own
accel/gyro output exactly when fed the same scaled inputs and fs=50.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from host.pipeline.features import _ACCEL_SCALE, _FS, _GYRO_SCALE, _freq_domain, _time_domain, compute_features

PUBLIC_DATA_DIR = Path(__file__).parent / "public_dataset" / "BFRB_Detection_Data" / "data"
G = 9.80665  # standard gravity: m/s^2 -> g

X_WINDOW_MS = 60_000     # matches host/training/build_dataset.py's DEFAULT_X_WINDOW_MS
PUBLIC_FS_HZ = 10.0      # empirically measured (median inter-sample gap = 100ms)
EXCLUDED_STAGE = 7       # excluded by the dataset's own WindowSplit scripts

ACCEL_GYRO_CHANNELS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")

# exp-5/exp-6 and exp-8/exp-9's timestamps.csv files are cross-assigned in
# the public repo: exp-5's declared recording "start" (1582118074000)
# matches exp-6's actual sensor-data start almost to the millisecond
# (1582118074035), and vice versa - same pattern for exp-8/exp-9. This
# maps each participant's SENSOR data to the timestamps.csv that actually
# describes it. Verified via near-exact absolute timestamp matches, not
# assumed - see experiments/informed_prior/results.md for the numbers.
TIMESTAMPS_SOURCE_OVERRIDE = {
    "exp-5": "exp-6", "exp-6": "exp-5",
    "exp-8": "exp-9", "exp-9": "exp-8",
}

# All three FFT-derived host/pipeline/features.py stats (power_0_5hz,
# power_5_15hz, dominant_freq) are excluded from cross-dataset prior
# transfer. mean/std/min/max/rms/zcr are kept - they're statistics of the
# signal's own raw amplitude/shape, genuinely comparable in physical units
# (g, deg/s) regardless of sampling rate.
#
# Why the FFT-derived ones aren't safe, found empirically across two
# iterations of this pipeline, not assumed up front:
#
# 1. power_5_15hz needs frequency content up to 15Hz; Nyquist at 10Hz
#    sampling is 5Hz, so it's structurally unmeasurable in this dataset -
#    not just noisy, literally outside what a 10Hz signal can represent.
#
# 2. power_0_5hz/power_5_15hz are computed as an UNNORMALIZED FFT-magnitude
#    sum (sum(|rfft(sig)|**2) * df), whose magnitude scales with the
#    number of samples in the window - ~5x different between our
#    50Hz/~3000-sample windows and this dataset's 10Hz/~600-sample ones.
#    First run of this pipeline produced power_0_5hz prior coefficients of
#    magnitude 10-20 against every other feature's <1 - this artifact
#    dominating the prior, not a real population effect.
#
# 3. dominant_freq looked safe in principle (a frequency in Hz means the
#    same physical thing regardless of sample rate) but ISN'T in practice:
#    its raw VALUE is comparable, but its VARIANCE across windows is not -
#    it's bounded by a 5Hz Nyquist ceiling in the public data vs. 25Hz in
#    ours, and our raw-unit conversion divides by each dataset's own
#    standard deviation for that feature. A second run of this pipeline
#    produced dominant_freq prior coefficients of magnitude ~11 against
#    everything else's <1 - same class of artifact as #2, different cause.
#    Any feature whose within-dataset spread differs greatly between the
#    two datasets is at risk of this, which frequency-domain features are
#    especially prone to (their scale is tied to sampling rate/window
#    length, not just the underlying physical quantity).
UNRELIABLE_FEATURES = tuple(
    f"{ch}_{stat}" for ch in ACCEL_GYRO_CHANNELS for stat in ("dominant_freq", "power_0_5hz", "power_5_15hz")
)


def _compute_imu_features(window_df: pd.DataFrame, fs: float) -> dict[str, float]:
    """accel/gyro time+freq-domain features via the exact same helper
    functions compute_features() uses internally, parameterized with the
    correct fs for this dataset. window_df's accel_*/gyro_* columns must
    already be in final physical units (g, deg/s) - no scaling applied here.
    """
    feats: dict[str, float] = {}
    for ch in ACCEL_GYRO_CHANNELS:
        sig = window_df[ch].to_numpy(dtype=float)
        feats.update(_time_domain(sig, ch))
        feats.update(_freq_domain(sig, ch, fs=fs))
    return feats


def verify_parity() -> bool:
    """_compute_imu_features(fs=50) on scaled synthetic data must exactly
    match compute_features()'s accel/gyro keys - proves we're reusing
    identical formulas, not a subtly different reimplementation."""
    rng = np.random.default_rng(0)
    n = 300
    raw = pd.DataFrame({
        "accel_x": rng.normal(0, 2000, n), "accel_y": rng.normal(0, 2000, n), "accel_z": rng.normal(16000, 800, n),
        "gyro_x": rng.normal(0, 300, n), "gyro_y": rng.normal(0, 300, n), "gyro_z": rng.normal(0, 300, n),
        "ppg_ir": rng.normal(50000, 500, n),
    }, index=pd.RangeIndex(0, n * 20, 20))

    full = compute_features(raw)

    scaled = pd.DataFrame({
        "accel_x": raw["accel_x"] * _ACCEL_SCALE, "accel_y": raw["accel_y"] * _ACCEL_SCALE, "accel_z": raw["accel_z"] * _ACCEL_SCALE,
        "gyro_x": raw["gyro_x"] * _GYRO_SCALE, "gyro_y": raw["gyro_y"] * _GYRO_SCALE, "gyro_z": raw["gyro_z"] * _GYRO_SCALE,
    })
    manual = _compute_imu_features(scaled, fs=_FS)

    ok = all(k in full and abs(manual[k] - full[k]) < 1e-9 for k in manual)
    print(f"Parity check ({len(manual)} accel/gyro features): {'PASSED' if ok else 'FAILED'}")
    if not ok:
        for k in manual:
            if k not in full or abs(manual[k] - full[k]) >= 1e-9:
                print(f"  MISMATCH: {k}  manual={manual.get(k)}  compute_features={full.get(k)}")
    return ok


def _parse_timestamp(raw_val: float, start_recording_ms: int) -> int:
    """Convert their MM.SS-decimal event time into absolute ms - formula
    copied exactly from 1+_WindowSplit.py / 1-_WindowSplit.py."""
    minutes = math.floor(raw_val)
    seconds_hundredths = raw_val - minutes
    return int(start_recording_ms + minutes * 60 * 1000 + seconds_hundredths * 100 * 1000)


def _load_participant(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Returns (accel_df, gyro_df) in physical units (g, deg/s), indexed by
    timestamp_ms, plus a list of valid BFRB event dicts {start_ms}."""
    acc = pd.read_csv(exp_dir / "accLabeled.csv").rename(columns={"x": "accel_x", "y": "accel_y", "z": "accel_z"})
    gyr = pd.read_csv(exp_dir / "gyrLabeled.csv").rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})

    for col in ("accel_x", "accel_y", "accel_z"):
        acc[col] = acc[col] / G
    # gyro already in deg/s - no conversion

    acc = acc.set_index("timestamp").sort_index()
    gyr = gyr.set_index("timestamp").sort_index()

    timestamps_source = TIMESTAMPS_SOURCE_OVERRIDE.get(exp_dir.name, exp_dir.name)
    timecodes = pd.read_csv(exp_dir.parent / timestamps_source / "timestamps.csv")
    start_recording = int(timecodes["start"].iloc[0])

    events_df = timecodes.iloc[1:].reset_index(drop=True)
    events = []
    for _, row in events_df.iterrows():
        if pd.isna(row["label"]) or row["label"] == "" or row["stage"] == EXCLUDED_STAGE:
            continue
        start_ms = _parse_timestamp(float(row["start"]), start_recording)
        if start_ms - X_WINDOW_MS < start_recording:
            continue  # not enough lookback data before this event
        events.append({"start_ms": start_ms, "behavior": row["label"]})

    return acc, gyr, events


def _window_features(acc: pd.DataFrame, gyr: pd.DataFrame, end_ms: int) -> dict[str, float] | None:
    """Extract [end_ms - X_WINDOW_MS, end_ms) from both streams and compute
    accel/gyro features. Accel and gyro are independently-timestamped
    series (not sample-aligned) at similar rates; compute_features'
    per-channel stats don't require row correspondence between channels
    (each column is processed as its own 1D array), so we align by
    position after independently slicing each stream to the same time
    window, truncating to the shorter of the two lengths.
    """
    start_ms = end_ms - X_WINDOW_MS
    acc_win = acc.loc[start_ms:end_ms - 1]
    gyr_win = gyr.loc[start_ms:end_ms - 1]
    n = min(len(acc_win), len(gyr_win))
    # require reasonable coverage: at ~10Hz, 60s nominal ~= 600 samples;
    # mirror build_dataset.py's coverage-fraction philosophy (>=50%)
    if n < int(X_WINDOW_MS / 1000 * PUBLIC_FS_HZ * 0.5):
        return None

    df = pd.DataFrame({
        "accel_x": acc_win["accel_x"].to_numpy()[:n], "accel_y": acc_win["accel_y"].to_numpy()[:n], "accel_z": acc_win["accel_z"].to_numpy()[:n],
        "gyro_x": gyr_win["gyro_x"].to_numpy()[:n], "gyro_y": gyr_win["gyro_y"].to_numpy()[:n], "gyro_z": gyr_win["gyro_z"].to_numpy()[:n],
    })
    return _compute_imu_features(df, fs=PUBLIC_FS_HZ)


def build_public_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns X (raw physical units, NOT standardized), y, participant_id
    groups, and feature_names - pooled across all participant directories
    found under PUBLIC_DATA_DIR.
    """
    exp_dirs = sorted(PUBLIC_DATA_DIR.glob("exp-*"))
    if not exp_dirs:
        raise FileNotFoundError(f"No exp-* participant directories found under {PUBLIC_DATA_DIR}")

    feature_names: list[str] | None = None
    rows, labels, groups = [], [], []

    for exp_dir in exp_dirs:
        pid = exp_dir.name
        acc, gyr, events = _load_participant(exp_dir)
        t_min = max(acc.index.min(), gyr.index.min())
        t_max = min(acc.index.max(), gyr.index.max())

        n_pos = 0
        event_starts = [e["start_ms"] for e in events]
        for e in events:
            feats = _window_features(acc, gyr, e["start_ms"])
            if feats is None:
                continue
            if feature_names is None:
                feature_names = sorted(feats.keys())
            rows.append([feats[k] for k in feature_names])
            labels.append(1)
            groups.append(pid)
            n_pos += 1

        # Count-matched negatives: random anchor points, at least X_WINDOW_MS
        # after recording start, whose window doesn't overlap any event's
        # [start - X_WINDOW_MS, start) exclusion zone (mirrors
        # host/pipeline/segmentation.py's negative-sampling philosophy).
        rng = np.random.default_rng(hash(pid) % (2**32))
        candidates = []
        t = t_min + X_WINDOW_MS
        while t <= t_max:
            if not any(abs(t - es) < X_WINDOW_MS for es in event_starts):
                candidates.append(t)
            t += 5000  # 5s stride for candidate search
        chosen = rng.choice(candidates, size=min(n_pos, len(candidates)), replace=False) if candidates else []

        for end_ms in sorted(int(c) for c in chosen):
            feats = _window_features(acc, gyr, end_ms)
            if feats is None:
                continue
            rows.append([feats[k] for k in feature_names])
            labels.append(0)
            groups.append(pid)

        print(f"  {pid}: {n_pos} positive events, {len(chosen)} negative candidates sampled")

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    return X, y, groups_arr, feature_names


if __name__ == "__main__":
    ok = verify_parity()
    if not ok:
        raise SystemExit("Parity check failed - fix before trusting any downstream feature extraction.")
    X, y, groups, names = build_public_dataset()
    print(f"\nTotal: X={X.shape}  positive_rate={y.mean():.2%}  participants={sorted(set(groups))}")
    print(f"Unreliable features (10Hz Nyquist limit, see module docstring): {UNRELIABLE_FEATURES}")
