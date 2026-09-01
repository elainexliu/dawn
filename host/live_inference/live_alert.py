"""
live_alert.py - real-time "upcoming pick" probability monitor.

Experiment, not the product: beeps and logs, doesn't actuate anything.
Reuses the training pipeline's feature extraction and packet parsing
directly rather than reimplementing them, and pins feature columns/NaN
fill values to what's saved in features.npz, since a single live window
can't reproduce build_dataset's batch-level NaN handling.

Runs the current LR baseline (models/lr.model.joblib) - the paired-bootstrap
sweep in experiments/model_sweep/ didn't turn up anything that reliably
beats it. AUC on held-out days is ~0.596, barely above chance, so the
threshold below is a starting point to tune against your own false-alarm
tolerance, not a validated operating point. Expect false alarms - that's
what the hotkey outcome log is for.

Needs exclusive access to the serial port, same as receiver.py, so it can't
run at the same time as a recording session on the same device. Doesn't
persist raw sensor bytes - this isn't also a recording session.

Usage:
    python -m host.live_inference.live_alert --port COM3 --session live_001
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from host.pipeline.features import compute_features
from host.training.build_dataset import DEFAULT_X_WINDOW_MS, _min_samples_for_window

# Re-use packet parsing from receiver - do NOT re-implement the wire format
from host.acquisition.receiver import (
    PKT_START,
    PKT_TYPE_IMU_PPG,
    IMU_PPG_LEN,
    _parse_imu_ppg,
)

# Re-use the non-blocking keypress primitives session_marker.py already built
from host.acquisition.session_marker import _make_nonblocking_check, _make_key_reader

import serial

try:
    import winsound
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False


DEFAULT_MODEL   = "models/lr.model.joblib"
DEFAULT_SCALER  = "models/lr.scaler.joblib"
DEFAULT_FEATURES_NPZ = "data/processed/features.npz"
DEFAULT_OUT_DIR = "data/live_sessions"

# A 60s window stepped every 5s overlaps ~92% with its predecessor - fast
# enough to matter given the window itself is only 60s, slow enough that
# evaluations aren't pure noise restated every tick.
DEFAULT_EVAL_INTERVAL_S = 5.0

DEFAULT_THRESHOLD = 0.5

# Without a cooldown, a sustained above-threshold stretch beeps every tick.
DEFAULT_ALERT_COOLDOWN_S = 30.0

# How long after an alert a hotkey mark still counts as "about this alert".
DEFAULT_OUTCOME_GRACE_S = 300.0

BAUD = 921600

# Hotkey -> outcome label, mirroring session_marker.KEY_EVENTS' pattern.
OUTCOME_KEYS = {
    "y": "confirmed_pick",   # a pick/urge did follow the most recent alert
    "n": "false_alarm",      # alert fired, nothing followed
    "m": "missed_pick",      # a pick just happened with no recent alert (false negative)
}


def _load_feature_spec(features_npz_path: str) -> tuple[list[str], np.ndarray]:
    """Return (feature_names, medians) exactly as build_dataset produced them -
    the column order the saved model expects, and the per-column fallback for
    any feature compute_features can't produce on a given window (e.g.
    ppg_hr_* when beat detection fails on a motion-corrupted stretch)."""
    npz = np.load(features_npz_path, allow_pickle=True)
    feature_names = [str(n) for n in npz["feature_names"]]
    medians = np.nanmedian(npz["X"], axis=0)
    return feature_names, medians


def _row_from_features(feats: dict[str, float], feature_names: list[str], medians: np.ndarray) -> np.ndarray:
    """Reindex a compute_features() dict onto the trained column order,
    imputing any NaN (or missing key) with the frozen training median."""
    row = np.empty(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        val = feats.get(name, np.nan)
        row[i] = medians[i] if (val is None or np.isnan(val)) else val
    return row


# Threaded so a few hundred ms of Beep() never stalls the serial read loop.
def _play_alert_tone() -> None:
    if not _HAVE_WINSOUND:
        print("\a", end="", flush=True)  # terminal bell fallback
        return

    def _beep():
        winsound.Beep(1200, 250)

    threading.Thread(target=_beep, daemon=True).start()


class SessionLog:
    """Two append-only CSVs: every evaluation, and every hotkey-marked outcome."""

    EVAL_FIELDS = [
        "eval_idx", "utc_iso", "timestamp_ms", "n_samples_in_window",
        "probability", "alert_fired", "suppressed_by_cooldown",
    ]
    OUTCOME_FIELDS = [
        "utc_iso", "timestamp_ms", "event",
        "matched_alert_idx", "seconds_since_alert",
    ]

    def __init__(self, out_dir: Path, session_id: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_path = out_dir / f"{session_id}.eval_log.csv"
        self.outcome_path = out_dir / f"{session_id}.outcomes.csv"

        self._eval_fh = open(self.eval_path, "w", newline="")
        self._eval_writer = csv.DictWriter(self._eval_fh, fieldnames=self.EVAL_FIELDS)
        self._eval_writer.writeheader()

        self._outcome_fh = open(self.outcome_path, "w", newline="")
        self._outcome_writer = csv.DictWriter(self._outcome_fh, fieldnames=self.OUTCOME_FIELDS)
        self._outcome_writer.writeheader()

    def log_eval(self, **row) -> None:
        self._eval_writer.writerow(row)
        self._eval_fh.flush()

    def log_outcome(self, **row) -> None:
        self._outcome_writer.writerow(row)
        self._outcome_fh.flush()

    def close(self) -> None:
        self._eval_fh.close()
        self._outcome_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run(
    port: str,
    baud: int,
    model_path: str,
    scaler_path: str,
    features_npz_path: str,
    session_id: str,
    out_dir: str,
    window_ms: int,
    eval_interval_s: float,
    threshold: float,
    alert_cooldown_s: float,
    outcome_grace_s: float,
) -> None:
    if window_ms != DEFAULT_X_WINDOW_MS:
        print(
            f"WARNING: --window-ms={window_ms} differs from the training window "
            f"({DEFAULT_X_WINDOW_MS}ms). FFT-derived features (dominant_freq, "
            f"power_*hz) depend on window duration - this reintroduces train/serve "
            f"skew. Proceeding anyway since you asked for it explicitly."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names, medians = _load_feature_spec(features_npz_path)
    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Model expects {model.n_features_in_} features but "
            f"{features_npz_path} has {len(feature_names)} feature_names - "
            f"they're out of sync with what trained this model."
        )
    min_samples = _min_samples_for_window(window_ms)

    meta = {
        "session_id": session_id,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_npz_path": features_npz_path,
        "window_ms": window_ms,
        "eval_interval_s": eval_interval_s,
        "threshold": threshold,
        "alert_cooldown_s": alert_cooldown_s,
        "note": "AUC ~0.596 on held-out days - weak-but-better-than-chance signal, not a validated alarm.",
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"{session_id}.meta.json").write_text(json.dumps(meta, indent=2))

    check_key = _make_nonblocking_check()
    read_key = _make_key_reader()

    buffer: deque[dict] = deque()
    eval_idx = 0
    last_eval_wall = 0.0
    last_alert_idx: int | None = None
    last_alert_wall: float | None = None

    print(f"Loaded model:  {model_path}")
    print(f"Loaded scaler: {scaler_path}")
    print(f"Feature spec:  {len(feature_names)} columns from {features_npz_path}")
    print(f"Window: {window_ms}ms  |  eval every {eval_interval_s}s  |  threshold={threshold}  |  cooldown={alert_cooldown_s}s")
    print(f"Connecting to {port} @ {baud} ...")
    print(f"Logging to {out_path / session_id}.eval_log.csv / .outcomes.csv")
    print("Hotkeys: y = pick followed last alert | n = false alarm | m = pick happened, no recent alert | Ctrl-C to stop\n")

    with serial.Serial(port, baud, timeout=0.02) as ser, SessionLog(out_path, session_id) as log:
        try:
            while True:
                # --- non-blocking hotkey check for outcome marking ---
                if check_key():
                    key = read_key()
                    event = OUTCOME_KEYS.get(key)
                    if event is not None:
                        now_wall = time.monotonic()
                        seconds_since_alert = (
                            now_wall - last_alert_wall if last_alert_wall is not None else None
                        )
                        matched = (
                            last_alert_idx
                            if seconds_since_alert is not None and seconds_since_alert <= outcome_grace_s
                            else None
                        )
                        log.log_outcome(
                            utc_iso=datetime.now(timezone.utc).isoformat(),
                            timestamp_ms=buffer[-1]["timestamp_ms"] if buffer else "",
                            event=event,
                            matched_alert_idx=matched if matched is not None else "",
                            seconds_since_alert=f"{seconds_since_alert:.1f}" if seconds_since_alert is not None else "",
                        )
                        print(f"  [marked] {event}"
                              + (f" (alert #{matched}, {seconds_since_alert:.0f}s ago)" if matched is not None else " (no recent alert)"))

                # --- read one packet ---
                byte = ser.read(1)
                if not byte or byte[0] != PKT_START:
                    continue
                type_byte = ser.read(1)
                if not type_byte or type_byte[0] != PKT_TYPE_IMU_PPG:
                    continue  # EMG/EDA packets skipped - not used by this feature set
                rest = ser.read(IMU_PPG_LEN - 2)
                raw = byte + type_byte + rest
                pkt = _parse_imu_ppg(raw)
                if pkt is None:
                    continue

                buffer.append(pkt)
                cutoff = pkt["timestamp_ms"] - window_ms
                while buffer and buffer[0]["timestamp_ms"] < cutoff:
                    buffer.popleft()

                # --- re-evaluate on a wall-clock cadence, independent of packet jitter ---
                now_wall = time.monotonic()
                if now_wall - last_eval_wall < eval_interval_s:
                    continue
                last_eval_wall = now_wall

                if len(buffer) < min_samples:
                    continue  # not enough coverage yet (session start, dropped packets)

                window_df = pd.DataFrame(list(buffer)).set_index("timestamp_ms").sort_index()
                feats = compute_features(window_df)
                X = _row_from_features(feats, feature_names, medians).reshape(1, -1)
                X_scaled = scaler.transform(X)
                prob = float(model.predict_proba(X_scaled)[0, 1])

                alert_fired = False
                suppressed = False
                if prob >= threshold:
                    if last_alert_wall is not None and (now_wall - last_alert_wall) < alert_cooldown_s:
                        suppressed = True
                    else:
                        alert_fired = True
                        last_alert_wall = now_wall
                        last_alert_idx = eval_idx
                        _play_alert_tone()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERT  p={prob:.3f}  (eval #{eval_idx})")

                log.log_eval(
                    eval_idx=eval_idx,
                    utc_iso=datetime.now(timezone.utc).isoformat(),
                    timestamp_ms=pkt["timestamp_ms"],
                    n_samples_in_window=len(buffer),
                    probability=f"{prob:.6f}",
                    alert_fired=alert_fired,
                    suppressed_by_cooldown=suppressed,
                )
                eval_idx += 1

        except KeyboardInterrupt:
            print(f"\nStopped. {eval_idx} evaluations logged.")

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="Serial port (e.g. COM3)")
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--features-npz", default=DEFAULT_FEATURES_NPZ,
                   help="Source of the trained feature_names + medians (must match --model)")
    p.add_argument("--session", required=True, help="Session id, e.g. live_2026-07-24_001")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--window-ms", type=int, default=DEFAULT_X_WINDOW_MS,
                   help="Must match the training window length")
    p.add_argument("--eval-interval-s", type=float, default=DEFAULT_EVAL_INTERVAL_S)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Alert probability threshold")
    p.add_argument("--alert-cooldown-s", type=float, default=DEFAULT_ALERT_COOLDOWN_S)
    p.add_argument("--outcome-grace-s", type=float, default=DEFAULT_OUTCOME_GRACE_S)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        port=args.port,
        baud=args.baud,
        model_path=args.model,
        scaler_path=args.scaler,
        features_npz_path=args.features_npz,
        session_id=args.session,
        out_dir=args.out_dir,
        window_ms=args.window_ms,
        eval_interval_s=args.eval_interval_s,
        threshold=args.threshold,
        alert_cooldown_s=args.alert_cooldown_s,
        outcome_grace_s=args.outcome_grace_s,
    )
