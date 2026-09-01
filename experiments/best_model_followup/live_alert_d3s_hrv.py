"""live_alert_d3s_hrv.py - real-time monitor for the D@3s+HRV LR config
(see results.md / train_deployable.py), the only statistically-confirmed
candidate this project has produced. Experiment, not the product: beeps
and logs, doesn't actuate anything.

Reuses host.live_inference.live_alert's session logging, packet parsing,
and hotkey/alert plumbing unmodified; only the feature computation differs
from that script (multi-horizon 30/60/180s + jerk, plus clean-segment HRV,
instead of a single 60s window). Lives here rather than in host/ to keep
the experiments -> host dependency direction one-way, same as every other
experiment folder.

Needs a model/scaler/features.npz from train_deployable.py first.

Usage:
    python -m experiments.best_model_followup.live_alert_d3s_hrv --port COM3 --session live_001
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import serial

from experiments.hrv_hr_early_window.clean_segment_features import clean_segment_cardiac_features
from experiments.model_sweep.feature_variants import HORIZON_LENGTHS_MS, feature_set_D
from host.acquisition.receiver import IMU_PPG_LEN, PKT_START, PKT_TYPE_IMU_PPG, _parse_imu_ppg
from host.acquisition.session_marker import _make_key_reader, _make_nonblocking_check
from host.live_inference.live_alert import (
    BAUD,
    DEFAULT_ALERT_COOLDOWN_S,
    DEFAULT_EVAL_INTERVAL_S,
    DEFAULT_OUTCOME_GRACE_S,
    DEFAULT_THRESHOLD,
    OUTCOME_KEYS,
    SessionLog,
    _load_feature_spec,
    _play_alert_tone,
    _row_from_features,
)
from host.training.build_dataset import _min_samples_for_window

HERE = Path(__file__).parent
DEFAULT_MODEL         = str(HERE / "d3s_hrv_lr.model.joblib")
DEFAULT_SCALER        = str(HERE / "d3s_hrv_lr.scaler.joblib")
DEFAULT_FEATURES_NPZ  = str(HERE / "d3s_hrv_features.npz")
DEFAULT_OUT_DIR       = "data/live_sessions"

BUFFER_MS = max(HORIZON_LENGTHS_MS)  # 180s - the largest horizon drives window validity


def _compute_feature_row(buffer: deque, end_ms: int, feature_names: list[str], medians: np.ndarray) -> np.ndarray:
    full_df = pd.DataFrame(list(buffer)).set_index("timestamp_ms").sort_index()
    multi = {L: full_df.loc[end_ms - L: end_ms - 1] for L in HORIZON_LENGTHS_MS}
    feats = feature_set_D(multi)
    hrv = clean_segment_cardiac_features(multi[BUFFER_MS], end_ms)["features"]
    feats["rmssd_clean"] = hrv["rmssd_clean"]
    feats["sdnn_clean"] = hrv["sdnn_clean"]
    return _row_from_features(feats, feature_names, medians)


def run(
    port: str,
    baud: int,
    model_path: str,
    scaler_path: str,
    features_npz_path: str,
    session_id: str,
    out_dir: str,
    eval_interval_s: float,
    threshold: float,
    alert_cooldown_s: float,
    outcome_grace_s: float,
) -> None:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names, medians = _load_feature_spec(features_npz_path)
    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Model expects {model.n_features_in_} features but "
            f"{features_npz_path} has {len(feature_names)} feature_names - "
            f"they're out of sync with what trained this model."
        )
    min_samples = _min_samples_for_window(BUFFER_MS)

    meta = {
        "session_id": session_id,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": "D@3000ms+hrv",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_npz_path": features_npz_path,
        "buffer_ms": BUFFER_MS,
        "eval_interval_s": eval_interval_s,
        "threshold": threshold,
        "alert_cooldown_s": alert_cooldown_s,
        "note": "Statistically-confirmed HRV effect (see results.md), but on n=191 - a live "
                "false-alarm rate is not validated. Not a validated alarm.",
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
    print(f"Buffer: {BUFFER_MS}ms (multi-horizon 30/60/180s + jerk + clean-segment HRV)  "
          f"|  eval every {eval_interval_s}s  |  threshold={threshold}  |  cooldown={alert_cooldown_s}s")
    print(f"Connecting to {port} @ {baud} ...")
    print(f"Logging to {out_path / session_id}.eval_log.csv / .outcomes.csv")
    print("Hotkeys: y = pick followed last alert | n = false alarm | m = pick happened, no recent alert | Ctrl-C to stop\n")

    with serial.Serial(port, baud, timeout=0.02) as ser, SessionLog(out_path, session_id) as log:
        try:
            while True:
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
                cutoff = pkt["timestamp_ms"] - BUFFER_MS
                while buffer and buffer[0]["timestamp_ms"] < cutoff:
                    buffer.popleft()

                now_wall = time.monotonic()
                if now_wall - last_eval_wall < eval_interval_s:
                    continue
                last_eval_wall = now_wall

                if len(buffer) < min_samples:
                    continue  # not enough coverage yet (session start, dropped packets)

                end_ms = pkt["timestamp_ms"] + 1
                X = _compute_feature_row(buffer, end_ms, feature_names, medians).reshape(1, -1)
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
    p.add_argument("--session", required=True, help="Session id, e.g. live_2026-08-31_001")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
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
        eval_interval_s=args.eval_interval_s,
        threshold=args.threshold,
        alert_cooldown_s=args.alert_cooldown_s,
        outcome_grace_s=args.outcome_grace_s,
    )
