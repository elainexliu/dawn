"""
verify_session_growth.py - is 2026-07-23-07-89pm structurally sound despite
growing from ~5K to ~245K packets between analysis runs?

Scans the raw .bin directly (both IMU+PPG and EMG packet types, per
docs/packet_format.md) instead of reusing build_dataset.py's IMU-only
parser, so any unrecognized byte region gets counted instead of skipped.

Usage:
    python -m experiments.diagnostics.verify_session_growth
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
TARGET_SESSION = "2026-07-23-07-89pm"

PKT_START = 0xAA
PKT_END = 0xBB
PKT_TYPE_IMU_PPG = 0x01
PKT_TYPE_EMG = 0x02
IMU_PPG_LEN = 37
EMG_LEN = 11
FMT_IMU_PPG = "<BBBIhhhhhhIIffBB"
FIELDS_IMU_PPG = (
    "start", "type", "version", "timestamp_ms",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "ppg_ir", "ppg_red", "ambient_c", "object_c", "checksum", "end",
)

MILLIS_WRAP = 2**32
WRAP_TOLERANCE_MS = 60_000  # new_ts must be small AND old_ts near the wrap point


def _checksum(buf: bytes, start: int, end_excl: int) -> int:
    cs = 0
    for b in buf[start:end_excl]:
        cs ^= b
    return cs


def scan_bin(bin_path: Path) -> dict:
    """Scan every 0xAA candidate; classify as valid IMU_PPG, valid EMG, or
    unrecognized/corrupt (checksum/end-byte/type failure)."""
    data = bin_path.read_bytes()
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8)

    imu_packets = []
    emg_count = 0
    corrupt_count = 0
    corrupt_positions = []
    next_allowed = 0

    for i in np.flatnonzero(arr == PKT_START):
        i = int(i)
        if i < next_allowed:
            continue
        if i + 2 > n:
            corrupt_count += 1
            corrupt_positions.append(i)
            continue
        pkt_type = data[i + 1]

        if pkt_type == PKT_TYPE_IMU_PPG and i + IMU_PPG_LEN <= n:
            chunk = data[i:i + IMU_PPG_LEN]
            if chunk[IMU_PPG_LEN - 2] == _checksum(chunk, 1, IMU_PPG_LEN - 2) and chunk[IMU_PPG_LEN - 1] == PKT_END:
                vals = struct.unpack(FMT_IMU_PPG, chunk)
                imu_packets.append(dict(zip(FIELDS_IMU_PPG, vals)))
                imu_packets[-1]["_file_offset"] = i
                next_allowed = i + IMU_PPG_LEN
                continue

        if pkt_type == PKT_TYPE_EMG and i + EMG_LEN <= n:
            chunk = data[i:i + EMG_LEN]
            if chunk[EMG_LEN - 2] == _checksum(chunk, 1, EMG_LEN - 2) and chunk[EMG_LEN - 1] == PKT_END:
                emg_count += 1
                next_allowed = i + EMG_LEN
                continue

        # Didn't validate as either type - could be a torn packet or just a
        # stray 0xAA inside another packet's payload; can't tell which here.
        corrupt_count += 1
        corrupt_positions.append(i)

    return {
        "file_size_bytes": n,
        "imu_ppg_packets": imu_packets,
        "n_imu_ppg": len(imu_packets),
        "n_emg": emg_count,
        "n_unclassified_0xAA": corrupt_count,
        "corrupt_positions_sample": corrupt_positions[:20],
    }


def check_monotonicity(imu_packets: list[dict]) -> dict:
    ts = np.array([p["timestamp_ms"] for p in imu_packets], dtype=np.int64)
    diffs = np.diff(ts)
    backward = np.flatnonzero(diffs < 0)

    real_anomalies = []
    wraparounds = []
    for idx in backward:
        old_ts, new_ts = int(ts[idx]), int(ts[idx + 1])
        is_wrap = (old_ts > MILLIS_WRAP - WRAP_TOLERANCE_MS) and (new_ts < WRAP_TOLERANCE_MS)
        (wraparounds if is_wrap else real_anomalies).append(
            {"file_index": int(idx), "prev_ts": old_ts, "next_ts": new_ts, "drop_ms": old_ts - new_ts}
        )
    return {
        "n_backward_jumps": len(backward),
        "n_plausible_wraparounds": len(wraparounds),
        "n_real_anomalies": len(real_anomalies),
        "real_anomalies_sample": real_anomalies[:20],
        "duration_covered_ms": int(ts[-1] - ts[0]) if len(ts) > 1 else 0,
    }


def check_duplicates(imu_packets: list[dict]) -> dict:
    exact_dupes = 0
    same_ts_diff_payload = 0
    payload_keys = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "ppg_ir", "ppg_red")
    for a, b in zip(imu_packets, imu_packets[1:]):
        if a["timestamp_ms"] == b["timestamp_ms"]:
            if all(a[k] == b[k] for k in payload_keys):
                exact_dupes += 1
            else:
                same_ts_diff_payload += 1
    return {"exact_consecutive_duplicates": exact_dupes, "same_timestamp_different_payload": same_ts_diff_payload}


def check_marker_density(session_id: str, imu_packets: list[dict]) -> dict:
    markers_path = RAW_DIR / f"{session_id}.markers.csv"
    markers = pd.read_csv(markers_path)
    if not imu_packets:
        return {"error": "no IMU packets to establish session timeline"}

    t_min = imu_packets[0]["timestamp_ms"]
    t_max = imu_packets[-1]["timestamp_ms"]
    duration_min = (t_max - t_min) / 60000

    n_buckets = max(1, int(np.ceil(duration_min / 5)))  # 5-minute buckets
    bucket_counts = [0] * n_buckets
    for ts in markers["timestamp_ms"]:
        bucket = min(int((ts - t_min) / 300000), n_buckets - 1)
        if 0 <= bucket < n_buckets:
            bucket_counts[bucket] += 1

    return {
        "n_markers": len(markers),
        "session_duration_min": round(duration_min, 1),
        "bucket_minutes": 5,
        "bucket_counts": bucket_counts,
        "last_marker_ts_relative_min": round((markers["timestamp_ms"].max() - t_min) / 60000, 1) if len(markers) else None,
    }


def check_all_filenames_for_invalid_time() -> list[dict]:
    """'HH-MMpm'-shaped filenames where MM > 59 are not valid clock times -
    check every session, not just the one in question."""
    results = []
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{1,2})-(\d+)(am|pm)")
    for bin_path in sorted(RAW_DIR.glob("*.bin")):
        m = pattern.match(bin_path.stem)
        if not m:
            results.append({"session_id": bin_path.stem, "pattern_match": False})
            continue
        date, hour, minute, ampm = m.groups()
        valid = minute.isdigit() and int(minute) <= 59 and hour.isdigit() and 1 <= int(hour) <= 12
        results.append({
            "session_id": bin_path.stem, "pattern_match": True,
            "parsed_hour": hour, "parsed_minute": minute, "valid_clock_time": valid,
        })
    return results


def main():
    print(f"=== Verifying {TARGET_SESSION} ===\n")

    print("--- Filename vs. actual metadata timestamp ---")
    meta = json.loads((RAW_DIR / f"{TARGET_SESSION}.meta.json").read_text())
    print(f"  session_id (user-typed, free text): {TARGET_SESSION}")
    print(f"  actual start_utc (host wall clock): {meta['start_utc']}")
    print(f"  actual end_utc:                     {meta['end_utc']}")
    print("  '89' as a minute value is invalid (minutes are 0-59) - confirms the filename's")
    print("  embedded time is a free-text label, not a validated/parsed timestamp.")

    print("\n--- Checking ALL session filenames for the same pattern ---")
    filename_check = check_all_filenames_for_invalid_time()
    for r in filename_check:
        if r.get("pattern_match") and not r["valid_clock_time"]:
            print(f"  INVALID: {r['session_id']}  (parsed minute={r['parsed_minute']})")
    n_invalid = sum(1 for r in filename_check if r.get("pattern_match") and not r["valid_clock_time"])
    print(f"  {n_invalid} of {len(filename_check)} session filenames have an invalid embedded clock time.")

    print(f"\n--- Scanning raw .bin for {TARGET_SESSION} ---")
    scan = scan_bin(RAW_DIR / f"{TARGET_SESSION}.bin")
    print(f"  file size: {scan['file_size_bytes']:,} bytes")
    print(f"  valid IMU+PPG packets: {scan['n_imu_ppg']:,}")
    print(f"  valid EMG packets: {scan['n_emg']:,}")
    print(f"  unclassified 0xAA candidates (potential corruption OR false-positive stray byte): {scan['n_unclassified_0xAA']:,}")
    corruption_rate = scan["n_unclassified_0xAA"] / max(1, scan["n_imu_ppg"] + scan["n_emg"] + scan["n_unclassified_0xAA"])
    print(f"  unclassified rate: {corruption_rate:.4%}")

    print("\n--- Timestamp monotonicity ---")
    mono = check_monotonicity(scan["imu_ppg_packets"])
    print(f"  packets span {mono['duration_covered_ms']/60000:.1f} minutes")
    print(f"  backward jumps: {mono['n_backward_jumps']}"
          f"  (plausible millis() wraparounds: {mono['n_plausible_wraparounds']},"
          f" real anomalies: {mono['n_real_anomalies']})")
    if mono["real_anomalies_sample"]:
        for a in mono["real_anomalies_sample"][:5]:
            print(f"    ANOMALY at packet #{a['file_index']}: {a['prev_ts']} -> {a['next_ts']} (dropped {a['drop_ms']}ms)")

    print("\n--- Duplicate packet check ---")
    dupes = check_duplicates(scan["imu_ppg_packets"])
    print(f"  exact consecutive duplicates (same timestamp AND same payload): {dupes['exact_consecutive_duplicates']}")
    print(f"  same timestamp, different payload (borderline, not necessarily a bug): {dupes['same_timestamp_different_payload']}")

    print("\n--- Marker density over time ---")
    density = check_marker_density(TARGET_SESSION, scan["imu_ppg_packets"])
    print(f"  {density['n_markers']} markers over {density['session_duration_min']} minutes")
    print(f"  markers per 5-minute bucket: {density['bucket_counts']}")
    print(f"  last marker at {density['last_marker_ts_relative_min']} min into the session"
          f" (session runs {density['session_duration_min']} min total)")

    result = {
        "target_session": TARGET_SESSION,
        "meta": meta,
        "filename_check": filename_check,
        "scan": {k: v for k, v in scan.items() if k != "imu_ppg_packets"},
        "monotonicity": mono,
        "duplicates": dupes,
        "marker_density": density,
    }
    Path("experiments/diagnostics/task1_raw_results.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nSaved: experiments/diagnostics/task1_raw_results.json")


if __name__ == "__main__":
    main()
