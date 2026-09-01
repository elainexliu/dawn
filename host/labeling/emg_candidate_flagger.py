"""
emg_candidate_flagger.py - automated candidate window detection from EMG.

Reads a raw .bin session, unpacks only EMG packets, computes a smoothed
envelope via rectification + Butterworth lowpass, then thresholds to find
candidate windows.

Output: data/labeled/<session_id>.candidates.csv
    columns: start_ms, end_ms, peak_emg

Usage:
    python -m host.labeling.emg_candidate_flagger --session 2026-07-07_001
"""
import argparse
import csv
import struct
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


# Packet constants (mirror receiver.py - single source is packet.h)
PKT_START       = 0xAA
PKT_TYPE_EMG    = 0x02
EMG_LEN         = 11
FMT_EMG         = "<BBBIhBB"
FIELDS_EMG      = ("start", "type", "version", "timestamp_ms", "emg_raw", "checksum", "end")

EMG_SAMPLE_RATE = 500  # Hz


def _load_emg_packets(bin_path: Path) -> list[dict]:
    """Scan a raw .bin file and return all valid EMG packets."""
    packets = []
    data = bin_path.read_bytes()
    i = 0
    while i < len(data) - EMG_LEN + 1:
        if data[i] != PKT_START:
            i += 1
            continue
        if data[i + 1] != PKT_TYPE_EMG:
            i += 1
            continue
        chunk = data[i:i + EMG_LEN]
        # Checksum: XOR bytes[1..EMG_LEN-3]
        cs = 0
        for b in chunk[1:EMG_LEN - 2]:
            cs ^= b
        if chunk[EMG_LEN - 2] != cs or chunk[EMG_LEN - 1] != 0xBB:
            i += 1
            continue
        vals = struct.unpack(FMT_EMG, chunk)
        packets.append(dict(zip(FIELDS_EMG, vals)))
        i += EMG_LEN
    return packets


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, cutoff_hz / nyq, btype="low")


def find_candidates(
    packets: list[dict],
    lowpass_hz: float = 10.0,
    threshold_multiplier: float = 3.0,
    min_gap_ms: int = 200,
    min_duration_ms: int = 50,
) -> list[dict]:
    """Detect candidate windows from EMG packets.

    Args:
        packets: Parsed EMG packet dicts with 'timestamp_ms' and 'emg_raw'.
        lowpass_hz: Lowpass cutoff for envelope smoothing.
        threshold_multiplier: Envelope threshold = multiplier * median envelope.
        min_gap_ms: Merge windows closer than this.
        min_duration_ms: Discard windows shorter than this.

    Returns:
        List of dicts with keys: start_ms, end_ms, peak_emg.
    """
    if not packets:
        return []

    timestamps = np.array([p["timestamp_ms"] for p in packets], dtype=np.float64)
    raw = np.array([p["emg_raw"] for p in packets], dtype=np.float64)

    # Rectify then smooth
    envelope = np.abs(raw)
    b, a = _butter_lowpass(lowpass_hz, EMG_SAMPLE_RATE)
    envelope = filtfilt(b, a, envelope)

    threshold = threshold_multiplier * np.median(envelope)
    above = envelope > threshold

    # Find rising/falling edges
    candidates = []
    in_window = False
    start_idx = 0

    for idx in range(len(above)):
        if above[idx] and not in_window:
            in_window = True
            start_idx = idx
        elif not above[idx] and in_window:
            in_window = False
            candidates.append((start_idx, idx - 1))

    if in_window:
        candidates.append((start_idx, len(above) - 1))

    # Convert to ms and merge close windows
    merged = []
    for s, e in candidates:
        start_ms = int(timestamps[s])
        end_ms   = int(timestamps[e])
        peak_emg = int(np.max(np.abs(raw[s:e + 1])))
        if not merged or (start_ms - merged[-1]["end_ms"]) > min_gap_ms:
            merged.append({"start_ms": start_ms, "end_ms": end_ms, "peak_emg": peak_emg})
        else:
            merged[-1]["end_ms"]   = max(merged[-1]["end_ms"], end_ms)
            merged[-1]["peak_emg"] = max(merged[-1]["peak_emg"], peak_emg)

    # Drop windows shorter than min_duration_ms
    merged = [w for w in merged if (w["end_ms"] - w["start_ms"]) >= min_duration_ms]
    return merged


def run(session_id: str, data_dir: str = "data") -> None:
    bin_path = Path(data_dir) / "raw" / f"{session_id}.bin"
    out_path = Path(data_dir) / "labeled" / f"{session_id}.candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading EMG packets from {bin_path} ...")
    packets = _load_emg_packets(bin_path)
    print(f"  {len(packets)} EMG packets loaded.")

    candidates = find_candidates(packets)
    print(f"  {len(candidates)} candidate windows found.")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["start_ms", "end_ms", "peak_emg"])
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Candidates written to {out_path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.session, args.data_dir)
