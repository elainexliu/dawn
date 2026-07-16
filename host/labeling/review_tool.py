"""
review_tool.py — interactive window review and labeling.

For each candidate window in <session>.candidates.csv, plots IMU / PPG / EMG
data and waits for a keypress decision.

Keys:
    y / Enter  → accept (label = 1, BFRB)
    n          → reject (label = 0, non-BFRB)
    a          → adjust (prompts new start/end, then re-plots)
    q          → quit and save progress

Output: data/labeled/<session_id>.labels.csv
    columns: start_ms, end_ms, label

Usage:
    python -m host.labeling.review_tool --session 2026-07-07_001
"""
import argparse
import csv
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse packet constants from emg_candidate_flagger
from host.labeling.emg_candidate_flagger import (
    _load_emg_packets,
    PKT_START, PKT_TYPE_IMU_PPG,
)

IMU_PPG_LEN   = 37
FMT_IMU_PPG   = "<BBBIhhhhhhIIffBB"
FIELDS_IMU_PPG = (
    "start", "type", "version", "timestamp_ms",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "ppg_ir", "ppg_red",
    "ambient_c", "object_c",
    "checksum", "end",
)


def _load_imu_ppg_packets(bin_path: Path) -> list[dict]:
    packets = []
    data = bin_path.read_bytes()
    i = 0
    while i < len(data) - IMU_PPG_LEN + 1:
        if data[i] != PKT_START:
            i += 1
            continue
        if data[i + 1] != PKT_TYPE_IMU_PPG:
            i += 1
            continue
        chunk = data[i:i + IMU_PPG_LEN]
        cs = 0
        for b in chunk[1:IMU_PPG_LEN - 2]:
            cs ^= b
        if chunk[IMU_PPG_LEN - 2] != cs or chunk[IMU_PPG_LEN - 1] != 0xBB:
            i += 1
            continue
        vals = struct.unpack(FMT_IMU_PPG, chunk)
        packets.append(dict(zip(FIELDS_IMU_PPG, vals)))
        i += IMU_PPG_LEN
    return packets


def _to_df(packets: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(packets).set_index("timestamp_ms").sort_index()


def _plot_window(imu_df: pd.DataFrame, emg_df: pd.DataFrame,
                 start_ms: int, end_ms: int, idx: int, total: int):
    margin = 500  # ms context around window
    t0, t1 = start_ms - margin, end_ms + margin

    imu_win = imu_df.loc[t0:t1]
    emg_win = emg_df.loc[t0:t1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Window {idx + 1}/{total}  |  [{start_ms} – {end_ms} ms]", fontsize=11)

    ax_accel, ax_gyro, ax_ppg, ax_emg = axes

    if not imu_win.empty:
        t_imu = imu_win.index.values
        ax_accel.plot(t_imu, imu_win["accel_x"] / 16384, label="ax")
        ax_accel.plot(t_imu, imu_win["accel_y"] / 16384, label="ay")
        ax_accel.plot(t_imu, imu_win["accel_z"] / 16384, label="az")
        ax_gyro.plot(t_imu, imu_win["gyro_x"] / 131, label="gx")
        ax_gyro.plot(t_imu, imu_win["gyro_y"] / 131, label="gy")
        ax_gyro.plot(t_imu, imu_win["gyro_z"] / 131, label="gz")
        ax_ppg.plot(t_imu, imu_win["ppg_ir"],  label="IR")
        ax_ppg.plot(t_imu, imu_win["ppg_red"], label="RED")

    if not emg_win.empty:
        ax_emg.plot(emg_win.index.values, emg_win["emg_raw"] * 0.125, label="EMG (mV)")

    ax_accel.set_ylabel("Accel (g)")
    ax_gyro.set_ylabel("Gyro (°/s)")
    ax_ppg.set_ylabel("PPG (raw)")
    ax_emg.set_ylabel("EMG (mV)")
    ax_emg.set_xlabel("timestamp_ms")

    for ax in axes:
        ax.axvspan(start_ms, end_ms, alpha=0.15, color="red")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show(block=False)
    return fig


def run(session_id: str, data_dir: str = "data") -> None:
    raw_dir     = Path(data_dir) / "raw"
    labeled_dir = Path(data_dir) / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = labeled_dir / f"{session_id}.candidates.csv"
    labels_path     = labeled_dir / f"{session_id}.labels.csv"
    bin_path        = raw_dir / f"{session_id}.bin"

    if not candidates_path.exists():
        print(f"No candidates file found at {candidates_path}")
        print("Run emg_candidate_flagger first.")
        return

    candidates = pd.read_csv(candidates_path)

    print(f"Loading packets from {bin_path} ...")
    imu_packets = _load_imu_ppg_packets(bin_path)
    emg_packets = _load_emg_packets(bin_path)
    imu_df = _to_df(imu_packets)
    emg_df = pd.DataFrame(emg_packets).set_index("timestamp_ms").sort_index()
    print(f"  {len(imu_packets)} IMU+PPG,  {len(emg_packets)} EMG packets")

    labels = []

    for idx, row in candidates.iterrows():
        start_ms = int(row["start_ms"])
        end_ms   = int(row["end_ms"])
        fig = _plot_window(imu_df, emg_df, start_ms, end_ms, idx, len(candidates))

        while True:
            key = input(f"  [y=accept / n=reject / a=adjust / q=quit]: ").strip().lower()
            if key in ("y", ""):
                labels.append({"start_ms": start_ms, "end_ms": end_ms, "label": 1})
                break
            elif key == "n":
                labels.append({"start_ms": start_ms, "end_ms": end_ms, "label": 0})
                break
            elif key == "a":
                try:
                    new_start = int(input("    new start_ms: "))
                    new_end   = int(input("    new end_ms:   "))
                    plt.close(fig)
                    fig = _plot_window(imu_df, emg_df, new_start, new_end, idx, len(candidates))
                    start_ms, end_ms = new_start, new_end
                except ValueError:
                    print("    Invalid input — keeping original bounds.")
            elif key == "q":
                plt.close("all")
                _save_labels(labels, labels_path)
                return
            else:
                print("    Unrecognized key.")

        plt.close(fig)

    _save_labels(labels, labels_path)


def _save_labels(labels: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["start_ms", "end_ms", "label"])
        writer.writeheader()
        writer.writerows(labels)
    print(f"Labels saved to {path}  ({len(labels)} rows)")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.session, args.data_dir)
