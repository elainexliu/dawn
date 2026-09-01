"""
receiver.py - read serial packets from the ESP32 and write a raw session.

Invoke as a module (not directly as a script):
    python -m host.acquisition.receiver --port COM3 --session 001

Outputs (under data/raw/):
    <session_id>.bin          Raw packet bytes, exactly as received
    <session_id>.meta.json    Session metadata
    <session_id>.markers.csv  ESP32-timestamped keypress markers
"""
import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import serial

from host.acquisition.session_marker import MarkerLogger

# Packet constants - must match packet.h
PKT_START   = 0xAA
PKT_END     = 0xBB
PKT_VERSION = 0x02

PKT_TYPE_IMU_PPG = 0x01  # IMU + PPG + thermal (shared I2C bus, one packet/tick)
PKT_TYPE_EMG     = 0x02  # Bus 0, dedicated ADS1115
PKT_TYPE_EDA     = 0x03  # Bus 1, shared ADS1115 @0x49

# Total packet lengths including start/end bytes
IMU_PPG_LEN = 37
EMG_LEN     = 11
EDA_LEN     = 11

# struct formats (little-endian)
# '<BBBIhhhhhhIIffBB'  - IMU+PPG+Thermal (37 bytes)
FMT_IMU_PPG = "<BBBIhhhhhhIIffBB"
# '<BBBIhBB'         - EMG (11 bytes)
FMT_EMG     = "<BBBIhBB"
# '<BBBIhBB'         - EDA (11 bytes, same shape as EMG)
FMT_EDA     = "<BBBIhBB"

assert struct.calcsize(FMT_IMU_PPG) == IMU_PPG_LEN
assert struct.calcsize(FMT_EMG)     == EMG_LEN
assert struct.calcsize(FMT_EDA)     == EDA_LEN

FIELDS_IMU_PPG = (
    "start", "type", "version", "timestamp_ms",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "ppg_ir", "ppg_red",
    "ambient_c", "object_c",
    "checksum", "end",
)
FIELDS_EMG = ("start", "type", "version", "timestamp_ms", "emg_raw", "checksum", "end")
FIELDS_EDA = ("start", "type", "version", "timestamp_ms", "eda_raw", "checksum", "end")


def _compute_checksum(buf: bytes, start: int, end_excl: int) -> int:
    cs = 0
    for b in buf[start:end_excl]:
        cs ^= b
    return cs


def _parse_imu_ppg(buf: bytes) -> dict | None:
    """Parse a 37-byte IMU+PPG+Thermal packet buffer. Returns None on checksum/framing error."""
    if len(buf) != IMU_PPG_LEN:
        return None
    expected_cs = _compute_checksum(buf, 1, IMU_PPG_LEN - 2)
    if buf[-2] != expected_cs or buf[-1] != PKT_END:
        return None
    vals = struct.unpack(FMT_IMU_PPG, buf)
    return dict(zip(FIELDS_IMU_PPG, vals))


def _parse_emg(buf: bytes) -> dict | None:
    """Parse an 11-byte EMG packet buffer. Returns None on checksum/framing error."""
    if len(buf) != EMG_LEN:
        return None
    expected_cs = _compute_checksum(buf, 1, EMG_LEN - 2)
    if buf[-2] != expected_cs or buf[-1] != PKT_END:
        return None
    vals = struct.unpack(FMT_EMG, buf)
    return dict(zip(FIELDS_EMG, vals))


def _parse_eda(buf: bytes) -> dict | None:
    """Parse an 11-byte EDA packet buffer. Returns None on checksum/framing error."""
    if len(buf) != EDA_LEN:
        return None
    expected_cs = _compute_checksum(buf, 1, EDA_LEN - 2)
    if buf[-2] != expected_cs or buf[-1] != PKT_END:
        return None
    vals = struct.unpack(FMT_EDA, buf)
    return dict(zip(FIELDS_EDA, vals))


def run(port: str, baud: int, session_id: str, data_dir: str = "data/raw") -> None:
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_path     = out_dir / f"{session_id}.bin"
    meta_path    = out_dir / f"{session_id}.meta.json"
    markers_path = out_dir / f"{session_id}.markers.csv"

    meta = {
        "session_id":  session_id,
        "port":        port,
        "baud":        baud,
        "start_utc":   datetime.now(timezone.utc).isoformat(),
        "pkt_version": PKT_VERSION,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Recording session '{session_id}' on {port} @ {baud} baud")
    print(f"  -> {bin_path}")
    print(f"  -> {markers_path}")
    print("Markers: [ = hand skin picking, ] = lip picking, \\ = face skin picking. Ctrl-C to stop.\n")

    last_ts_ms = 0
    imu_ppg_count = 0
    emg_count = 0
    eda_count = 0

    with serial.Serial(port, baud, timeout=0.02) as ser, \
         open(bin_path, "wb") as bin_fh, \
         MarkerLogger(str(markers_path)) as marker:

        try:
            while True:
                # Sync: scan for start byte
                byte = ser.read(1)
                if not byte:
                    continue
                if byte[0] != PKT_START:
                    continue

                # Read type byte to know how many more bytes to expect
                type_byte = ser.read(1)
                if not type_byte:
                    continue

                pkt_type = type_byte[0]
                if pkt_type == PKT_TYPE_IMU_PPG:
                    rest = ser.read(IMU_PPG_LEN - 2)
                    raw = byte + type_byte + rest
                    pkt = _parse_imu_ppg(raw)
                    if pkt:
                        bin_fh.write(raw)
                        last_ts_ms = pkt["timestamp_ms"]
                        imu_ppg_count += 1

                elif pkt_type == PKT_TYPE_EMG:
                    rest = ser.read(EMG_LEN - 2)
                    raw = byte + type_byte + rest
                    pkt = _parse_emg(raw)
                    if pkt:
                        bin_fh.write(raw)
                        last_ts_ms = pkt["timestamp_ms"]
                        emg_count += 1

                elif pkt_type == PKT_TYPE_EDA:
                    rest = ser.read(EDA_LEN - 2)
                    raw = byte + type_byte + rest
                    pkt = _parse_eda(raw)
                    if pkt:
                        bin_fh.write(raw)
                        last_ts_ms = pkt["timestamp_ms"]
                        eda_count += 1
                else:
                    # Unknown type - skip
                    continue

                marker.check_and_log(last_ts_ms)

        except KeyboardInterrupt:
            pass

    # Update meta with end time and counts
    meta["end_utc"]        = datetime.now(timezone.utc).isoformat()
    meta["imu_ppg_packets"] = imu_ppg_count
    meta["emg_packets"]    = emg_count
    meta["eda_packets"]    = eda_count
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone. IMU+PPG: {imu_ppg_count}  EMG: {emg_count}  EDA: {eda_count}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Dawn serial receiver")
    parser.add_argument("--port",    required=True,  help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",    type=int, default=921600)
    parser.add_argument("--session", required=True,  help="Session ID string (e.g. 2026-07-07_001)")
    parser.add_argument("--data-dir", default="data/raw")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.port, args.baud, args.session, args.data_dir)