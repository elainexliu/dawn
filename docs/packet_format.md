# Packet Format

All packets share a common 7-byte header and are framed with `0xAA` (start) and `0xBB` (end).
Both packet types interleave on the same serial line; the receiver demuxes on the TYPE byte.

## Framing / sync

| Byte | Value  | Meaning                         |
|------|--------|---------------------------------|
| 0    | `0xAA` | Start-of-packet marker          |
| last | `0xBB` | End-of-packet marker            |

Receiver algorithm: scan for `0xAA`, read TYPE byte, dispatch to fixed-length reader, validate checksum + end byte.

## Common header (bytes 0–6)

| Offset | Size | Type | Field          | Notes                          |
|--------|------|------|----------------|--------------------------------|
| 0      | 1    | u8   | `start`        | Always `0xAA`                  |
| 1      | 1    | u8   | `type`         | `0x01` = IMU+PPG+Thermal, `0x02` = EMG |
| 2      | 1    | u8   | `version`      | Currently `0x02`               |
| 3      | 4    | u32  | `timestamp_ms` | ESP32 `millis()`, little-endian |

---

## Type `0x01` — IMU + PPG + Thermal (50 Hz)

IMU (MPU6050), PPG (MAX30102), and thermal (MLX90614) share one I2C bus on
the XIAO ESP32S3 (SDA = GPIO5/A4, SCL = GPIO6/A5), so all three are read and
packed into one packet per tick — no separate clock to reconcile later.

**Total: 37 bytes**
Python struct string: `<BBBIhhhhhhIIffBB`

| Offset | Size | Type | Field      | Scale / Notes                   |
|--------|------|------|------------|---------------------------------|
| 0      | 1    | u8   | `start`    | `0xAA`                          |
| 1      | 1    | u8   | `type`     | `0x01`                          |
| 2      | 1    | u8   | `version`  | `0x02`                          |
| 3      | 4    | u32  | `timestamp_ms` | little-endian               |
| 7      | 2    | i16  | `accel_x`  | ÷ 16384 → g  (±2 g range)      |
| 9      | 2    | i16  | `accel_y`  |                                 |
| 11     | 2    | i16  | `accel_z`  |                                 |
| 13     | 2    | i16  | `gyro_x`   | ÷ 131 → °/s  (±250 °/s range)  |
| 15     | 2    | i16  | `gyro_y`   |                                 |
| 17     | 2    | i16  | `gyro_z`   |                                 |
| 19     | 4    | u32  | `ppg_ir`   | MAX30102 FIFO IR value          |
| 23     | 4    | u32  | `ppg_red`  | MAX30102 FIFO RED value         |
| 27     | 4    | f32  | `ambient_c`| MLX90614 Ta — sensor die/ambient temp, °C |
| 31     | 4    | f32  | `object_c` | MLX90614 Tobj1 — skin temp, °C  |
| 35     | 1    | u8   | `checksum` | XOR of bytes[1..34] inclusive   |
| 36     | 1    | u8   | `end`      | `0xBB`                          |

Checksum range: bytes 1 through 34 (type → object_c), i.e. everything between the start byte and checksum.

---

## Type `0x02` — EMG (500 Hz)

**Not currently streamed.** The MyoWare 2.0 + ADS1115 hardware hasn't
arrived; `emg.cpp`/`emg.h` and this packet type exist but are commented out
of the active firmware build (see `main.cpp`). Format retained here for when
it's wired back in.

**Total: 11 bytes**
Python struct string: `<BBBIhBB`

| Offset | Size | Type | Field      | Scale / Notes                                  |
|--------|------|------|------------|------------------------------------------------|
| 0      | 1    | u8   | `start`    | `0xAA`                                         |
| 1      | 1    | u8   | `type`     | `0x02`                                         |
| 2      | 1    | u8   | `version`  | `0x01`                                         |
| 3      | 4    | u32  | `timestamp_ms` | little-endian                              |
| 7      | 2    | i16  | `emg_raw`  | ADS1115 ch0, GAIN_ONE (±4.096 V), 0.125 mV/LSB |
| 9      | 1    | u8   | `checksum` | XOR of bytes[1..8] inclusive                   |
| 10     | 1    | u8   | `end`      | `0xBB`                                         |

Checksum range: bytes 1 through 8 (type → emg_raw).

---

## Versioning

The `version` byte (`0x02` currently) allows the receiver to detect sessions
recorded with a different firmware. Old sessions with `version=0x01` (29-byte
IMU+PPG, no thermal fields) remain unambiguous even after format changes.
Bump this byte when the byte layout changes.

## Serial parameters

| Parameter | Value  |
|-----------|--------|
| Baud rate | 921600 |
| Data bits | 8      |
| Stop bits | 1      |
| Parity    | None   |

## Timing

| Packet             | Interval | Nominal rate |
|--------------------|----------|--------------|
| IMU + PPG + Thermal| 20 ms    | 50 Hz        |
| EMG (disabled)     | 2 ms     | 500 Hz       |

The timer fires from `millis()` in the Arduino loop with `if (now - last >= interval)`.
