#pragma once
#include <stdint.h>
#include <stddef.h>

// Framing constants
static const uint8_t PKT_START   = 0xAA;
static const uint8_t PKT_END     = 0xBB;
static const uint8_t PKT_VERSION = 0x02; // bumped: ImuPpgPacket gained thermal fields

// Packet type bytes
static const uint8_t PKT_TYPE_IMU_PPG = 0x01;
static const uint8_t PKT_TYPE_EMG     = 0x02;
static const uint8_t PKT_TYPE_EDA     = 0x03;

// IMU+PPG+Thermal packet - 37 bytes total, 50 Hz
// IMU, PPG, and thermal all share one I2C bus (Wire) on the XIAO ESP32S3,
// so they're read and packed together each tick - see main.cpp.
// Python struct: '<BBBIhhhhhhIIffBB'
#pragma pack(1)
struct ImuPpgPacket {
    uint8_t  start;        // 0xAA
    uint8_t  type;         // 0x01
    uint8_t  version;      // 0x02
    uint32_t timestamp_ms; // ESP32 millis()
    int16_t  accel_x;      // /16384 -> g  (+/-2g)
    int16_t  accel_y;
    int16_t  accel_z;
    int16_t  gyro_x;       // /131 -> deg/s  (+/-250deg/s)
    int16_t  gyro_y;
    int16_t  gyro_z;
    uint32_t ppg_ir;       // MAX30102 FIFO IR
    uint32_t ppg_red;      // MAX30102 FIFO RED
    float    ambient_c;    // MLX90614 Ta  - sensor die/ambient temp
    float    object_c;     // MLX90614 Tobj1 - skin temp
    uint8_t  checksum;     // XOR of bytes[1..34] inclusive
    uint8_t  end;          // 0xBB
};
#pragma pack()

static_assert(sizeof(ImuPpgPacket) == 37, "ImuPpgPacket must be 37 bytes");

// EMG packet - 11 bytes total, 500 Hz
// Python struct: '<BBBIhBB'
#pragma pack(1)
struct EmgPacket {
    uint8_t  start;        // 0xAA
    uint8_t  type;         // 0x02
    uint8_t  version;      // 0x01
    uint32_t timestamp_ms; // ESP32 millis()
    int16_t  emg_raw;      // ADS1115 ch0, /8000 -> V (GAIN_ONE = +/-4.096V, 0.125mV/LSB)
    uint8_t  checksum;     // XOR of bytes[1..8] inclusive
    uint8_t  end;          // 0xBB
};
#pragma pack()

static_assert(sizeof(EmgPacket) == 11, "EmgPacket must be 11 bytes");

// EDA packet - 11 bytes total, ~20 Hz
// Python struct: '<BBBIhBB'
#pragma pack(1)
struct EdaPacket {
    uint8_t  start;        // 0xAA
    uint8_t  type;         // 0x03
    uint8_t  version;      // 0x02
    uint32_t timestamp_ms; // ESP32 millis()
    int16_t  eda_raw;      // ADS1115 ch0 (@0x49), GAIN_ONE (+/-4.096V)
    uint8_t  checksum;     // XOR of bytes[1..8] inclusive
    uint8_t  end;          // 0xBB
};
#pragma pack()

static_assert(sizeof(EdaPacket) == 11, "EdaPacket must be 11 bytes");

// Shared helper - XOR bytes[start_idx .. end_excl-1] of an arbitrary buffer
uint8_t packet_compute_checksum(const uint8_t* buf, size_t start_idx, size_t end_excl);

// Fill a packet struct from sensor readings and compute its checksum
struct IMUReading;
struct PPGReading;
struct ThermalReading;
struct EMGReading;
struct EDAReading;

void packet_fill_imu_ppg(ImuPpgPacket* pkt,
                         uint32_t timestamp_ms,
                         const IMUReading& imu,
                         const PPGReading& ppg,
                         const ThermalReading& thermal);

void packet_fill_emg(EmgPacket* pkt,
                     uint32_t timestamp_ms,
                     const EMGReading& emg);

void packet_fill_eda(EdaPacket* pkt,
                     uint32_t timestamp_ms,
                     const EDAReading& eda);