#include <Arduino.h>
#include <Wire.h>
#include "packet.h"
#include "serial_stream.h"
#include "sensors/imu.h"
#include "sensors/ppg.h"
#include "sensors/thermal.h"
// Reverted to 3-sensor setup (IMU, PPG, thermal) — EMG/EDA disabled for now.
// #include "sensors/emg.h"
// #include "sensors/eda.h"

// IMU (MPU6050), PPG (MAX30102), and thermal (MLX90614) all share the XIAO
// ESP32S3's one I2C bus: SDA = GPIO5 (A4), SCL = GPIO6 (A5).
static const int I2C_SDA_PIN = 5; // A4
static const int I2C_SCL_PIN = 6; // A5


// Bus 0 (dedicated): EMG (ADS1115 @0x48) — unused while EMG is disabled
// static const int EMG_SDA_PIN = 3; // A2
// static const int EMG_SCL_PIN = 4; // A3

// static const uint32_t EMG_INTERVAL_MS    = 2;   // 500 Hz
// static const uint32_t EDA_INTERVAL_MS     = 50;  // ~20Hz — plenty for GSR
static const uint32_t IMU_PPG_INTERVAL_MS = 20; // 50 Hz


// static uint32_t last_emg_ms    = 0;
// static uint32_t last_eda_ms     = 0;
static uint32_t last_imu_ppg_ms = 0;

void setup() {
    stream_init(921600);

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 100000);
    // Wire1.begin(EMG_SDA_PIN, EMG_SCL_PIN, 400000);  // Bus 0 — EMG alone, can run fast-mode

    if (!imu_init()) {
        // Halt — sensor failure is unrecoverable at this phase
        while (true) { delay(1000); }
    }
    if (!ppg_init()) {
        while (true) { delay(1000); }
    }
    if (!thermal_init(Wire)) {
        while (true) { delay(1000); }
    }
    // if (!emg_init(Wire1)) {
    //     while (true) { delay(1000); }
    // }
    // if (!eda_init(Wire)) {
    //     while (true) { delay(1000); }
    // }
}

void loop() {
    uint32_t now = millis();
    // Serial.println("alive");

    // EMG disabled — 3-sensor setup (IMU, PPG, thermal) for now.
    // if (now - last_emg_ms >= EMG_INTERVAL_MS) {
    //     last_emg_ms = now;
    //     EMGReading emg;
    //     if (emg_read(Wire1, &emg)) {
    //         EmgPacket pkt;
    //         packet_fill_emg(&pkt, now, emg);
    //         stream_send_emg(&pkt);
    //     }
    // }

    // EDA disabled — 3-sensor setup (IMU, PPG, thermal) for now.
    // if (now - last_eda_ms >= EDA_INTERVAL_MS) {
    //     last_eda_ms = now;
    //     EDAReading eda;
    //     if (eda_read(Wire, &eda)) {
    //         EdaPacket pkt;
    //         packet_fill_eda(&pkt, now, eda);
    //         stream_send_eda(&pkt);
    //     }
    // }

    // IMU + PPG + thermal at 50 Hz (shared I2C bus, read together each tick)
    if (now - last_imu_ppg_ms >= IMU_PPG_INTERVAL_MS) {
        last_imu_ppg_ms = now;
        IMUReading imu;
        PPGReading ppg;
        ThermalReading thermal;
        if (imu_read(&imu) && ppg_read(&ppg) && thermal_read(Wire, &thermal)) {
            ImuPpgPacket pkt;
            packet_fill_imu_ppg(&pkt, now, imu, ppg, thermal);
            stream_send_imu_ppg(&pkt);
        }
    }
}
