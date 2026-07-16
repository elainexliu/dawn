#include "imu.h"
#include <Wire.h>
#include <Arduino.h>

static const uint8_t MPU6050_ADDR       = 0x68;
static const uint8_t REG_PWR_MGMT_1     = 0x6B;
static const uint8_t REG_CONFIG         = 0x1A;
static const uint8_t REG_GYRO_CONFIG    = 0x1B;
static const uint8_t REG_ACCEL_CONFIG   = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H   = 0x3B;

bool imu_init() {
    // Wire.begin() is called once in main.cpp — this bus is shared with
    // the PPG and thermal sensors.

    // Wake the MPU6050 (clear sleep bit)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;

    // DLPF bandwidth ~44 Hz (config = 3)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(REG_CONFIG);
    Wire.write(0x03);
    if (Wire.endTransmission() != 0) return false;

    // Gyro full-scale ±250°/s
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(REG_GYRO_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;

    // Accel full-scale ±2g
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(REG_ACCEL_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;

    delay(100); // sensor startup
    return true;
}

bool imu_read(IMUReading* out) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;

    if (Wire.requestFrom(MPU6050_ADDR, (uint8_t)14) != 14) return false;

    uint8_t buf[14];
    for (int i = 0; i < 14; i++) buf[i] = Wire.read();

    out->ax = (int16_t)((buf[0]  << 8) | buf[1]);
    out->ay = (int16_t)((buf[2]  << 8) | buf[3]);
    out->az = (int16_t)((buf[4]  << 8) | buf[5]);
    // buf[6..7] = temperature — skip
    out->gx = (int16_t)((buf[8]  << 8) | buf[9]);
    out->gy = (int16_t)((buf[10] << 8) | buf[11]);
    out->gz = (int16_t)((buf[12] << 8) | buf[13]);

    return true;
}
