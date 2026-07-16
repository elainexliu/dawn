#pragma once
#include <stdint.h>

struct IMUReading {
    int16_t ax, ay, az; // raw accel (÷16384 → g, ±2g)
    int16_t gx, gy, gz; // raw gyro  (÷131 → °/s, ±250°/s)
};

// Initialize MPU6050: wake, set DLPF, configure ±2g / ±250°/s
bool imu_init();

// Read 14 bytes from register 0x3B in one burst
bool imu_read(IMUReading* out);
