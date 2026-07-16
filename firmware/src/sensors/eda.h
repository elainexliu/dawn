// eda.h
#pragma once
#include <stdint.h>
#include <Wire.h>

struct EDAReading {
    int16_t raw; // ADS1115 ch0, GAIN_ONE (±4.096V)
};

// EDA's ADS1115 shares Bus 1 (same bus as IMU, PPG, thermal).
bool eda_init(TwoWire &bus);

// Non-blocking: returns last conversion result from continuous mode.
// Returns false if a new conversion isn't ready yet — expected often,
// since EDA only needs ~10-32Hz while this gets polled every loop tick.
bool eda_read(TwoWire &bus, EDAReading* out);