
#pragma once
#include <stdint.h>
#include <Wire.h>

struct EMGReading {
    int16_t raw; // raw ADS1115 conversion, single-ended, channel A0
};

// EMG's ADS1115 lives on its own dedicated I2C bus (Wire1) for timing isolation.
bool emg_init(TwoWire &bus);
bool emg_read(TwoWire &bus, EMGReading* out);