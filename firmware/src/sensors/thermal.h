// thermal.h
#pragma once
#include <stdint.h>
#include <Wire.h>

struct ThermalReading {
    float ambient_c;  // Ta - sensor's own ambient/die temperature
    float object_c;   // Tobj1 - target (skin) temperature
};

// Verifies the MLX90614 responds on the given bus. No wake/config needed -
// unlike the MPU6050, this sensor has no sleep mode to clear.
bool thermal_init(TwoWire &bus);

// Reads both ambient and object temperature registers.
bool thermal_read(TwoWire &bus, ThermalReading* out);