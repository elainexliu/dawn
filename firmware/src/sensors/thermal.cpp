// thermal.cpp
#include "thermal.h"

static const uint8_t MLX90614_ADDR   = 0x5A;
static const uint8_t REG_TA          = 0x06; // ambient temperature RAM register
static const uint8_t REG_TOBJ1       = 0x07; // object 1 temperature RAM register

// Reads a 16-bit RAM register plus its trailing PEC byte (read, not verified).
static bool read_word(TwoWire &bus, uint8_t reg, uint16_t* raw_out) {
    bus.beginTransmission(MLX90614_ADDR);
    bus.write(reg);
    if (bus.endTransmission(false) != 0) return false; // repeated start, no stop

    if (bus.requestFrom(MLX90614_ADDR, (uint8_t)3) != 3) return false;

    uint8_t low  = bus.read();
    uint8_t high = bus.read();
    bus.read(); // PEC byte — discarded for now, not CRC-checked

    *raw_out = ((uint16_t)high << 8) | low;
    return true;
}

bool thermal_init(TwoWire &bus) {
    uint16_t raw;
    // No wake/config step exists for this sensor — just confirm it responds.
    return read_word(bus, REG_TA, &raw);
}

bool thermal_read(TwoWire &bus, ThermalReading* out) {
    uint16_t ta_raw, tobj_raw;

    if (!read_word(bus, REG_TA, &ta_raw)) return false;
    if (!read_word(bus, REG_TOBJ1, &tobj_raw)) return false;

    // Bit 15 of the object register is an error/status flag, not temperature data.
    tobj_raw &= 0x7FFF;

    // Conversion per datasheet: raw * 0.02 gives Kelvin, in 0.02K steps.
    out->ambient_c = (ta_raw * 0.02f) - 273.15f;
    out->object_c  = (tobj_raw * 0.02f) - 273.15f;

    return true;
}