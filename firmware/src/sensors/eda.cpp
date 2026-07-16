// eda.cpp
#include "eda.h"
#include <Adafruit_ADS1X15.h>

static Adafruit_ADS1115 _ads;

bool eda_init(TwoWire &bus) {
    // Different address than any other device on this shared bus.
    // Tie this chip's ADDR pin to VDD for address 0x49 (default 0x48 is GND —
    // already used if any other ADS1115 shares this specific bus).
    if (!_ads.begin(0x49, &bus)) return false;

    _ads.setGain(GAIN_ONE);                 // ±4.096V full scale
    _ads.setDataRate(RATE_ADS1115_128SPS);  // GSR only needs ~10-32Hz — no need to run fast
    _ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
    return true;
}

bool eda_read(TwoWire &bus, EDAReading* out) {
    if (!_ads.conversionComplete()) return false;
    out->raw = _ads.getLastConversionResults();
    return true;
}