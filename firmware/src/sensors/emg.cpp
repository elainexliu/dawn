#include "emg.h"
#include <Adafruit_ADS1X15.h>

static Adafruit_ADS1115 _ads;

bool emg_init(TwoWire &bus) {
    if (!_ads.begin(ADS1X15_ADDRESS, &bus)) return false; // default addr 0x48

    _ads.setGain(GAIN_ONE);              // ±4.096V range
    _ads.setDataRate(RATE_ADS1115_860SPS); // fastest available — EMG needs it
    _ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, /*continuous=*/true);
    return true;
}

bool emg_read(TwoWire &bus, EMGReading* out) {
    if (!_ads.conversionComplete()) return false;
    out->raw = _ads.getLastConversionResults();
    return true;
}