#include "emg.h"
#include <Adafruit_ADS1X15.h>

static Adafruit_ADS1115 _ads;

bool emg_init(TwoWire &bus, int sda_pin, int scl_pin, uint32_t freq_hz) {
    if (!_ads.begin(ADS1X15_ADDRESS, &bus)) return false;
    bus.begin(sda_pin, scl_pin, freq_hz); // re-apply - begin() can silently reset this

    _ads.setGain(GAIN_ONE);                // +/-4.096V range
    _ads.setDataRate(RATE_ADS1115_860SPS); // fastest single-shot rate available
    return true;
}

bool emg_read(TwoWire &bus, EMGReading* out) {
    out->raw = _ads.readADC_SingleEnded(0); // blocking single-shot, channel A0
    return true;
}