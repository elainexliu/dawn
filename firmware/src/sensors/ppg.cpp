#include "ppg.h"
#include <MAX30105.h> // SparkFun MAX3010x library (MAX30105 = same silicon as MAX30102)
#include <Wire.h>

static MAX30105 _sensor;

bool ppg_init() {
    if (!_sensor.begin(Wire, I2C_SPEED_STANDARD)) return false;

    // Heart-rate mode: IR + Red LEDs, 25 samples/s, 411µs pulse, 16-bit ADC
    _sensor.setup(
        60,          // LED brightness (0–255)
        4,           // sample average
        2,           // LED mode: 1=IR only, 2=IR+Red, 3=IR+Red+Green
        25,          // sample rate (Hz)
        411,         // pulse width (µs) — 18-bit ADC resolution
        4096         // ADC range
    );
    return true;
}

bool ppg_read(PPGReading* out) {
    out->ir  = _sensor.getIR();
    out->red = _sensor.getRed();
    return true;
}
