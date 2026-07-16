#pragma once
#include <stdint.h>

struct PPGReading {
    uint32_t ir;
    uint32_t red;
};

bool ppg_init();
bool ppg_read(PPGReading* out);
