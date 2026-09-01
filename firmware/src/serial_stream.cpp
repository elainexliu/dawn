#include "serial_stream.h"
#include <Arduino.h>

void stream_init(unsigned long baud) {
    Serial.begin(baud);
    while (!Serial) { /* wait for USB CDC on XIAO */ }
}

void stream_send_imu_ppg(const ImuPpgPacket* pkt) {
    Serial.write(reinterpret_cast<const uint8_t*>(pkt), sizeof(ImuPpgPacket));
}

void stream_send_emg(const EmgPacket* pkt) {
    Serial.write(reinterpret_cast<const uint8_t*>(pkt), sizeof(EmgPacket));
}

void stream_send_eda(const EdaPacket* pkt) {
    Serial.write(reinterpret_cast<const uint8_t*>(pkt), sizeof(EdaPacket));
}
