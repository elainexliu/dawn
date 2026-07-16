#pragma once
#include "packet.h"

void stream_init(unsigned long baud);
void stream_send_imu_ppg(const ImuPpgPacket* pkt);
void stream_send_emg(const EmgPacket* pkt);
void stream_send_eda(const EdaPacket* pkt);