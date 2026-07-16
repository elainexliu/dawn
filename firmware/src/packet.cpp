#include "packet.h"
#include "sensors/imu.h"
#include "sensors/ppg.h"
#include "sensors/thermal.h"
#include "sensors/emg.h"
#include "sensors/eda.h"

uint8_t packet_compute_checksum(const uint8_t* buf, size_t start_idx, size_t end_excl) {
    uint8_t cs = 0;
    for (size_t i = start_idx; i < end_excl; i++) {
        cs ^= buf[i];
    }
    return cs;
}

void packet_fill_imu_ppg(ImuPpgPacket* pkt,
                         uint32_t timestamp_ms,
                         const IMUReading& imu,
                         const PPGReading& ppg,
                         const ThermalReading& thermal) {
    pkt->start        = PKT_START;
    pkt->type         = PKT_TYPE_IMU_PPG;
    pkt->version      = PKT_VERSION;
    pkt->timestamp_ms = timestamp_ms;
    pkt->accel_x      = imu.ax;
    pkt->accel_y      = imu.ay;
    pkt->accel_z      = imu.az;
    pkt->gyro_x       = imu.gx;
    pkt->gyro_y       = imu.gy;
    pkt->gyro_z       = imu.gz;
    pkt->ppg_ir       = ppg.ir;
    pkt->ppg_red      = ppg.red;
    pkt->ambient_c    = thermal.ambient_c;
    pkt->object_c     = thermal.object_c;

    // Checksum over bytes[1..34] (type through object_c, inclusive)
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(pkt);
    pkt->checksum = packet_compute_checksum(buf, 1, offsetof(ImuPpgPacket, checksum));
    pkt->end = PKT_END;
}

void packet_fill_emg(EmgPacket* pkt,
                     uint32_t timestamp_ms,
                     const EMGReading& emg) {
    pkt->start        = PKT_START;
    pkt->type         = PKT_TYPE_EMG;
    pkt->version      = PKT_VERSION;
    pkt->timestamp_ms = timestamp_ms;
    pkt->emg_raw      = emg.raw;

    // Checksum over bytes[1..8] (type through emg_raw, inclusive)
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(pkt);
    pkt->checksum = packet_compute_checksum(buf, 1, offsetof(EmgPacket, checksum));
    pkt->end = PKT_END;
}

void packet_fill_eda(EdaPacket* pkt,
                     uint32_t timestamp_ms,
                     const EDAReading& eda) {
    pkt->start        = PKT_START;
    pkt->type         = PKT_TYPE_EDA;
    pkt->version      = PKT_VERSION;
    pkt->timestamp_ms = timestamp_ms;
    pkt->eda_raw      = eda.raw;

    // Checksum over bytes[1..8] (type through eda_raw, inclusive)
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(pkt);
    pkt->checksum = packet_compute_checksum(buf, 1, offsetof(EdaPacket, checksum));
    pkt->end = PKT_END;
}