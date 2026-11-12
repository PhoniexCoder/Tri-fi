/**
 * @file wire_dump.c
 * @brief Host tool: emit firmware wire-format packets as hex for decoder tests.
 *
 * Builds packets with the real firmware code (csi_serialize_frame) and the
 * real packed structs (edge_vitals_pkt_t, edge_fused_vitals_pkt_t), so the
 * Python decoder can be checked against what the node actually sends.
 *
 * Build (host gcc/clang, no ESP-IDF needed):
 *   gcc -std=gnu11 -Istubs -I../main wire_dump.c ../main/csi_collector.c \
 *       stubs/esp_stubs.c -o wire_dump -lm
 *
 * Used by tests/test_csi_frame_parser.py.
 */

#include "esp_stubs.h"
#include "nvs_config.h"
#include "csi_collector.h"
#include "edge_processing.h"

#include <stdio.h>
#include <string.h>

nvs_config_t g_nvs_config;

/* csi_collector.c forwards frames to the edge DSP; not under test here. */
bool edge_enqueue_csi(const uint8_t *iq_data, uint16_t iq_len,
                      int8_t rssi, uint8_t channel)
{
    (void)iq_data; (void)iq_len; (void)rssi; (void)channel;
    return true;
}

static void dump(const char *tag, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    printf("%s ", tag);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", p[i]);
    }
    printf("\n");
}

int main(void)
{
    /* Raw CSI frame: node 3, 64 subcarriers, known I/Q pattern. */
    g_nvs_config.node_id = 3;

    int8_t iq[128];
    for (int i = 0; i < 128; i++) {
        iq[i] = (int8_t)(i * 7 - 60);
    }

    wifi_csi_info_t info;
    memset(&info, 0, sizeof(info));
    info.rx_ctrl.rssi = -47;
    info.rx_ctrl.channel = 6;
    info.rx_ctrl.noise_floor = (uint8_t)(int8_t)-92;
    info.len = (int16_t)sizeof(iq);
    info.buf = iq;

    uint8_t buf[CSI_MAX_FRAME_SIZE];
    size_t n = csi_serialize_frame(&info, buf, sizeof(buf));
    dump("raw", buf, n);
    n = csi_serialize_frame(&info, buf, sizeof(buf));
    dump("raw2", buf, n);

    /* Vitals packet. */
    edge_vitals_pkt_t v;
    memset(&v, 0, sizeof(v));
    v.magic = EDGE_VITALS_MAGIC;
    v.node_id = 2;
    v.flags = 0x05;                         /* presence + motion */
    v.breathing_rate = (uint16_t)(14.5f * 100.0f);
    v.heartrate = (uint32_t)(72.25f * 10000.0f);
    v.rssi = -61;
    v.n_persons = 1;
    v.motion_energy = 0.375f;
    v.presence_score = 0.8125f;
    v.timestamp_ms = 123456;
    dump("vitals", &v, sizeof(v));

    /* Fused CSI + mmWave vitals packet. */
    edge_fused_vitals_pkt_t f;
    memset(&f, 0, sizeof(f));
    f.magic = EDGE_FUSED_MAGIC;
    f.node_id = 1;
    f.flags = 0x09;                         /* presence + mmwave_present */
    f.breathing_rate = (uint16_t)(15.25f * 100.0f);
    f.heartrate = (uint32_t)(71.5f * 10000.0f);
    f.rssi = -55;
    f.n_persons = 2;
    f.mmwave_type = 2;
    f.fusion_confidence = 90;
    f.motion_energy = 1.5f;
    f.presence_score = 0.9375f;
    f.timestamp_ms = 654321;
    f.mmwave_hr_bpm = 71.5f;
    f.mmwave_br_bpm = 15.25f;
    f.mmwave_distance = 132.5f;
    f.mmwave_targets = 1;
    f.mmwave_confidence = 80;
    dump("fused", &f, sizeof(f));

    return 0;
}
