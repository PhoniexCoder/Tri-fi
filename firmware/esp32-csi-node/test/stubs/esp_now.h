/**
 * Stub: minimal ESP-NOW API for host tests of anchor.c.
 * Records sends, the receive callback and the peer rate so tests can inspect them.
 */
#ifndef ESP_NOW_H_STUB
#define ESP_NOW_H_STUB

#include "esp_stubs.h"

#define ESP_NOW_ETH_ALEN 6

typedef int wifi_phy_mode_t;
typedef int wifi_phy_rate_t;
#define WIFI_PHY_MODE_11G  2
#define WIFI_PHY_RATE_6M   0x0B

typedef struct {
    uint8_t          peer_addr[ESP_NOW_ETH_ALEN];
    uint8_t          lmk[16];
    uint8_t          channel;
    wifi_interface_t ifidx;
    bool             encrypt;
    void            *priv;
} esp_now_peer_info_t;

typedef struct {
    uint8_t *src_addr;
    uint8_t *des_addr;
    void    *rx_ctrl;
} esp_now_recv_info_t;

typedef struct {
    wifi_phy_mode_t phymode;
    wifi_phy_rate_t rate;
    bool            ersu;
    bool            dcm;
} esp_now_rate_config_t;

typedef void (*esp_now_recv_cb_t)(const esp_now_recv_info_t *info, const uint8_t *data, int len);

/* Capture state, defined by the test program. */
extern esp_now_recv_cb_t g_stub_espnow_recv_cb;
extern uint8_t           g_stub_espnow_last[64];
extern int               g_stub_espnow_last_len;
extern int               g_stub_espnow_sends;
extern wifi_phy_rate_t   g_stub_espnow_rate;

static inline esp_err_t esp_now_init(void) { return ESP_OK; }
static inline esp_err_t esp_now_add_peer(const esp_now_peer_info_t *p) { (void)p; return ESP_OK; }
static inline esp_err_t esp_now_register_recv_cb(esp_now_recv_cb_t cb) {
    g_stub_espnow_recv_cb = cb;
    return ESP_OK;
}
static inline esp_err_t esp_now_set_peer_rate_config(const uint8_t *addr, esp_now_rate_config_t *cfg) {
    (void)addr;
    g_stub_espnow_rate = cfg->rate;
    return ESP_OK;
}
static inline esp_err_t esp_now_send(const uint8_t *addr, const uint8_t *data, size_t len) {
    (void)addr;
    if (len > sizeof(g_stub_espnow_last)) len = sizeof(g_stub_espnow_last);
    memcpy(g_stub_espnow_last, data, len);
    g_stub_espnow_last_len = (int)len;
    g_stub_espnow_sends++;
    return ESP_OK;
}
static inline esp_err_t esp_wifi_get_mac(wifi_interface_t ifx, uint8_t mac[6]) {
    (void)ifx;
    memset(mac, 0xAA, 6);
    return ESP_OK;
}

#endif
