/**
 * @file anchor.c
 * @brief Fixed-rate ESP-NOW anchor transmitter and receiver-side anchor lock.
 */

#include "anchor.h"
#include "nvs_config.h"

#include <string.h>
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"

static const char *TAG = "anchor";

extern nvs_config_t g_nvs_config;

static const uint8_t s_broadcast[ESP_NOW_ETH_ALEN] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

static esp_timer_handle_t s_beacon_timer = NULL;
static anchor_beacon_t    s_beacon;
static uint16_t           s_rate_hz = ANCHOR_HZ_DEFAULT;
static volatile uint32_t  s_sent = 0;
static volatile uint32_t  s_failed = 0;

/* ---- Anchor role ---------------------------------------------------- */

static void beacon_timer_cb(void *arg)
{
    (void)arg;
    s_beacon.seq++;
    if (esp_now_send(s_broadcast, (const uint8_t *)&s_beacon, sizeof(s_beacon)) == ESP_OK) {
        s_sent++;
    } else {
        s_failed++;   /* Usually ESP_ERR_ESPNOW_NO_MEM: TX queue momentarily full. */
    }
}

/**
 * CSI needs an OFDM preamble. ESP-NOW defaults to 1 Mbps DSSS (802.11b),
 * which carries no OFDM training fields, so receivers would get no CSI.
 * Force 6 Mbps OFDM for the broadcast peer.
 */
static esp_err_t set_ofdm_rate(void)
{
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 1, 0)
    esp_now_rate_config_t rate_cfg = {
        .phymode = WIFI_PHY_MODE_11G,
        .rate    = WIFI_PHY_RATE_6M,
        .ersu    = false,
        .dcm     = false,
    };
    return esp_now_set_peer_rate_config(s_broadcast, &rate_cfg);
#else
    return esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_6M);
#endif
}

esp_err_t anchor_start(uint8_t node_id, uint16_t rate_hz)
{
    if (rate_hz < ANCHOR_HZ_MIN || rate_hz > ANCHOR_HZ_MAX) {
        ESP_LOGW(TAG, "anchor_hz=%u out of range, using %u",
                 (unsigned)rate_hz, (unsigned)ANCHOR_HZ_DEFAULT);
        rate_hz = ANCHOR_HZ_DEFAULT;
    }
    s_rate_hz = rate_hz;

    esp_err_t err = esp_now_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_init failed: %s", esp_err_to_name(err));
        return err;
    }

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, s_broadcast, ESP_NOW_ETH_ALEN);
    peer.channel = 0;              /* Current channel. */
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    err = esp_now_add_peer(&peer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_add_peer failed: %s", esp_err_to_name(err));
        return err;
    }

    err = set_ofdm_rate();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Could not force OFDM rate (%s); receivers may get no CSI",
                 esp_err_to_name(err));
    }

    memset(&s_beacon, 0, sizeof(s_beacon));
    s_beacon.magic       = ANCHOR_BEACON_MAGIC;
    s_beacon.node_id     = node_id;
    s_beacon.rate_hz_div = (uint8_t)(rate_hz / 2);

    esp_timer_create_args_t args = {
        .callback = beacon_timer_cb,
        .arg      = NULL,
        .name     = "anchor_tx",
    };
    err = esp_timer_create(&args, &s_beacon_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "beacon timer create failed: %s", esp_err_to_name(err));
        return err;
    }
    err = esp_timer_start_periodic(s_beacon_timer, 1000000ULL / rate_hz);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "beacon timer start failed: %s", esp_err_to_name(err));
        esp_timer_delete(s_beacon_timer);
        s_beacon_timer = NULL;
        return err;
    }

    uint8_t mac[6] = {0};
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Anchor beacons started: %u Hz, 6 Mbps OFDM, MAC %02x:%02x:%02x:%02x:%02x:%02x",
             (unsigned)rate_hz, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return ESP_OK;
}

void anchor_log_stats(void)
{
    ESP_LOGI(TAG, "beacons sent=%lu failed=%lu (target %u Hz)",
             (unsigned long)s_sent, (unsigned long)s_failed, (unsigned)s_rate_hz);
}

/* ---- Receiver role -------------------------------------------------- */

static void beacon_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    if (g_nvs_config.filter_mac_set || info == NULL || data == NULL ||
        len < (int)sizeof(anchor_beacon_t)) {
        return;
    }
    anchor_beacon_t beacon;
    memcpy(&beacon, data, sizeof(beacon));
    if (beacon.magic != ANCHOR_BEACON_MAGIC) {
        return;
    }

    /* Runs in the WiFi task, the same context as the CSI callback, so the
     * filter can be updated without a lock. Write the MAC before the flag. */
    memcpy(g_nvs_config.filter_mac, info->src_addr, 6);
    g_nvs_config.filter_mac_set = 1;

    const uint8_t *m = info->src_addr;
    ESP_LOGI(TAG, "Locked CSI to anchor node %u (%02x:%02x:%02x:%02x:%02x:%02x, %u Hz)",
             (unsigned)beacon.node_id, m[0], m[1], m[2], m[3], m[4], m[5],
             (unsigned)beacon.rate_hz_div * 2);
}

esp_err_t anchor_listen_start(void)
{
    if (g_nvs_config.filter_mac_set) {
        ESP_LOGI(TAG, "Filter MAC already set; not listening for anchor");
        return ESP_OK;
    }
    esp_err_t err = esp_now_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_init failed: %s", esp_err_to_name(err));
        return err;
    }
    err = esp_now_register_recv_cb(beacon_recv_cb);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_register_recv_cb failed: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "Listening for anchor beacons");
    return ESP_OK;
}
