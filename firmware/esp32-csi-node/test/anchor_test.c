/**
 * @file anchor_test.c
 * @brief Host unit test for anchor.c (beacon TX and receiver anchor lock).
 *
 * Build (host gcc/clang, no ESP-IDF needed):
 *   gcc -std=gnu11 -Wall -Werror -Istubs -I../main anchor_test.c \
 *       ../main/anchor.c stubs/esp_stubs.c -o anchor_test -lm
 *
 * Prints "anchor_test: OK" and exits 0 on success.
 */

#include "esp_stubs.h"
#include "esp_now.h"
#include "nvs_config.h"
#include "anchor.h"

#include <stdio.h>
#include <string.h>

nvs_config_t      g_nvs_config;
esp_now_recv_cb_t g_stub_espnow_recv_cb = NULL;
uint8_t           g_stub_espnow_last[64];
int               g_stub_espnow_last_len = 0;
int               g_stub_espnow_sends = 0;
wifi_phy_rate_t   g_stub_espnow_rate = 0;

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

static anchor_beacon_t last_beacon(void)
{
    anchor_beacon_t b;
    memset(&b, 0, sizeof(b));
    memcpy(&b, g_stub_espnow_last, sizeof(b));
    return b;
}

static void test_anchor_beacons(void)
{
    g_stub_last_timer_cb = NULL;
    CHECK(anchor_start(7, 100) == ESP_OK);
    CHECK(g_stub_espnow_rate == WIFI_PHY_RATE_6M);      /* OFDM, so receivers get CSI */
    CHECK(g_stub_last_timer_cb != NULL);
    if (g_stub_last_timer_cb == NULL) return;

    g_stub_last_timer_cb(NULL);
    g_stub_last_timer_cb(NULL);
    anchor_beacon_t b = last_beacon();
    CHECK(g_stub_espnow_last_len == (int)sizeof(anchor_beacon_t));
    CHECK(b.magic == ANCHOR_BEACON_MAGIC);
    CHECK(b.node_id == 7);
    CHECK(b.rate_hz_div == 50);
    CHECK(b.seq == 2);
}

static void test_anchor_rate_clamped(void)
{
    CHECK(anchor_start(1, 5) == ESP_OK);                /* below ANCHOR_HZ_MIN */
    g_stub_last_timer_cb(NULL);
    CHECK(last_beacon().rate_hz_div == ANCHOR_HZ_DEFAULT / 2);
}

static void deliver(const uint8_t src[6], const void *data, int len)
{
    uint8_t addr[6];
    memcpy(addr, src, 6);
    esp_now_recv_info_t info = { .src_addr = addr, .des_addr = NULL, .rx_ctrl = NULL };
    g_stub_espnow_recv_cb(&info, (const uint8_t *)data, len);
}

static void test_receiver_locks_to_first_anchor(void)
{
    memset(&g_nvs_config, 0, sizeof(g_nvs_config));
    g_stub_espnow_recv_cb = NULL;
    CHECK(anchor_listen_start() == ESP_OK);
    CHECK(g_stub_espnow_recv_cb != NULL);
    if (g_stub_espnow_recv_cb == NULL) return;

    const uint8_t anchor_mac[6] = {0x24, 0x6f, 0x28, 0x01, 0x02, 0x03};
    const uint8_t other_mac[6]  = {0x24, 0x6f, 0x28, 0x09, 0x09, 0x09};
    anchor_beacon_t beacon = { .magic = ANCHOR_BEACON_MAGIC, .node_id = 0, .rate_hz_div = 50 };

    anchor_beacon_t wrong = beacon;
    wrong.magic = 0xDEADBEEF;
    deliver(other_mac, &wrong, sizeof(wrong));          /* not a beacon */
    deliver(other_mac, &beacon, 8);                     /* truncated */
    CHECK(g_nvs_config.filter_mac_set == 0);

    deliver(anchor_mac, &beacon, sizeof(beacon));
    CHECK(g_nvs_config.filter_mac_set == 1);
    CHECK(memcmp(g_nvs_config.filter_mac, anchor_mac, 6) == 0);

    deliver(other_mac, &beacon, sizeof(beacon));        /* a second anchor is ignored */
    CHECK(memcmp(g_nvs_config.filter_mac, anchor_mac, 6) == 0);
}

static void test_receiver_respects_provisioned_filter(void)
{
    memset(&g_nvs_config, 0, sizeof(g_nvs_config));
    const uint8_t fixed[6] = {1, 2, 3, 4, 5, 6};
    memcpy(g_nvs_config.filter_mac, fixed, 6);
    g_nvs_config.filter_mac_set = 1;
    g_stub_espnow_recv_cb = NULL;
    CHECK(anchor_listen_start() == ESP_OK);
    CHECK(g_stub_espnow_recv_cb == NULL);               /* never listens */
}

int main(void)
{
    test_anchor_beacons();
    test_anchor_rate_clamped();
    test_receiver_locks_to_first_anchor();
    test_receiver_respects_provisioned_filter();
    if (s_failures == 0) {
        printf("anchor_test: OK\n");
        return 0;
    }
    printf("anchor_test: %d failure(s)\n", s_failures);
    return 1;
}
