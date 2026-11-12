/**
 * @file anchor.h
 * @brief Fixed-rate anchor transmitter and receiver-side anchor lock.
 *
 * One node in the kit runs as the anchor: it broadcasts small ESP-NOW frames
 * at a fixed rate (default 100 Hz) on the current WiFi channel. Every other
 * node (a receiver) measures CSI on those frames, so the sample rate and the
 * transmitter position are known and controlled, independent of other
 * traffic on the network.
 *
 * Receivers find the anchor automatically: the first valid beacon they hear
 * locks their CSI MAC filter to the anchor's MAC (unless a filter MAC was
 * provisioned explicitly).
 */

#ifndef ANCHOR_H
#define ANCHOR_H

#include <stdint.h>
#include "esp_err.h"

/** Beacon payload magic (registered alongside the UDP packet magics). */
#define ANCHOR_BEACON_MAGIC  0xC5110006

#define ANCHOR_HZ_MIN       10
#define ANCHOR_HZ_MAX       200
#define ANCHOR_HZ_DEFAULT   100

/** Beacon payload carried in each ESP-NOW frame (12 bytes). */
typedef struct __attribute__((packed)) {
    uint32_t magic;      /**< ANCHOR_BEACON_MAGIC. */
    uint8_t  node_id;    /**< Anchor node ID. */
    uint8_t  rate_hz_div;/**< Beacon rate / 2 (fits 10-200 Hz in a byte). */
    uint16_t reserved;
    uint32_t seq;        /**< Increments every beacon. */
} anchor_beacon_t;

_Static_assert(sizeof(anchor_beacon_t) == 12, "anchor beacon must be 12 bytes");

/**
 * Start broadcasting beacons (anchor role). WiFi must already be started
 * in STA mode; beacons go out on the channel the station is on.
 *
 * @param node_id  This node's ID, carried in each beacon.
 * @param rate_hz  Beacon rate, clamped to [ANCHOR_HZ_MIN, ANCHOR_HZ_MAX].
 */
esp_err_t anchor_start(uint8_t node_id, uint16_t rate_hz);

/** Log beacon counters (anchor role). */
void anchor_log_stats(void);

/**
 * Listen for anchor beacons and lock the CSI MAC filter to the first anchor
 * heard (receiver role). No-op if a filter MAC is already set.
 */
esp_err_t anchor_listen_start(void);

#endif /* ANCHOR_H */
