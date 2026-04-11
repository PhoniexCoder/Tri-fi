/*
 * Trifi Rescue System - ESP8266 Rescuer Device
 * -------------------------------------------
 * This device sends status updates to the backend to track the rescuer
 * position and confirm successful rescues via a button press.
 *
 * Requirements:
 * - ESP8266 (NodeMCU, Wemos D1 Mini, etc.)
 * - WiFi connection to the same network as the backend.
 * - Button on GPIO 0 (Flash button on NodeMCU).
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// --- Configuration ---
const char* ssid     = "YOUR_SSID";         // TODO: User to set
const char* password = "YOUR_PASSWORD";     // TODO: User to set
const char* backend_ip = "192.168.1.100";   // TODO: User to set
const int udp_port   = 4444;

const int BUTTON_PIN = 0; // GPIO 0 (Flash button)

WiFiUDP udp;
unsigned long last_active_tx = 0;
const int active_interval = 200; // 5Hz heartbeat

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.println("\n--- Trifi Rescuer Device ---");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  unsigned long now = millis();

  // 1. Send Active Heartbeat
  if (now - last_active_tx >= active_interval) {
    last_active_tx = now;
    send_udp_msg("RESCUER_ACTIVE");
  }

  // 2. Check for Button Press (Rescue Confirmation)
  if (digitalRead(BUTTON_PIN) == LOW) {
    Serial.println("!!! RESCUE CONFIRMED !!!");
    send_udp_msg("RESCUE_CONFIRMED");
    
    // De-bounce and visual feedback
    delay(1000); 
  }
}

void send_udp_msg(const char* msg) {
  udp.beginPacket(backend_ip, udp_port);
  udp.write(msg);
  udp.endPacket();
  Serial.print("Sent: ");
  Serial.println(msg);
}
