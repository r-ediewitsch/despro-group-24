#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>

// --- Wi-Fi & HiveMQ Configuration ---
const char* WIFI_SSID     = "Asd";
const char* WIFI_PASSWORD = "w3zrabfz";

// HiveMQ Public Broker matching receiver configuration
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;

// Topics matching receiver.ino
const char* TOPIC_CMD     = "my_stepper_esp32/cmd";
const char* TOPIC_STATUS  = "my_stepper_esp32/status";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

unsigned long lastMqttReconnectAttempt = 0;
unsigned long lastSerialCheckTime = 0;
const unsigned long SERIAL_POLL_INTERVAL_MS = 50; // Periodic polling interval

// Buffer for non-blocking serial input accumulation
String inputBuffer = "";

void printHelp() {
  Serial.println("\n--- Stepper MQTT Transmitter ---");
  Serial.println("Enter commands to transmit via MQTT:");
  Serial.println("  M <steps>   : Move relative steps (e.g., 'M 200' or 'M -400')");
  Serial.println("  G <pos>     : Go to absolute position (e.g., 'G 1000')");
  Serial.println("  S <speed>   : Set max speed in steps/sec (e.g., 'S 800')");
  Serial.println("  A <accel>   : Set acceleration in steps/sec^2 (e.g., 'A 400')");
  Serial.println("  E <1/0>     : Enable (1) or Disable (0) driver outputs");
  Serial.println("  STOP        : Halt movement immediately");
  Serial.println("  STATUS      : Request current status from receiver");
  Serial.println("--------------------------------\n");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String response = "";
  for (unsigned int i = 0; i < length; i++) {
    response += (char)payload[i];
  }
  response.trim();
  Serial.printf("[STATUS FEEDBACK] %s\n", response.c_str());
}

void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;

  if (millis() - lastMqttReconnectAttempt > 5000) {
    lastMqttReconnectAttempt = millis();
    String clientId = "ESP32-Sender-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    
    Serial.printf("[MQTT] Connecting to broker as %s...\n", clientId.c_str());
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("[MQTT] Connected!");
      mqttClient.subscribe(TOPIC_STATUS); // Listen for feedback from receiver
      printHelp();
    } else {
      Serial.printf("[MQTT] Connection failed, rc=%d. Retrying in 5s...\n", mqttClient.state());
    }
  }
}

void processPeriodicSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        if (mqttClient.connected()) {
          bool published = mqttClient.publish(TOPIC_CMD, inputBuffer.c_str());
          if (published) {
            Serial.printf("[TX -> %s] %s\n", TOPIC_CMD, inputBuffer.c_str());
          } else {
            Serial.println("[ERROR] Failed to publish message via MQTT.");
          }
        } else {
          Serial.println("[ERROR] Cannot send command: MQTT broker not connected.");
        }
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000);

  // Connect to Wi-Fi
  Serial.printf("\n[WIFI] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

  // Setup MQTT Client
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
}

void loop() {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop();
  }

  // Periodically read user Serial input
  if (millis() - lastSerialCheckTime >= SERIAL_POLL_INTERVAL_MS) {
    lastSerialCheckTime = millis();
    processPeriodicSerialInput();
  }
}