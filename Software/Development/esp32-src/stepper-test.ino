#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// --- Wi-Fi & HiveMQ Configuration ---
const char* WIFI_SSID     = "iQOO Z9 5G";
const char* WIFI_PASSWORD = "nggaktau";

// HiveMQ Public Broker
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;

// Topic definitions (Customize prefix to avoid collisions on public broker)
const char* TOPIC_CMD     = "my_stepper_esp32/cmd";
const char* TOPIC_STATUS  = "my_stepper_esp32/status";

// --- Hardware Pins ---
#define STEP_PIN   14
#define DIR_PIN    12
#define ENABLE_PIN 21

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
WiFiClient espClient;
PubSubClient mqttClient(espClient);

unsigned long lastMqttReconnectAttempt = 0;
bool wasRunning = false;

// Forward declarations
void handleCommand(String cmd);
void mqttCallback(char* topic, byte* payload, unsigned int length);
void reconnectMQTT();

void setup() {
  Serial.begin(115200);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); // Active LOW -> Enable driver

  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(500.0);

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  // Setup HiveMQ connection
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  message.trim();

  // Print only the incoming MQTT message to Serial Monitor
  Serial.println(message);

  handleCommand(message);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = toupper(cmd.charAt(0));
  String argument = cmd.substring(1);
  argument.trim();

  char response[128];

  switch (type) {
    case 'M': { // Relative move
      long steps = argument.toInt();
      stepper.move(steps);
      snprintf(response, sizeof(response), "[MOVE] Relative target: %ld steps", steps);
      break;
    }
    case 'G': { // Absolute move
      long target = argument.toInt();
      stepper.moveTo(target);
      snprintf(response, sizeof(response), "[GOTO] Absolute target: %ld", target);
      break;
    }
    case 'S': { // Set Max Speed
      float spd = argument.toFloat();
      if (spd > 0) {
        stepper.setMaxSpeed(spd);
        snprintf(response, sizeof(response), "[SPEED] Max speed: %.1f steps/s", spd);
      }
      break;
    }
    case 'A': { // Set Acceleration
      float acc = argument.toFloat();
      if (acc > 0) {
        stepper.setAcceleration(acc);
        snprintf(response, sizeof(response), "[ACCEL] Acceleration: %.1f steps/s^2", acc);
      }
      break;
    }
    case 'E': { // Enable/Disable Driver
      int en = argument.toInt();
      digitalWrite(ENABLE_PIN, en == 1 ? LOW : HIGH);
      snprintf(response, sizeof(response), "[ENABLE] Driver %s", en == 1 ? "ENABLED" : "DISABLED");
      break;
    }
    default: {
      if (cmd.equalsIgnoreCase("STOP")) {
        stepper.stop();
        snprintf(response, sizeof(response), "[HALT] Motor stopped");
      } else if (cmd.equalsIgnoreCase("STATUS")) {
        snprintf(response, sizeof(response), "Pos: %ld | Target: %ld | Running: %s",
                 stepper.currentPosition(),
                 stepper.targetPosition(),
                 stepper.isRunning() ? "YES" : "NO");
      } else {
        snprintf(response, sizeof(response), "[ERROR] Unknown command: %s", cmd.c_str());
      }
      break;
    }
  }

  if (mqttClient.connected()) {
    mqttClient.publish(TOPIC_STATUS, response);
  }
}

void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;

  // Non-blocking reconnect attempt every 5 seconds
  if (millis() - lastMqttReconnectAttempt > 5000) {
    lastMqttReconnectAttempt = millis();
    
    String clientId = "ESP32-HiveMQ-" + String((uint32_t)ESP.getEfuseMac(), HEX);

    if (mqttClient.connect(clientId.c_str())) {
      mqttClient.subscribe(TOPIC_CMD);
      mqttClient.publish(TOPIC_STATUS, "[ONLINE] ESP32 Stepper Controller Ready");
    }
  }
}

void loop() {
  stepper.run();

  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop();
  }

  // Publish status report when the target is reached
  if (wasRunning && !stepper.isRunning()) {
    char doneMsg[64];
    snprintf(doneMsg, sizeof(doneMsg), "[DONE] Reached target at step %ld", stepper.currentPosition());
    if (mqttClient.connected()) {
      mqttClient.publish(TOPIC_STATUS, doneMsg);
    }
  }
  wasRunning = stepper.isRunning();

  // Fallback Serial input
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    handleCommand(input);
  }
}