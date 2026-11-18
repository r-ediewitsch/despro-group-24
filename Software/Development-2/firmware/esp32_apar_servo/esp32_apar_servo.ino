/**
 * =====================================================================
 * DEEZPRO FIRMWARE - ESP32 #3: KENDALI SERVO PEMICU APAR
 * =====================================================================
 * File: esp32_apar_servo.ino
 * Deskripsi: Mengendalikan aktuator servo penekan tuas APAR dengan timer
 *            non-blocking millis(), merespons perintah semprot MQTT,
 *            dan menerbitkan status eksekusi secara real-time.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

// --- Konfigurasi Jaringan & MQTT Broker ---
const char* WIFI_SSID     = "iQOO Z9 5G";
const char* WIFI_PASSWORD = "nggaktau";
const char* MQTT_BROKER   = "broker.emqx.io";
const int   MQTT_PORT     = 1883;

// --- Definisi Topik MQTT APAR ---
const char* TOPIC_CMD     = "deezpro/apar/cmd";
const char* TOPIC_STATUS  = "deezpro/apar/status";
const char* TOPIC_SYS_CMD = "deezpro/system/cmd";

// --- Pinout & Kalibrasi Sudut Servo ---
#define SERVO_PIN         18
#define SERVO_IDLE_ANGLE  0    // Sudut tuas bebas (tidak menekan)
#define SERVO_PRESS_ANGLE 90   // Sudut tuas tertekan penuh (menyemprot)

// --- Inisialisasi Objek ---
Servo aparServo;
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// --- Variabel State & Non-blocking Timer ---
unsigned long lastMqttReconnectAttempt = 0;
bool isSpraying = false;
unsigned long sprayStartTime = 0;
unsigned long sprayDurationMs = 3000;
long activeSeq = 0;

// Forward declarations
void handleJsonCommand(String jsonStr);
void mqttCallback(char* topic, byte* payload, unsigned int length);
void reconnectMQTT();
void publishStatus(const char* state, const char* msg = "");
void triggerSpray(unsigned long durationMs);
void releaseServo();

void setup() {
  Serial.begin(115200);

  // Inisialisasi Servo ESP32
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  aparServo.setPeriodHertz(50); // Standard 50Hz servo
  aparServo.attach(SERVO_PIN, 500, 2400);

  // Posisi awal tuas aman (bebas)
  aparServo.write(SERVO_IDLE_ANGLE);

  // Hubungkan ke Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Menghubungkan ke ");
  Serial.println(WIFI_SSID);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[WiFi] Terhubung! IP: " + WiFi.localIP().toString());

  // Konfigurasi MQTT
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
}

// Ekstraksi field JSON sederhana tanpa parser eksternal
String extractJsonField(String json, String key) {
  int keyIndex = json.indexOf("\"" + key + "\"");
  if (keyIndex == -1) return "";
  int colonIndex = json.indexOf(":", keyIndex);
  if (colonIndex == -1) return "";

  int startIdx = colonIndex + 1;
  while (startIdx < json.length() && (json.charAt(startIdx) == ' ' || json.charAt(startIdx) == '\"')) {
    startIdx++;
  }

  int endIdx = startIdx;
  while (endIdx < json.length() && json.charAt(endIdx) != ',' && json.charAt(endIdx) != '}' && json.charAt(endIdx) != '\"') {
    endIdx++;
  }
  return json.substring(startIdx, endIdx);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  message.trim();
  Serial.print("[MQTT IN APAR] ");
  Serial.println(message);

  handleJsonCommand(message);
}

void handleJsonCommand(String input) {
  input.trim();
  if (input.length() == 0) return;

  String cmd = "";
  unsigned long duration = 3000;
  int testAngle = -1;
  activeSeq = 0;

  // Format JSON Standard: {"cmd": "SPRAY", "duration_ms": 3000, "seq": 201}
  if (input.startsWith("{")) {
    cmd = extractJsonField(input, "cmd");
    cmd.toUpperCase();

    String seqStr = extractJsonField(input, "seq");
    if (seqStr.length() > 0) activeSeq = seqStr.toInt();

    String durStr = extractJsonField(input, "duration_ms");
    if (durStr.length() > 0) duration = durStr.toInt();

    String angleStr = extractJsonField(input, "angle");
    if (angleStr.length() > 0) testAngle = angleStr.toInt();
  } else {
    // Format Shorthand Legacy: SPRAY 3000, RELEASE
    if (input.startsWith("SPRAY")) {
      cmd = "SPRAY";
      String arg = input.substring(5);
      arg.trim();
      if (arg.length() > 0) duration = arg.toInt();
    } else if (input.equalsIgnoreCase("RELEASE") || input.equalsIgnoreCase("STOP")) {
      cmd = "RELEASE";
    } else if (input.equalsIgnoreCase("STATUS")) {
      cmd = "STATUS";
    }
  }

  // Eksekusi Perintah
  if (cmd == "SPRAY") {
    triggerSpray(duration);
  } else if (cmd == "RELEASE" || cmd == "STOP" || cmd == "EMERGENCY_STOP") {
    releaseServo();
    publishStatus("IDLE", "Tuas APAR Dilepas");
  } else if (cmd == "TEST" && testAngle >= 0) {
    aparServo.write(testAngle);
    publishStatus("READY", "Sudut Servo Diuji");
  } else if (cmd == "STATUS") {
    const char* st = isSpraying ? "SPRAYING" : "READY";
    publishStatus(st, isSpraying ? "Sedang Menyemprot" : "Siap Siaga");
  }
}

void triggerSpray(unsigned long durationMs) {
  Serial.printf("[APAR] Menekan tuas APAR selama %lu ms...\n", durationMs);
  aparServo.write(SERVO_PRESS_ANGLE);
  isSpraying = true;
  sprayStartTime = millis();
  sprayDurationMs = durationMs;
  publishStatus("SPRAYING", "Menyemprotkan APAR");
}

void releaseServo() {
  aparServo.write(SERVO_IDLE_ANGLE);
  isSpraying = false;
  Serial.println("[APAR] Tuas APAR dilepaskan kembali ke posisi aman.");
}

void publishStatus(const char* state, const char* msg) {
  if (!mqttClient.connected()) return;

  char payload[160];
  snprintf(payload, sizeof(payload),
           "{\"node\":\"apar\",\"state\":\"%s\",\"seq\":%ld,\"msg\":\"%s\"}",
           state, activeSeq, msg);
  mqttClient.publish(TOPIC_STATUS, payload);
  Serial.print("[MQTT OUT APAR] ");
  Serial.println(payload);
}

void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;

  if (millis() - lastMqttReconnectAttempt > 5000) {
    lastMqttReconnectAttempt = millis();
    String clientId = "Deezpro-ESP32-APAR-" + String((uint32_t)ESP.getEfuseMac(), HEX);

    if (mqttClient.connect(clientId.c_str())) {
      mqttClient.subscribe(TOPIC_CMD);
      mqttClient.subscribe(TOPIC_SYS_CMD);
      publishStatus("READY", "ESP32 APAR Servo Online");
      Serial.println("[MQTT APAR] Terhubung ke Broker!");
    }
  }
}

void loop() {
  // 1. Timer Non-blocking Pemantauan Durasi Semprot
  if (isSpraying) {
    if (millis() - sprayStartTime >= sprayDurationMs) {
      releaseServo();
      publishStatus("COMPLETED", "Penyemprotan Selesai");
    }
  }

  // 2. Loop MQTT Client
  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop();
  }

  // 3. Fallback Serial Console
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    handleJsonCommand(input);
  }
}
