/**
 * =====================================================================
 * DEEZPRO FIRMWARE - ESP32 #2: KENDALI MOTOR STEPPER SUMBU Y
 * =====================================================================
 * File: esp32_stepper_y.ino
 * Deskripsi: Mengendalikan pergerakan linear Sumbu Y menggunakan motor
 *            stepper, membaca limit switch untuk auto-homing, serta
 *            berkomunikasi via MQTT JSON standar.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// --- Konfigurasi Jaringan & MQTT Broker ---
const char* WIFI_SSID     = "iQOO Z9 5G";
const char* WIFI_PASSWORD = "nggaktau";
const char* MQTT_BROKER   = "broker.emqx.io";
const int   MQTT_PORT     = 1883;

// --- Definisi Topik MQTT Sumbu Y ---
const char* TOPIC_CMD     = "deezpro/rail/y/cmd";
const char* TOPIC_STATUS  = "deezpro/rail/y/status";
const char* TOPIC_SYS_CMD = "deezpro/system/cmd";

// --- Pinout Hardware ---
#define STEP_PIN   14
#define DIR_PIN    12
#define ENABLE_PIN 21
#define LIMIT_PIN  27  // Limit switch pin (Active LOW dengan internal pull-up)

// --- Inisialisasi Objek ---
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// --- Variabel Status & Homing ---
unsigned long lastMqttReconnectAttempt = 0;
bool wasRunning = false;
long activeSeq = 0;
bool isHoming = false;

// Forward declarations
void handleJsonCommand(String jsonStr);
void mqttCallback(char* topic, byte* payload, unsigned int length);
void reconnectMQTT();
void publishStatus(const char* state, long pos, long target = 0, const char* msg = "");
void startHoming();

void setup() {
  Serial.begin(115200);

  // Setup Enable & Limit Switch Pin
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); // Active LOW -> Enable driver

  pinMode(LIMIT_PIN, INPUT_PULLUP);

  // Default Motor Profile
  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(500.0);

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

// Ekstraksi nilai string dari JSON sederhana tanpa dependensi library eksternal
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
  Serial.print("[MQTT IN] ");
  Serial.println(message);

  handleJsonCommand(message);
}

void handleJsonCommand(String input) {
  input.trim();
  if (input.length() == 0) return;

  String cmd = "";
  long target = 0;
  long steps = 0;
  float speed = 0;
  float accel = 0;
  activeSeq = 0;

  // Format 1: JSON Standard ({"cmd": "GOTO", "target": 1850, "seq": 104})
  if (input.startsWith("{")) {
    cmd = extractJsonField(input, "cmd");
    cmd.toUpperCase();

    String seqStr = extractJsonField(input, "seq");
    if (seqStr.length() > 0) activeSeq = seqStr.toInt();

    String targetStr = extractJsonField(input, "target");
    if (targetStr.length() > 0) target = targetStr.toInt();

    String stepsStr = extractJsonField(input, "steps");
    if (stepsStr.length() > 0) steps = stepsStr.toInt();

    String spdStr = extractJsonField(input, "speed");
    if (spdStr.length() > 0) speed = spdStr.toFloat();

    String accStr = extractJsonField(input, "accel");
    if (accStr.length() > 0) accel = accStr.toFloat();
  } else {
    // Format 2: Shorthand Legacy (G 1850, M 200, HOME, STOP)
    char type = toupper(input.charAt(0));
    String arg = input.substring(1);
    arg.trim();

    if (type == 'G') {
      cmd = "GOTO";
      target = arg.toInt();
    } else if (type == 'M') {
      cmd = "MOVE";
      steps = arg.toInt();
    } else if (input.equalsIgnoreCase("HOME")) {
      cmd = "HOME";
    } else if (input.equalsIgnoreCase("STOP") || input.equalsIgnoreCase("HALT")) {
      cmd = "STOP";
    } else if (input.equalsIgnoreCase("STATUS")) {
      cmd = "STATUS";
    }
  }

  // Update profil kecepatan & akselerasi jika disertakan
  if (speed > 0) stepper.setMaxSpeed(speed);
  if (accel > 0) stepper.setAcceleration(accel);

  // Eksekusi Perintah
  if (cmd == "GOTO") {
    isHoming = false;
    stepper.moveTo(target);
    publishStatus("MOVING", stepper.currentPosition(), target);
  } else if (cmd == "MOVE") {
    isHoming = false;
    stepper.move(steps);
    publishStatus("MOVING", stepper.currentPosition(), stepper.targetPosition());
  } else if (cmd == "HOME") {
    startHoming();
  } else if (cmd == "STOP" || cmd == "EMERGENCY_STOP") {
    isHoming = false;
    stepper.stop();
    publishStatus("DONE", stepper.currentPosition(), stepper.currentPosition(), "Stopped");
  } else if (cmd == "SET_ZERO") {
    stepper.setCurrentPosition(0);
    publishStatus("HOMED", 0, 0, "Zero reset");
  } else if (cmd == "STATUS") {
    const char* st = stepper.isRunning() ? "MOVING" : "IDLE";
    publishStatus(st, stepper.currentPosition(), stepper.targetPosition());
  }
}

void startHoming() {
  Serial.println("[HOMING] Memulai prosedur auto-homing Sumbu Y...");
  isHoming = true;
  stepper.setMaxSpeed(400.0);
  stepper.setAcceleration(300.0);
  stepper.moveTo(-100000); // Gerak mundur menuju limit switch
  publishStatus("HOMING", stepper.currentPosition(), 0);
}

void publishStatus(const char* state, long pos, long target, const char* msg) {
  if (!mqttClient.connected()) return;

  char payload[160];
  if (strlen(msg) > 0) {
    snprintf(payload, sizeof(payload),
             "{\"node\":\"y\",\"state\":\"%s\",\"pos\":%ld,\"target\":%ld,\"seq\":%ld,\"msg\":\"%s\"}",
             state, pos, target, activeSeq, msg);
  } else {
    snprintf(payload, sizeof(payload),
             "{\"node\":\"y\",\"state\":\"%s\",\"pos\":%ld,\"target\":%ld,\"seq\":%ld}",
             state, pos, target, activeSeq);
  }
  mqttClient.publish(TOPIC_STATUS, payload);
  Serial.print("[MQTT OUT] ");
  Serial.println(payload);
}

void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;

  if (millis() - lastMqttReconnectAttempt > 5000) {
    lastMqttReconnectAttempt = millis();
    String clientId = "Deezpro-ESP32-Y-" + String((uint32_t)ESP.getEfuseMac(), HEX);

    if (mqttClient.connect(clientId.c_str())) {
      mqttClient.subscribe(TOPIC_CMD);
      mqttClient.subscribe(TOPIC_SYS_CMD);
      publishStatus("IDLE", stepper.currentPosition(), stepper.currentPosition(), "ESP32 Sumbu Y Online");
      Serial.println("[MQTT] Terhubung ke Broker!");
    }
  }
}

void loop() {
  // 1. Eksekusi Homing jika aktif
  if (isHoming) {
    // Limit switch membaca LOW saat tertekan (PULLUP)
    if (digitalRead(LIMIT_PIN) == LOW) {
      stepper.stop();
      stepper.setCurrentPosition(0);
      stepper.setMaxSpeed(1000.0);
      stepper.setAcceleration(500.0);
      isHoming = false;
      Serial.println("[HOMING] Sumbu Y berhasil menemukan titik asal (HOME 0)!");
      publishStatus("HOMED", 0, 0, "Homing Complete");
    } else {
      stepper.run();
    }
  } else {
    // 2. Eksekusi Gerakan Stepper Normal
    stepper.run();
  }

  // 3. Pengelolaan Koneksi MQTT
  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop();
  }

  // 4. Deteksi Selesai Gerak (Transisi dari Running ke Berhenti)
  if (wasRunning && !stepper.isRunning() && !isHoming) {
    publishStatus("DONE", stepper.currentPosition(), stepper.currentPosition());
  }
  wasRunning = stepper.isRunning();

  // 5. Input Serial Konsol Lokal (Opsional untuk Debug)
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    handleJsonCommand(input);
  }
}
