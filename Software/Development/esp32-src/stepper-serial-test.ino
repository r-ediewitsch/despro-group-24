#include <Arduino.h>
#include <AccelStepper.h>

// =========================================================================
// ESP32 Stepper Controller - Serial Test (Tanpa Wi-Fi / MQTT)
// =========================================================================
// Kontrol motor stepper langsung melalui Serial Monitor (Baud rate: 115200)
// Format Line Ending di Serial Monitor: "Newline" atau "Both NL & CR"
// =========================================================================

// --- Konfigurasi Pin Hardware ---
#define STEP_PIN   14
#define DIR_PIN    12
#define ENABLE_PIN 21

// Inisialisasi AccelStepper (Driver type 1 = DRIVER with Step & Direction pins)
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Variabel status
bool wasRunning = false;
String inputBuffer = "";

void printHelp() {
  Serial.println();
  Serial.println("=====================================================");
  Serial.println("   ESP32 STEPPER SERIAL CONTROLLER - TEST CONSOLE   ");
  Serial.println("=====================================================");
  Serial.println("Perintah yang tersedia (Ketik di Serial Monitor):");
  Serial.println("  M <steps>   : Gerak relatif (cth: 'M 400', 'M -200')");
  Serial.println("  G <pos>     : Gerak ke posisi absolut (cth: 'G 1000', 'G 0')");
  Serial.println("  S <speed>   : Ubah kecepatan maks (steps/s, cth: 'S 800')");
  Serial.println("  A <accel>   : Ubah akselerasi (steps/s^2, cth: 'A 500')");
  Serial.println("  E <1|0>     : Enable (1) atau Disable (0) driver");
  Serial.println("  STOP        : Hentikan motor seketika");
  Serial.println("  ZERO        : Reset posisi saat ini menjadi 0");
  Serial.println("  STATUS      : Tampilkan posisi & status motor");
  Serial.println("  HELP / ?    : Tampilkan menu bantuan ini");
  Serial.println("=====================================================");
  Serial.print("ESP32 Ready > ");
}

void printStatus() {
  Serial.printf("[STATUS] Pos: %ld | Target: %ld | Speed: %.1f | Running: %s\n",
                stepper.currentPosition(),
                stepper.targetPosition(),
                stepper.speed(),
                stepper.isRunning() ? "YES" : "NO");
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = toupper(cmd.charAt(0));
  String arg = cmd.substring(1);
  arg.trim();

  switch (type) {
    case 'M': { // Relative move
      long steps = arg.toInt();
      stepper.move(steps);
      Serial.printf("[MOVE] Gerak relatif: %ld steps (Target absolut: %ld)\n", steps, stepper.targetPosition());
      break;
    }
    case 'G': { // Absolute move
      long target = arg.toInt();
      stepper.moveTo(target);
      Serial.printf("[GOTO] Gerak ke posisi absolut: %ld\n", target);
      break;
    }
    case 'S': { // Set max speed
      float spd = arg.toFloat();
      if (spd > 0) {
        stepper.setMaxSpeed(spd);
        Serial.printf("[SPEED] Kecepatan maks disetel ke: %.1f steps/s\n", spd);
      } else {
        Serial.println("[ERROR] Kecepatan harus > 0 (cth: S 800)");
      }
      break;
    }
    case 'A': { // Set acceleration
      float acc = arg.toFloat();
      if (acc > 0) {
        stepper.setAcceleration(acc);
        Serial.printf("[ACCEL] Akselerasi disetel ke: %.1f steps/s^2\n", acc);
      } else {
        Serial.println("[ERROR] Akselerasi harus > 0 (cth: A 400)");
      }
      break;
    }
    case 'E': { // Enable / Disable driver
      int en = arg.toInt();
      // Driver A4988/TMC2208/DRV8825 umumnya Active LOW
      digitalWrite(ENABLE_PIN, en == 1 ? LOW : HIGH);
      Serial.printf("[ENABLE] Driver %s\n", en == 1 ? "ENABLED (Aktif)" : "DISABLED (Mati)");
      break;
    }
    default: {
      if (cmd.equalsIgnoreCase("STOP")) {
        stepper.stop();
        Serial.println("[HALT] Motor dihentikan!");
      } else if (cmd.equalsIgnoreCase("STATUS")) {
        printStatus();
      } else if (cmd.equalsIgnoreCase("ZERO")) {
        stepper.setCurrentPosition(0);
        Serial.println("[ZERO] Posisi di-reset ke 0.");
      } else if (cmd.equalsIgnoreCase("HELP") || cmd.equals("?")) {
        printHelp();
        return;
      } else {
        Serial.printf("[ERROR] Perintah tidak dikenal: '%s'. Ketik 'HELP' untuk panduan.\n", cmd.c_str());
      }
      break;
    }
  }

  Serial.print("ESP32 Ready > ");
}

void setup() {
  Serial.begin(115200);
  delay(1000); // Tunggu serial monitor terhubung

  // Konfigurasi pin ENABLE (Active LOW)
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); // Aktifkan driver

  // Konfigurasi awal AccelStepper
  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(500.0);
  stepper.setCurrentPosition(0);

  printHelp();
}

void loop() {
  // Wajib dipanggil sesering mungkin agar motor berputar halus
  stepper.run();

  // Deteksi ketika motor selesai bergerak
  if (wasRunning && !stepper.isRunning()) {
    Serial.printf("\n[DONE] Target tercapai di step %ld\nESP32 Ready > ", stepper.currentPosition());
  }
  wasRunning = stepper.isRunning();

  // Pembacaan Serial non-blocking
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        Serial.println(inputBuffer); // Echo input yang dikirim user
        handleCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}
