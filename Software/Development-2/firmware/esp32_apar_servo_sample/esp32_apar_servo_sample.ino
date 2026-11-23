#include <ESP32Servo.h>

Servo myServo;

// Safe PWM pin (avoids boot strapping pins 0, 2, 12, 15)
const int servoPin = 13;

// Conservative pulse limits (in microseconds) to prevent hitting mechanical stops
const int MIN_SAFE_US = 500;
const int MAX_SAFE_US = 2500;

// Speed control: lower value = faster movement (was 20ms)
const int STEP_DELAY_MS = 500; 

void setup() {
  Serial.begin(115200);
  delay(500); // Allow serial monitor and power rails to stabilize

  // Allocate hardware timers for ESP32 PWM[cite: 1]
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, MIN_SAFE_US, MAX_SAFE_US);

  Serial.println("1. Moving to starting position (45 deg)...");
  myServo.write(45);
  delay(800);

  // Detach PWM signal so the servo relaxes and draws near-zero idle current[cite: 1]
}

void loop() {
  // Intentionally empty: runs one safe cycle per reset[cite: 1]

  myServo.write(135);
  delay(2000);

  myServo.write(45);
  delay(5000);
}