# Setup Guide: ESP32 Stepper Controllers & Python MQTT Transmitter

This guide walks you through setting up a brand-new device (PC/Laptop and ESP32 microcontrollers) to run the **`esp32-src`** firmware and the **`Python Transmitter`** control software.

---

## Architecture Overview

```
+-------------------------------------------------------------+
|                     HiveMQ Public Broker                    |
|             (broker.hivemq.com : 1883 TCP / 8000 WS)         |
+-------------------------------------------------------------+
        ^                              ^               |
        | Commands                     | Commands      | Status Updates
        |                              |               v
+-----------------------+     +-----------------------+   +-----------------------------+
|       ESP32 #1        |     |       ESP32 #2        |   |     Python Transmitter      |
|  (stepper-test.ino)   |     | (stepper-test-2.ino)  |   |          (main.py)          |
|                       |     |                       |   |                             |
| Cmd:    .../1/cmd     |     | Cmd:    .../2/cmd     |   | Controls #1, #2, or BOTH;   |
| Status: .../1/status  |     | Status: .../2/status  |   | Receives real-time telemetry|
+-----------------------+     +-----------------------+   +-----------------------------+
```

---

## 1. Prerequisites & Software Requirements

Install the following software on your new computer:

1. **Python 3.10+**:
   - Download from [python.org](https://www.python.org/downloads/).
   - ⚠️ **Important**: During installation on Windows, check the box **"Add python.exe to PATH"**.
2. **Git**:
   - Download from [git-scm.com](https://git-scm.com/).
3. **Arduino IDE (v2.x recommended)**:
   - Download from [arduino.cc/en/software](https://www.arduino.cc/en/software).
4. **USB-to-UART Drivers** (required for ESP32 COM port detection):
   - **CP210x Driver**: [Silicon Labs CP210x USB to UART Bridge VCP Drivers](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
   - **CH340/CH341 Driver**: [WCH CH340 Driver](https://www.wch-ic.com/downloads/CH341SER_EXE.html) (common on clone ESP32 boards)

---

## 2. Setting Up ESP32 Firmware (`esp32-src`)

### 2.1 Arduino IDE Configuration

1. Open **Arduino IDE**.
2. Add ESP32 Board Manager URL:
   - Go to **File** > **Preferences**.
   - In **Additional boards manager URLs**, add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Click **OK**.
3. Install ESP32 Board Package:
   - Go to **Tools** > **Board** > **Boards Manager...** (or click the Board icon on the left sidebar).
   - Search for `esp32` by **Espressif Systems** and click **Install**.

### 2.2 Install Required Arduino Libraries

Go to **Tools** > **Manage Libraries...** (or click the Library icon on the left sidebar) and install:
1. **PubSubClient** (by Nick O'Leary)
2. **AccelStepper** (by Mike McCauley)

### 2.3 Hardware Wiring Reference

For each ESP32 board connected to a stepper driver (e.g. A4988, DRV8825, TMC2208):

| Pin Name | ESP32 GPIO | Driver Pin | Description |
|---|---|---|---|
| **STEP** | `GPIO 14` | `STEP` | Pulse signal for motor steps |
| **DIR** | `GPIO 12` | `DIR` | Direction control |
| **ENABLE** | `GPIO 21` | `EN` / `ENABLE` | Active LOW enable signal |
| **GND** | `GND` | `GND` | Common ground (connect to driver logic GND & power supply GND) |
| **VMOT** | External PSU | `VMOT` & `GND` | 12V-24V motor power (never power motors from ESP32 5V/3.3V) |

### 2.4 Configure & Flash the ESP32 Controllers

#### Controller #1 (`stepper-test.ino`):
1. Open [`Software/Development/esp32-src/stepper-test.ino`](file:///d:/Tugas/Desain%20Proyek/Deezpro/Software/Development/esp32-src/stepper-test.ino).
2. Update the Wi-Fi credentials around line 7:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   ```
   *(Note: ESP32 only supports 2.4 GHz Wi-Fi networks).*
3. Connect ESP32 #1 via USB.
4. Select **Tools** > **Board** > **esp32** > **ESP32 Dev Module**.
5. Select the matching **Port** (e.g., `COM3`).
6. Click **Upload** (Arrow icon).
7. Open **Tools** > **Serial Monitor** at **115200 baud** to confirm it connects to Wi-Fi and the HiveMQ broker.

#### Controller #2 (`stepper-test-2.ino`):
1. Open [`Software/Development/esp32-src/stepper-test-2.ino`](file:///d:/Tugas/Desain%20Proyek/Deezpro/Software/Development/esp32-src/stepper-test-2.ino).
2. Update `WIFI_SSID` and `WIFI_PASSWORD` as above.
3. Connect ESP32 #2 via USB, select its COM port, and click **Upload**.

---

## 3. Setting Up Python Transmitter (`Python Transmitter`)

### 3.1 Clone or Open the Project
Open a terminal (PowerShell, Command Prompt, or VS Code terminal) in the project root directory:
```powershell
cd "d:\Tugas\Desain Proyek\Deezpro"
```

### 3.2 Create & Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Allow script execution if restricted (only needed once)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Dependencies
Install all required packages using the requirements file:
```powershell
pip install -r "Software/Development/Python Transmitter/requirements.txt"
```
*(This installs `paho-mqtt >= 2.0.0`)*

### 3.4 Run the Transmitter Console
Start the interactive transmitter:
```powershell
python "Software/Development/Python Transmitter/main.py"
```

> **Network Note:** The script automatically probes port 1883. If your network (such as a university or corporate firewall) blocks MQTT TCP port 1883, the transmitter automatically falls back to WebSockets over port 8000. You can also force WebSockets with:
> ```powershell
> python "Software/Development/Python Transmitter/main.py" --ws
> ```

---

## 4. Quick Command Reference

Once `main.py` is running, enter commands into the console prompt:

### Targeting & Multi-Motor Control
| Command | Behavior | Example |
|---|---|---|
| `<cmd>` | Send to current target mode (default: `BOTH`) | `M 200` |
| `1: <cmd>` | Send to ESP32 #1 only | `1: M 200` |
| `2: <cmd>` | Send to ESP32 #2 only | `2: M -200` |
| `BOTH: <cmd>` | Send to both ESP32 controllers | `BOTH: M 400` |
| `1: <cmd1> \| 2: <cmd2>` | Send simultaneous split commands | `1: M 200 \| 2: M -200` |
| `TARGET <1\|2\|BOTH>` | Switch default target mode | `TARGET 1` |

### Motion & Configuration Commands
| Command | Parameter / Syntax | Description |
|---|---|---|
| `M` | `M <steps>` | Relative step move (`M 200`, `M -400`) |
| `G` | `G <position>` | Move to absolute coordinate (`G 1000`, `G 0`) |
| `S` | `S <speed>` | Set max speed in steps/sec (`S 800`) |
| `A` | `A <accel>` | Set acceleration in steps/sec² (`A 400`) |
| `E` | `E <1/0>` | Enable (`E 1`) or Disable (`E 0`) driver coils |
| `STOP` | — | Decelerate to immediate halt |
| `STATUS` | — | Query current position and state from ESP32 |
| `HELP` / `?` | — | Display help menu |
| `EXIT` / `QUIT` | — | Exit transmitter |

---

## 5. Troubleshooting Checklist

| Issue | Cause | Fix |
|---|---|---|
| **`Cannot find module paho.mqtt.client`** | Python interpreter not using `.venv` or package not installed | Activate `.venv` (`.\.venv\Scripts\Activate.ps1`) and run `pip install -r "Software/Development/Python Transmitter/requirements.txt"`. Select `.venv` as interpreter in VS Code (`Ctrl+Shift+P` > `Python: Select Interpreter`). |
| **ESP32 COM Port not detected** | Missing USB driver or charge-only USB cable | Install CP210x or CH340 drivers (see Section 1). Ensure the USB cable supports data transfer. |
| **ESP32 stuck connecting to Wi-Fi** | 5 GHz network or wrong credentials | Verify `WIFI_SSID` and `WIFI_PASSWORD`. Ensure Wi-Fi is 2.4 GHz (ESP32 does not support 5 GHz). |
| **MQTT connect timeout / refused** | Port 1883 blocked by router or firewall | Run transmitter with WebSockets: `python "Software/Development/Python Transmitter/main.py" --ws`. |
| **Motor buzzes / vibrates without turning** | Phase wire mismatch or insufficient driver current | 1. Check A+/A- and B+/B- coil pairings with a multimeter.<br>2. Adjust the driver current potentiometer (VREF).<br>3. Lower speed with `S 400` and acceleration with `A 200`. |
| **Driver overheating / motor turns freely** | Driver disabled | Send `E 1` to enable driver outputs. Ensure external 12V-24V power supply is on. |
