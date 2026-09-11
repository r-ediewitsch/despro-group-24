"""
MQTT Stepper Motor Transmitter (Python Console)
==============================================
Converted from: test_motor_31_aug_2026.ino
Receiver:       receiver.ino (ESP32 Stepper Controller)

This script replaces the ESP32 transmitter sketch with a native Python console.
It connects to the HiveMQ public broker, subscribes to status feedback from the
stepper motor receiver, and allows you to transmit motion and configuration
commands directly from your keyboard.
"""

import argparse
import socket
import sys
import threading
import uuid
import paho.mqtt.client as mqtt

# --- Default Broker & Topic Configurations ---
DEFAULT_BROKER = "broker.hivemq.com"
DEFAULT_TCP_PORT = 1883
DEFAULT_WS_PORT = 8000
DEFAULT_TOPIC_CMD = "my_stepper_esp32/cmd"
DEFAULT_TOPIC_STATUS = "my_stepper_esp32/status"

# Generate a unique client ID to prevent collisions on the public broker
CLIENT_ID = f"Python-Sender-{uuid.uuid4().hex[:8]}"

# Global connection state and sync event
is_connected = False
connect_event = threading.Event()


def print_help():
    """Print the command instructions matching the Arduino transmitter."""
    print("\n--- Stepper MQTT Transmitter ---")
    print("Enter commands to transmit via MQTT:")
    print("  M <steps>   : Move relative steps (e.g., 'M 200' or 'M -400')")
    print("  G <pos>     : Go to absolute position (e.g., 'G 1000')")
    print("  S <speed>   : Set max speed in steps/sec (e.g., 'S 800')")
    print("  A <accel>   : Set acceleration in steps/sec^2 (e.g., 'A 400')")
    print("  E <1/0>     : Enable (1) or Disable (0) driver outputs")
    print("  STOP        : Halt movement immediately")
    print("  STATUS      : Request current status from receiver")
    print("  HELP / ?    : Re-display this menu")
    print("  EXIT / QUIT : Exit transmitter console")
    print("--------------------------------\n")


def on_connect(client, userdata, flags, rc, properties=None):
    """Callback invoked when successfully connected to the MQTT broker."""
    global is_connected
    # Support both paho-mqtt v1 (int code) and v2 (ReasonCode object)
    code = rc.value if hasattr(rc, "value") else rc

    if code == 0:
        is_connected = True
        topic_status = userdata.get("topic_status", DEFAULT_TOPIC_STATUS)
        client.subscribe(topic_status)
        connect_event.set()
        print(f"\n[MQTT] Connected successfully to {client._host}:{client._port}!")
        print(f"[MQTT] Client ID : {CLIENT_ID}")
        print(f"[MQTT] Subscribed: {topic_status}")
        print_help()
    else:
        is_connected = False
        connect_event.set()
        print(f"\n[MQTT] Connection failed with status code: {rc}")


def on_disconnect(client, userdata, *args):
    """Callback invoked when disconnected from the broker."""
    global is_connected
    is_connected = False
    print("\n[MQTT] Disconnected from broker.")


def on_message(client, userdata, msg):
    """Callback invoked when a status message is published by the receiver."""
    try:
        payload = msg.payload.decode("utf-8", errors="replace").strip()
    except Exception:
        payload = str(msg.payload)

    # Print incoming feedback asynchronously without breaking the active prompt
    sys.stdout.write(f"\r[STATUS FEEDBACK] {payload}\nCommand > ")
    sys.stdout.flush()


def check_tcp_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quickly check if a given TCP port is reachable (avoids long hanging timeouts on firewalled networks)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, OSError):
        return False


def build_client(transport: str, userdata: dict) -> mqtt.Client:
    """Instantiate a Paho MQTT client supporting both v1 and v2 API versions."""
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=CLIENT_ID,
            transport=transport,
            userdata=userdata,
        )
    else:
        client = mqtt.Client(
            client_id=CLIENT_ID,
            transport=transport,
            userdata=userdata,
        )

    if transport == "websockets":
        client.ws_set_options(path="/mqtt")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    return client


def main():
    parser = argparse.ArgumentParser(description="MQTT Stepper Motor Transmitter Console")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help=f"MQTT broker address (default: {DEFAULT_BROKER})")
    parser.add_argument("--port", type=int, default=None, help="Port (defaults to 1883 for TCP or 8000 for WebSockets)")
    parser.add_argument("--topic-cmd", default=DEFAULT_TOPIC_CMD, help=f"Command topic (default: {DEFAULT_TOPIC_CMD})")
    parser.add_argument("--topic-status", default=DEFAULT_TOPIC_STATUS, help=f"Status topic (default: {DEFAULT_TOPIC_STATUS})")
    parser.add_argument("--ws", action="store_true", help="Force MQTT over WebSockets (port 8000)")
    parser.add_argument("--tcp", action="store_true", help="Force standard MQTT TCP (port 1883)")
    args = parser.parse_args()

    broker = args.broker
    topic_cmd = args.topic_cmd
    topic_status = args.topic_status
    userdata = {"topic_cmd": topic_cmd, "topic_status": topic_status}

    # Determine transport and port:
    # If network blocks port 1883 (common on enterprise/campus Wi-Fi), auto-detect and switch to WebSockets
    if args.ws:
        transport = "websockets"
        port = args.port or DEFAULT_WS_PORT
        print(f"[INIT] Mode: WebSockets on port {port}")
    elif args.tcp:
        transport = "tcp"
        port = args.port or DEFAULT_TCP_PORT
        print(f"[INIT] Mode: Standard TCP on port {port}")
    else:
        # Smart auto-detection: check if TCP 1883 is reachable
        print(f"[INIT] Testing broker reachability ({broker}:{DEFAULT_TCP_PORT})...")
        if check_tcp_port_open(broker, DEFAULT_TCP_PORT, timeout=2.0):
            transport = "tcp"
            port = args.port or DEFAULT_TCP_PORT
            print(f"[INIT] Standard TCP port {port} is open. Using TCP transport.")
        else:
            transport = "websockets"
            port = args.port or DEFAULT_WS_PORT
            print(
                f"[INIT] Port {DEFAULT_TCP_PORT} is unreachable/firewalled on this network.\n"
                f"[INIT] Automatically switching to WebSockets on port {port} (shares identical HiveMQ topics)."
            )

    client = build_client(transport=transport, userdata=userdata)

    print(f"[MQTT] Connecting to {broker}:{port}...")
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"[ERROR] Connection attempt failed: {e}")
        sys.exit(1)

    # Start non-blocking network thread
    client.loop_start()

    # Wait up to 15 seconds for connection handshake
    if not connect_event.wait(timeout=15.0):
        print("[WARNING] Connection handshake is taking longer than expected. Continuing...")

    # Interactive console command loop
    try:
        while True:
            try:
                raw_input_line = input("Command > ")
            except EOFError:
                break

            cmd = raw_input_line.strip()
            if not cmd:
                continue

            upper_cmd = cmd.upper()

            if upper_cmd in ("EXIT", "QUIT", "Q"):
                print("[INFO] Exiting transmitter...")
                break

            if upper_cmd in ("HELP", "?"):
                print_help()
                continue

            if not is_connected:
                print("[ERROR] Cannot send command: MQTT broker not connected.")
                continue

            # Transmit command to receiver topic
            result = client.publish(topic_cmd, cmd)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[TX -> {topic_cmd}] {cmd}")
            else:
                print(f"[ERROR] Failed to publish message (rc={result.rc})")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[INFO] Disconnected. Done.")


if __name__ == "__main__":
    main()
