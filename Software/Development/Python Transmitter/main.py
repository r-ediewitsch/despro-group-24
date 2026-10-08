"""
MQTT Stepper Motor Transmitter (Python Console) - Dual Controller Support
==========================================================================
Receivers:
  1) ESP32 #1 -> Topic: my_stepper_esp32/1/cmd (Status: my_stepper_esp32/1/status)
  2) ESP32 #2 -> Topic: my_stepper_esp32/2/cmd (Status: my_stepper_esp32/2/status)

This script provides an interactive Python console to transmit motion and
configuration commands to two ESP32 stepper motor controllers simultaneously or
individually over MQTT (HiveMQ public broker).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import re
import socket
import sys
import threading
import uuid
import paho.mqtt.client as mqtt

# --- Default Broker & Topic Configurations ---
DEFAULT_BROKER = "broker.hivemq.com"
DEFAULT_TCP_PORT = 1883
DEFAULT_WS_PORT = 8000

DEFAULT_TOPIC_CMD_1 = "my_stepper_esp32/1/cmd"
DEFAULT_TOPIC_STATUS_1 = "my_stepper_esp32/1/status"
DEFAULT_TOPIC_CMD_2 = "my_stepper_esp32/2/cmd"
DEFAULT_TOPIC_STATUS_2 = "my_stepper_esp32/2/status"

# Generate a unique client ID to prevent collisions on public broker
CLIENT_ID = f"Python-DualSender-{uuid.uuid4().hex[:8]}"

# Global connection state, target state, and sync event
is_connected = False
connect_event = threading.Event()
active_target = "BOTH"  # "BOTH", "1", or "2"


def print_help(current_target="BOTH"):
    """Print the command instructions matching the dual-motor architecture."""
    print("\n=================== Dual Stepper MQTT Transmitter ===================")
    print(f"Current Target Mode: [{current_target}] (commands without prefix go here)")
    print("\nTargeting & Multi-Motor Control:")
    print("  <cmd>                : Send to current target mode (default: BOTH simultaneously)")
    print("  1: <cmd>             : Send command to ESP32 #1 only (e.g. '1: M 200' or '1 M 200')")
    print("  2: <cmd>             : Send command to ESP32 #2 only (e.g. '2: M -200' or '2 M -200')")
    print("  BOTH: <cmd>          : Send command to BOTH ESP32 #1 and #2 simultaneously")
    print("  1: <cmd1> | 2: <cmd2>: Send different commands simultaneously (e.g. '1: M 200 | 2: M -200')")
    print("  TARGET <1|2|BOTH>    : Switch active target mode (e.g. 'TARGET 1', 'TARGET BOTH')")
    print("\nMotion & Configuration Commands:")
    print("  M <steps>            : Move relative steps (e.g., 'M 200' or 'M -400')")
    print("  G <pos>              : Go to absolute position (e.g., 'G 1000')")
    print("  S <speed>            : Set max speed in steps/sec (e.g., 'S 800')")
    print("  A <accel>            : Set acceleration in steps/sec^2 (e.g., 'A 400')")
    print("  E <1/0>              : Enable (1) or Disable (0) driver outputs")
    print("  STOP                 : Halt movement immediately")
    print("  STATUS               : Request current status from receivers")
    print("  HELP / ?             : Re-display this menu")
    print("  EXIT / QUIT          : Exit transmitter console")
    print("=====================================================================\n")


def on_connect(client, userdata, flags, rc, properties=None):
    """Callback invoked when successfully connected to the MQTT broker."""
    global is_connected
    code = rc.value if hasattr(rc, "value") else rc

    if code == 0:
        is_connected = True
        topic_status_1 = userdata.get("topic_status_1", DEFAULT_TOPIC_STATUS_1)
        topic_status_2 = userdata.get("topic_status_2", DEFAULT_TOPIC_STATUS_2)

        client.subscribe([(topic_status_1, 0), (topic_status_2, 0)])
        connect_event.set()

        print(f"\n[MQTT] Connected successfully to {client._host}:{client._port}!")
        print(f"[MQTT] Client ID   : {CLIENT_ID}")
        print(f"[MQTT] Subscribed 1: {topic_status_1}")
        print(f"[MQTT] Subscribed 2: {topic_status_2}")
        print_help(active_target)
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
    """Callback invoked when a status message is published by either receiver."""
    try:
        payload = msg.payload.decode("utf-8", errors="replace").strip()
    except Exception:
        payload = str(msg.payload)

    topic_status_1 = userdata.get("topic_status_1", DEFAULT_TOPIC_STATUS_1)
    topic_status_2 = userdata.get("topic_status_2", DEFAULT_TOPIC_STATUS_2)

    if msg.topic == topic_status_1:
        source = "ESP32 #1"
    elif msg.topic == topic_status_2:
        source = "ESP32 #2"
    else:
        source = msg.topic

    sys.stdout.write(f"\r[STATUS {source}] {payload}\nCommand [{active_target}] > ")
    sys.stdout.flush()


def check_tcp_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quickly check if a given TCP port is reachable."""
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


def publish_single(client: mqtt.Client, topic: str, command: str, label: str):
    """Publish a command to an MQTT topic and log result."""
    result = client.publish(topic, command)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"[TX -> {topic} ({label})] {command}")
    else:
        print(f"[ERROR] Failed to publish to {topic} (rc={result.rc})")


def publish_simultaneous(client: mqtt.Client, tasks: list):
    """
    Publish multiple commands simultaneously using a ThreadPoolExecutor.
    tasks: list of tuples: (topic, command, label)
    """
    with ThreadPoolExecutor(max_workers=max(len(tasks), 2)) as executor:
        futures = [
            executor.submit(publish_single, client, topic, cmd, label)
            for (topic, cmd, label) in tasks
        ]
        for f in futures:
            f.result()


def parse_and_dispatch(client: mqtt.Client, userdata: dict, raw_input: str):
    """Parse user command and dispatch to respective MQTT topic(s) simultaneously."""
    global active_target

    raw = raw_input.strip()
    if not raw:
        return

    upper = raw.upper()

    # Target mode switching
    if upper.startswith("TARGET") or upper.startswith("MODE"):
        parts = upper.split()
        if len(parts) >= 2 and parts[1] in ("1", "2", "BOTH", "ALL"):
            active_target = "BOTH" if parts[1] in ("BOTH", "ALL") else parts[1]
            print(f"[TARGET] Active mode switched to: {active_target}")
        else:
            print("[TARGET] Usage: TARGET <1 | 2 | BOTH>")
        return

    # Check for combined simultaneous command (e.g., "1: M 200 | 2: M -200" or separated by ";")
    if "|" in raw or ";" in raw:
        delimiter = "|" if "|" in raw else ";"
        parts = [p.strip() for p in raw.split(delimiter) if p.strip()]
        tasks = []
        for part in parts:
            match = re.match(r"^([12]|BOTH|ALL)\s*[:\s]\s*(.+)$", part, re.IGNORECASE)
            if match:
                tgt = match.group(1).upper()
                c = match.group(2).strip()
                if tgt == "1":
                    tasks.append((userdata["topic_cmd_1"], c, "ESP32 #1"))
                elif tgt == "2":
                    tasks.append((userdata["topic_cmd_2"], c, "ESP32 #2"))
                else:
                    tasks.append((userdata["topic_cmd_1"], c, "ESP32 #1"))
                    tasks.append((userdata["topic_cmd_2"], c, "ESP32 #2"))
            else:
                print(f"[WARN] Ignored ambiguous segment '{part}'. Use '1: <cmd>' or '2: <cmd>'.")

        if tasks:
            publish_simultaneous(client, tasks)
        return

    # Check for single-command prefix (e.g., "1: M 200", "2: G 100", "BOTH: STOP")
    prefix_match = re.match(r"^([12]|BOTH|ALL)\s*[:\s]\s*(.+)$", raw, re.IGNORECASE)
    if prefix_match:
        tgt = prefix_match.group(1).upper()
        cmd_body = prefix_match.group(2).strip()
        if tgt == "1":
            publish_single(client, userdata["topic_cmd_1"], cmd_body, "ESP32 #1")
        elif tgt == "2":
            publish_single(client, userdata["topic_cmd_2"], cmd_body, "ESP32 #2")
        else:
            # BOTH / ALL
            publish_simultaneous(client, [
                (userdata["topic_cmd_1"], cmd_body, "ESP32 #1"),
                (userdata["topic_cmd_2"], cmd_body, "ESP32 #2"),
            ])
        return

    # Direct command without prefix -> dispatch based on active_target
    if active_target == "1":
        publish_single(client, userdata["topic_cmd_1"], raw, "ESP32 #1")
    elif active_target == "2":
        publish_single(client, userdata["topic_cmd_2"], raw, "ESP32 #2")
    else:
        # Default: BOTH simultaneously
        publish_simultaneous(client, [
            (userdata["topic_cmd_1"], raw, "ESP32 #1"),
            (userdata["topic_cmd_2"], raw, "ESP32 #2"),
        ])


def main():
    global active_target

    parser = argparse.ArgumentParser(description="MQTT Dual Stepper Motor Transmitter Console")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help=f"MQTT broker address (default: {DEFAULT_BROKER})")
    parser.add_argument("--port", type=int, default=None, help="Port (defaults to 1883 for TCP or 8000 for WebSockets)")
    parser.add_argument("--topic-cmd-1", default=DEFAULT_TOPIC_CMD_1, help=f"ESP32 #1 command topic (default: {DEFAULT_TOPIC_CMD_1})")
    parser.add_argument("--topic-status-1", default=DEFAULT_TOPIC_STATUS_1, help=f"ESP32 #1 status topic (default: {DEFAULT_TOPIC_STATUS_1})")
    parser.add_argument("--topic-cmd-2", default=DEFAULT_TOPIC_CMD_2, help=f"ESP32 #2 command topic (default: {DEFAULT_TOPIC_CMD_2})")
    parser.add_argument("--topic-status-2", default=DEFAULT_TOPIC_STATUS_2, help=f"ESP32 #2 status topic (default: {DEFAULT_TOPIC_STATUS_2})")
    parser.add_argument("--target", default="BOTH", choices=["BOTH", "1", "2"], help="Default command target (default: BOTH)")
    parser.add_argument("--ws", action="store_true", help="Force MQTT over WebSockets (port 8000)")
    parser.add_argument("--tcp", action="store_true", help="Force standard MQTT TCP (port 1883)")
    args = parser.parse_args()

    active_target = args.target.upper()
    broker = args.broker

    userdata = {
        "topic_cmd_1": args.topic_cmd_1,
        "topic_status_1": args.topic_status_1,
        "topic_cmd_2": args.topic_cmd_2,
        "topic_status_2": args.topic_status_2,
    }

    # Determine transport and port
    if args.ws:
        transport = "websockets"
        port = args.port or DEFAULT_WS_PORT
        print(f"[INIT] Mode: WebSockets on port {port}")
    elif args.tcp:
        transport = "tcp"
        port = args.port or DEFAULT_TCP_PORT
        print(f"[INIT] Mode: Standard TCP on port {port}")
    else:
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
                f"[INIT] Automatically switching to WebSockets on port {port}."
            )

    client = build_client(transport=transport, userdata=userdata)

    print(f"[MQTT] Connecting to {broker}:{port}...")
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"[ERROR] Connection attempt failed: {e}")
        sys.exit(1)

    client.loop_start()

    if not connect_event.wait(timeout=15.0):
        print("[WARNING] Connection handshake is taking longer than expected. Continuing...")

    # Interactive console command loop
    try:
        while True:
            try:
                raw_input_line = input(f"Command [{active_target}] > ")
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
                print_help(active_target)
                continue

            if not is_connected:
                print("[ERROR] Cannot send command: MQTT broker not connected.")
                continue

            parse_and_dispatch(client, userdata, cmd)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[INFO] Disconnected. Done.")


if __name__ == "__main__":
    main()
