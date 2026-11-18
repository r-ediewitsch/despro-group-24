"""
=====================================================================
DEEZPRO INTEGRATION - MQTT BRIDGE
=====================================================================
File: mqtt_bridge.py
Deskripsi: Jembatan komunikasi asinkronus MQTT antara Python Coordinator
            dengan 3 modul ESP32 (Sumbu X, Sumbu Y, dan APAR Servo).
            Menyediakan abstraksi perintah JSON, pelacakan status node,
            serta Synchronization Barrier untuk koordinasi gerak simultan.
"""

import json
import time
import uuid
import threading
from typing import Dict, Any, Optional, Callable, List
import paho.mqtt.client as mqtt

from .system_logger import get_logger

logger = get_logger()


class MqttBridge:
    """
    Handler MQTT Client terpadu untuk ekosistem Deezpro.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        mqtt_cfg = config.get("mqtt", {})
        self.broker = mqtt_cfg.get("broker", "broker.emqx.io")
        self.port = int(mqtt_cfg.get("port", 1883))
        self.keepalive = int(mqtt_cfg.get("keepalive", 60))
        prefix = mqtt_cfg.get("client_id_prefix", "Deezpro-Coordinator")
        self.client_id = f"{prefix}-{uuid.uuid4().hex[:6]}"

        topics = mqtt_cfg.get("topics", {})
        self.topic_rail_x_cmd = topics.get("rail_x_cmd", "deezpro/rail/x/cmd")
        self.topic_rail_x_status = topics.get("rail_x_status", "deezpro/rail/x/status")
        self.topic_rail_y_cmd = topics.get("rail_y_cmd", "deezpro/rail/y/cmd")
        self.topic_rail_y_status = topics.get("rail_y_status", "deezpro/rail/y/status")
        self.topic_apar_cmd = topics.get("apar_cmd", "deezpro/apar/cmd")
        self.topic_apar_status = topics.get("apar_status", "deezpro/apar/status")
        self.topic_sys_cmd = topics.get("system_cmd", "deezpro/system/cmd")
        self.topic_sys_status = topics.get("system_status", "deezpro/system/status")

        # Status Internal Nodes (Thread-Safe)
        self.lock = threading.RLock()
        self.nodes_status = {
            "x": {"state": "OFFLINE", "pos": 0, "target": 0, "seq": 0, "last_seen": 0.0},
            "y": {"state": "OFFLINE", "pos": 0, "target": 0, "seq": 0, "last_seen": 0.0},
            "apar": {"state": "OFFLINE", "seq": 0, "last_seen": 0.0},
        }

        # Sequence counter untuk pelacakan transaksi perintah
        self._seq_counter = 100
        self._sync_condition = threading.Condition(self.lock)

        # Callbacks terdaftar
        self._status_listeners: List[Callable[[str, Dict[str, Any]], None]] = []

        # Inisialisasi Paho MQTT Client
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        else:
            self.client = mqtt.Client(client_id=self.client_id)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.is_connected = False
        self._connect_event = threading.Event()

    def start(self, timeout_sec: float = 10.0) -> bool:
        """Menghubungkan ke broker MQTT dan memulai background network loop."""
        logger.info(f"Menghubungkan ke MQTT Broker: {self.broker}:{self.port} ...")
        try:
            self.client.connect(self.broker, self.port, keepalive=self.keepalive)
            self.client.loop_start()
            connected = self._connect_event.wait(timeout=timeout_sec)
            if not connected:
                logger.warning(f"Timeout menunggu konfirmasi koneksi MQTT ({timeout_sec}s). Tetap mencoba di latar belakang.")
            return self.is_connected
        except Exception as e:
            logger.error(f"Gagal menghubungkan ke MQTT broker: {e}")
            return False

    def stop(self):
        """Memutuskan koneksi MQTT dengan bersih."""
        logger.info("Menghentikan MQTT Bridge...")
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.warning(f"Error saat menutup MQTT: {e}")
        self.is_connected = False

    def register_status_listener(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Mendaftarkan fungsi callback saat ada pembaruan status dari node mana pun."""
        with self.lock:
            if callback not in self._status_listeners:
                self._status_listeners.append(callback)

    def _next_seq(self) -> int:
        with self.lock:
            self._seq_counter += 1
            return self._seq_counter

    # --- Callbacks MQTT Internal ---
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        code = rc.value if hasattr(rc, "value") else rc
        if code == 0:
            self.is_connected = True
            logger.info(f"Terhubung sukses ke MQTT Broker! (Client ID: {self.client_id})")

            # Subscribe ke seluruh topik status node
            topics_to_sub = [
                (self.topic_rail_x_status, 0),
                (self.topic_rail_y_status, 0),
                (self.topic_apar_status, 0),
                (self.topic_sys_status, 0),
            ]
            self.client.subscribe(topics_to_sub)
            logger.info("Berhasil subscribe ke topik status Sumbu X, Sumbu Y, dan APAR.")
            self._connect_event.set()
        else:
            self.is_connected = False
            logger.error(f"Koneksi MQTT ditolak dengan kode respon: {code}")
            self._connect_event.set()

    def _on_disconnect(self, client, userdata, *args):
        self.is_connected = False
        logger.warning("Terputus dari MQTT Broker! Mencoba reconnect otomatis...")

    def _on_message(self, client, userdata, msg):
        payload_str = msg.payload.decode("utf-8", errors="replace").strip()
        topic = msg.topic

        data = {}
        try:
            data = json.loads(payload_str)
        except Exception:
            # Fallback untuk pesan non-JSON legacy
            data = {"raw": payload_str}

        node = data.get("node", "")
        if not node:
            if topic == self.topic_rail_x_status:
                node = "x"
            elif topic == self.topic_rail_y_status:
                node = "y"
            elif topic == self.topic_apar_status:
                node = "apar"

        state = data.get("state", data.get("raw", "UNKNOWN"))

        with self.lock:
            if node in self.nodes_status:
                self.nodes_status[node]["state"] = state
                self.nodes_status[node]["last_seen"] = time.time()
                if "pos" in data:
                    self.nodes_status[node]["pos"] = int(data["pos"])
                if "target" in data:
                    self.nodes_status[node]["target"] = int(data["target"])
                if "seq" in data:
                    self.nodes_status[node]["seq"] = int(data["seq"])

            # Bangunkan thread yang sedang menunggu Barrier Synchronization
            self._sync_condition.notify_all()

        # Notifikasi listener eksternal
        for cb in self._status_listeners:
            try:
                cb(node, data)
            except Exception as e:
                logger.error(f"Error pada status listener callback: {e}")

    # --- Metode Pengiriman Perintah Baku ---
    def send_stepper_goto(
        self, node: str, target_step: int, speed: float = None, accel: float = None
    ) -> int:
        """Mengirim perintah GOTO langkah absolut ke Sumbu X atau Y."""
        seq = self._next_seq()
        topic = self.topic_rail_x_cmd if node.lower() == "x" else self.topic_rail_y_cmd
        payload = {"cmd": "GOTO", "target": int(target_step), "seq": seq}
        if speed is not None:
            payload["speed"] = float(speed)
        if accel is not None:
            payload["accel"] = float(accel)

        self._publish_json(topic, payload)
        logger.debug(f"[SEND GOTO] Node {node.upper()} -> Target {target_step} steps (seq: {seq})")
        return seq

    def send_stepper_move(self, node: str, steps: int) -> int:
        """Mengirim perintah MOVE langkah relatif ke Sumbu X atau Y."""
        seq = self._next_seq()
        topic = self.topic_rail_x_cmd if node.lower() == "x" else self.topic_rail_y_cmd
        payload = {"cmd": "MOVE", "steps": int(steps), "seq": seq}
        self._publish_json(topic, payload)
        return seq

    def send_stepper_home(self, node: str) -> int:
        """Mengirim perintah HOME ke Sumbu X atau Y."""
        seq = self._next_seq()
        topic = self.topic_rail_x_cmd if node.lower() == "x" else self.topic_rail_y_cmd
        payload = {"cmd": "HOME", "seq": seq}
        self._publish_json(topic, payload)
        logger.info(f"[SEND HOME] Memerintahkan Auto-Homing ke Sumbu {node.upper()} (seq: {seq})")
        return seq

    def send_stepper_stop(self, node: str) -> int:
        """Menghentikan pergerakan motor stepper seketika."""
        seq = self._next_seq()
        topic = self.topic_rail_x_cmd if node.lower() == "x" else self.topic_rail_y_cmd
        payload = {"cmd": "STOP", "seq": seq}
        self._publish_json(topic, payload)
        return seq

    def send_apar_spray(self, duration_ms: int = 3000) -> int:
        """Mengirim perintah aktivasi semprotan APAR selama durasi tertentu."""
        seq = self._next_seq()
        payload = {"cmd": "SPRAY", "duration_ms": int(duration_ms), "seq": seq}
        self._publish_json(self.topic_apar_cmd, payload)
        logger.info(f"[SEND SPRAY] Memicu semprotan APAR selama {duration_ms} ms (seq: {seq})")
        return seq

    def send_apar_release(self) -> int:
        """Melepaskan tuas APAR kembali ke posisi aman."""
        seq = self._next_seq()
        payload = {"cmd": "RELEASE", "seq": seq}
        self._publish_json(self.topic_apar_cmd, payload)
        return seq

    def send_emergency_stop(self) -> int:
        """Mengirim perintah darurat broadcast ke seluruh node."""
        seq = self._next_seq()
        payload = {"cmd": "EMERGENCY_STOP", "seq": seq}
        self._publish_json(self.topic_sys_cmd, payload)
        # Kirim juga langsung ke masing-masing node untuk redundansi keamanan
        self._publish_json(self.topic_rail_x_cmd, {"cmd": "STOP", "seq": seq})
        self._publish_json(self.topic_rail_y_cmd, {"cmd": "STOP", "seq": seq})
        self._publish_json(self.topic_apar_cmd, {"cmd": "RELEASE", "seq": seq})
        logger.warning(f"[E-STOP] Perintah EMERGENCY STOP diterbitkan ke seluruh perangkat! (seq: {seq})")
        return seq

    def send_home_all(self) -> int:
        """Memerintahkan auto-homing untuk kedua sumbu secara simultan."""
        seq_x = self.send_stepper_home("x")
        seq_y = self.send_stepper_home("y")
        return max(seq_x, seq_y)

    def _publish_json(self, topic: str, payload_dict: Dict[str, Any]):
        msg = json.dumps(payload_dict)
        self.client.publish(topic, msg, qos=0)

    # --- Synchronization Barrier Methods ---
    def wait_for_axes_done(self, timeout_sec: float = 30.0) -> bool:
        """
        SYNCHRONIZATION BARRIER:
        Menahan eksekusi thread sampai KEDUA Sumbu X dan Sumbu Y melaporkan
        status 'DONE' atau 'HOMED'.
        
        Returns:
            True jika kedua sumbu selesai sebelum timeout, False jika batas waktu terlewati.
        """
        start_time = time.time()
        with self._sync_condition:
            while True:
                state_x = self.nodes_status["x"]["state"]
                state_y = self.nodes_status["y"]["state"]

                # Selesai jika kedua sumbu berada dalam status selesai
                x_done = state_x in ["DONE", "HOMED", "IDLE"]
                y_done = state_y in ["DONE", "HOMED", "IDLE"]

                if x_done and y_done:
                    logger.info(f"[BARRIER] Kedua sumbu selesai mencapai target! (X:{state_x}, Y:{state_y})")
                    return True

                elapsed = time.time() - start_time
                remaining = timeout_sec - elapsed
                if remaining <= 0:
                    logger.error(
                        f"[BARRIER TIMEOUT] Melebihi batas waktu {timeout_sec}s! Status saat ini: "
                        f"X={state_x} (pos:{self.nodes_status['x']['pos']}), "
                        f"Y={state_y} (pos:{self.nodes_status['y']['pos']})"
                    )
                    return False

                # Tunggu notifikasi dari _on_message dengan sisa timeout
                self._sync_condition.wait(timeout=min(0.5, remaining))

    def wait_for_apar_completed(self, timeout_sec: float = 10.0) -> bool:
        """
        Menunggu hingga ESP 3 melaporkan bahwa aksi penyemprotan APAR selesai.
        """
        start_time = time.time()
        with self._sync_condition:
            while True:
                state_apar = self.nodes_status["apar"]["state"]
                if state_apar in ["COMPLETED", "IDLE", "READY"]:
                    logger.info(f"[APAR DONE] Penyemprotan APAR terkonfirmasi selesai (State: {state_apar})")
                    return True

                elapsed = time.time() - start_time
                remaining = timeout_sec - elapsed
                if remaining <= 0:
                    logger.warning(f"[APAR TIMEOUT] Menunggu konfirmasi semprot APAR melebihi {timeout_sec}s.")
                    return False

                self._sync_condition.wait(timeout=min(0.5, remaining))

    def get_node_states(self) -> Dict[str, Any]:
        """Mengambil snapshot status seluruh node perangkat keras."""
        with self.lock:
            return {
                "x": dict(self.nodes_status["x"]),
                "y": dict(self.nodes_status["y"]),
                "apar": dict(self.nodes_status["apar"]),
                "is_connected": self.is_connected,
            }
