"""
=====================================================================
DEEZPRO INTEGRATION - SAFETY MONITOR
=====================================================================
File: safety_monitor.py
Deskripsi: Pengawas keamanan hardware, mendeteksi hilangnya koneksi
            (heartbeat watchdog), memantau batas timeout pergerakan,
            dan menangani fungsi darurat (Emergency Stop / E-STOP).
"""

import time
from typing import Dict, Any
from .system_logger import get_logger

logger = get_logger()


class SafetyMonitor:
    """
    Sistem pengawas keselamatan operasional sistem pemadam Deezpro.
    """

    def __init__(self, heartbeat_timeout_sec: float = 15.0):
        self.heartbeat_timeout_sec = float(heartbeat_timeout_sec)
        self.is_emergency_active = False

    def check_node_health(self, node_states: Dict[str, Any]) -> Dict[str, bool]:
        """
        Memeriksa apakah ketiga node ESP32 aktif mengirimkan pesan status dalam batas timeout.
        
        Returns:
            Dict[node_name, is_alive]
        """
        now = time.time()
        health = {}

        for node_name in ["x", "y", "apar"]:
            info = node_states.get(node_name, {})
            last_seen = info.get("last_seen", 0.0)
            elapsed = now - last_seen if last_seen > 0 else 999.0

            is_alive = elapsed <= self.heartbeat_timeout_sec
            health[node_name] = is_alive

            if not is_alive and last_seen > 0:
                logger.warning(
                    f"[WATCHDOG] Node {node_name.upper()} tidak merespons selama {elapsed:.1f} detik! (Batas: {self.heartbeat_timeout_sec}s)"
                )

        return health

    def trigger_emergency_stop(self, mqtt_bridge) -> bool:
        """Memicu penghentian darurat sistem secara instan."""
        self.is_emergency_active = True
        logger.critical("[E-STOP ACTIVATED] Penghentian darurat dipicu oleh Safety Monitor!")
        try:
            mqtt_bridge.send_emergency_stop()
            return True
        except Exception as e:
            logger.critical(f"Gagal memancarkan E-STOP via MQTT: {e}")
            return False

    def reset_emergency_stop(self):
        """Mereset status darurat setelah intervensi manual oleh operator."""
        self.is_emergency_active = False
        logger.info("[E-STOP RESET] Status darurat telah di-reset oleh operator.")
