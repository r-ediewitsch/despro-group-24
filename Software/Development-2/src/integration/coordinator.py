"""
=====================================================================
DEEZPRO INTEGRATION - CENTRAL COORDINATOR (CORE CONTROLLER)
=====================================================================
File: coordinator.py
Deskripsi: Otak utama pengendali sistem pemadam api otonom Deezpro.
            Mengintegrasikan Kinematics, MQTT Bridge, dan Safety Monitor.
            Mengatur siklus state machine, barrier sinkronisasi gerak,
            logika batas retry semprot (maksimal 3x), penanganan target
            hilang/bergeser, dan failsafe status FAULT.
"""

import time
import json
import os
import threading
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Callable

from .system_logger import get_logger
from .kinematics import KinematicsEngine
from .mqtt_bridge import MqttBridge
from .safety_monitor import SafetyMonitor

logger = get_logger()


class SystemState(Enum):
    """Definisi status siklus hidup sistem Deezpro."""
    INIT = "INIT"
    BOOT_HOMING = "BOOT_HOMING"
    IDLE_MONITORING = "IDLE_MONITORING"
    TARGET_LOCKED = "TARGET_LOCKED"
    AXES_MOVING = "AXES_MOVING"
    SYNC_BARRIER = "SYNC_BARRIER"
    SPRAYING = "SPRAYING"
    VERIFYING = "VERIFYING"
    RETURN_HOME = "RETURN_HOME"
    FAULT = "FAULT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class CentralCoordinator:
    """
    Central Coordinator untuk sistem pemadam api otomatis Deezpro.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kinematics = KinematicsEngine.from_config(config)
        self.mqtt = MqttBridge(config)
        self.safety = SafetyMonitor()

        coord_cfg = config.get("coordinator", {})
        self.max_spray_retries = int(coord_cfg.get("max_spray_retries", 3))
        self.move_timeout_sec = float(coord_cfg.get("move_timeout_sec", 30.0))
        self.verify_wait_sec = float(coord_cfg.get("verify_wait_sec", 2.5))
        self.spray_radius_threshold_cm = float(coord_cfg.get("spray_radius_threshold_cm", 20.0))
        self.home_after_extinguish = bool(coord_cfg.get("home_after_extinguish", True))

        apar_cfg = config.get("apar", {})
        self.default_spray_duration_ms = int(apar_cfg.get("default_spray_duration_ms", 3000))

        # State Tracking & Persistence
        self.state = SystemState.INIT
        self.retry_count = 0
        self.lock = threading.RLock()
        self.active_target_cm: Optional[Tuple[float, float]] = None
        self.last_target_steps: Optional[Tuple[int, int]] = None
        self.is_running = False

        # Lokasi berkas penyimpanan persisten koordinat & step terkini
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.state_file_path = os.path.join(base_dir, "logs", "runtime_state.json")

    def start(self, auto_home: bool = True) -> bool:
        """Menghidupkan sistem, menghubungkan MQTT, dan mengeksekusi Homing awal."""
        logger.info("Menjalankan Central Coordinator Deezpro...")
        self.is_running = True

        if not self.mqtt.start():
            logger.error("Gagal memulai MQTT Bridge! Sistem tidak dapat beroperasi.")
            self.state = SystemState.FAULT
            return False

        if auto_home:
            logger.info("Mengeksekusi inisialisasi Auto-Homing pada seluruh sumbu...")
            if not self.execute_homing_all():
                logger.error("Auto-Homing saat startup gagal! Sistem berpindah ke status FAULT.")
                self.set_state(SystemState.FAULT)
                return False

        self.set_state(SystemState.IDLE_MONITORING)
        logger.info("Sistem siap siaga dan berada dalam status IDLE_MONITORING.")
        return True

    def stop(self):
        """Mematikan koordinator dan memutus jaringan MQTT secara aman."""
        logger.info("Menghentikan Central Coordinator...")
        self.is_running = False
        self.mqtt.stop()
        self.set_state(SystemState.INIT)

    def set_state(self, new_state: SystemState):
        """Memperbarui status sistem dan mencatat perubahan ke log."""
        with self.lock:
            if self.state != new_state:
                logger.info(f"[STATE TRANSITION] {self.state.value} -> {new_state.value}")
                self.state = new_state
                self.save_runtime_state()

    def get_state(self) -> SystemState:
        with self.lock:
            return self.state

    def save_runtime_state(self):
        """
        Menyimpan koordinat asli, target aktif, posisi step terkini,
        dan status sistem ke berkas JSON persisten (logs/runtime_state.json).
        """
        try:
            node_states = self.mqtt.get_node_states()
            pos_x = node_states["x"].get("pos", 0)
            pos_y = node_states["y"].get("pos", 0)
            curr_x_cm, curr_y_cm = self.kinematics.steps_to_coordinates(pos_x, pos_y)

            state_data = {
                "timestamp": time.time(),
                "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_state": self.state.value,
                "current_position_cm": {"x": curr_x_cm, "y": curr_y_cm},
                "current_position_steps": {"x": pos_x, "y": pos_y},
                "last_active_target_cm": (
                    {"x": self.active_target_cm[0], "y": self.active_target_cm[1]}
                    if self.active_target_cm else None
                ),
                "last_active_target_steps": (
                    {"step_x": self.last_target_steps[0], "step_y": self.last_target_steps[1]}
                    if self.last_target_steps else None
                ),
                "retry_count": self.retry_count,
                "nodes": {
                    "x": node_states["x"],
                    "y": node_states["y"],
                    "apar": node_states["apar"],
                }
            }

            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Gagal menyimpan runtime state ke file: {e}")

    # --- Operasi Auto-Homing ---
    def execute_homing_all(self, timeout_sec: float = 40.0) -> bool:
        """
        Memerintahkan auto-homing Sumbu X dan Y secara simultan
        dan menunggu konfirmasi dari limit switch kedua sumbu.
        """
        self.set_state(SystemState.BOOT_HOMING)
        logger.info("Menerbitkan perintah HOME ke Sumbu X dan Sumbu Y...")
        self.mqtt.send_home_all()

        success = self.mqtt.wait_for_axes_done(timeout_sec=timeout_sec)
        if success:
            logger.info("Auto-Homing kedua sumbu berhasil! Posisi referensi terkalibrasi ke (0, 0).")
            self.set_state(SystemState.IDLE_MONITORING)
            return True
        else:
            logger.error("Timeout menunggu konfirmasi Auto-Homing dari limit switch!")
            self.set_state(SystemState.FAULT)
            return False

    # --- Operasi Penanganan Target Api (Dispatch Target Pipeline) ---
    def dispatch_target(
        self,
        x_api_cm: float,
        y_api_cm: float,
        is_single_cam: bool = False,
        fire_verifier_cb: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """
        Alur kerja pemadaman lengkap:
        1. Clamping koordinat & translasi langkah motor.
        2. Penguncian target (TARGET_LOCKED).
        3. Penggerakan simultan kedua sumbu (AXES_MOVING).
        4. Menunggu konfirmasi selesai (SYNC_BARRIER).
        5. Pemeriksaan target hilang/bergeser.
        6. Penyemprotan APAR (SPRAYING).
        7. Verifikasi pemadaman hingga maksimal 3x retry (VERIFYING).
        8. Kembali ke titik asal (RETURN_HOME) atau masuk status FAULT.
        """
        with self.lock:
            if self.state not in [SystemState.IDLE_MONITORING, SystemState.TARGET_LOCKED]:
                logger.warning(
                    f"Pengiriman target diabaikan karena sistem sedang dalam status: {self.state.value}"
                )
                return False

            self.set_state(SystemState.TARGET_LOCKED)
            self.retry_count = 0

        cam_mode = "Single-Camera Fallback" if is_single_cam else "Dual-Camera Fusion"
        logger.info(f"Target api diterima ({cam_mode}): ({x_api_cm:.1f}, {y_api_cm:.1f}) cm")

        # 1. Kinematics Clamping & Step Calculation
        step_x, step_y, (clamped_x, clamped_y), is_clamped = self.kinematics.coordinates_to_steps(
            x_api_cm, y_api_cm
        )
        if is_clamped:
            logger.warning(
                f"[CLAMPING AKTIF] Titik api ({x_api_cm:.1f}, {y_api_cm:.1f}) di zona clearance! "
                f"Disesuaikan ke titik jangkauan terdekat: ({clamped_x:.1f}, {clamped_y:.1f}) cm"
            )

        self.active_target_cm = (clamped_x, clamped_y)
        self.last_target_steps = (step_x, step_y)
        self.save_runtime_state()

        # 2. Gerakan Simultan Kedua Sumbu
        self.set_state(SystemState.AXES_MOVING)
        logger.info(f"Menggerakkan kedua sumbu serentak ke Step X={step_x}, Step Y={step_y} ...")
        self.mqtt.send_stepper_goto("x", step_x)
        self.mqtt.send_stepper_goto("y", step_y)

        # 3. Synchronization Barrier (Menunggu kedua motor selesai)
        self.set_state(SystemState.SYNC_BARRIER)
        move_success = self.mqtt.wait_for_axes_done(timeout_sec=self.move_timeout_sec)
        self.save_runtime_state()
        if not move_success:
            logger.critical("Gagal mencapai target pergerakan! Kemungkinan motor macet atau komunikasi putus.")
            self.set_state(SystemState.FAULT)
            return False

        # 4. Skenario Edge Case: Periksa apakah target api hilang sebelum semprot
        if fire_verifier_cb is not None:
            time.sleep(1.0) # Jeda stabilisasi citra
            still_burning = fire_verifier_cb()
            if not still_burning:
                logger.warning(
                    "[EDGE CASE: TARGET LOST] Api sudah tidak terdeteksi saat nozzle tiba! "
                    "Membatalkan penyemprotan APAR untuk menghemat media pemadam."
                )
                self._return_to_standby()
                return True

        # 5. Siklus Penyemprotan & Verifikasi (Maksimal 3x Retry)
        while self.retry_count < self.max_spray_retries:
            self.set_state(SystemState.SPRAYING)
            logger.info(
                f"Mengeksekusi penyemprotan APAR (Percobaan {self.retry_count + 1}/{self.max_spray_retries}) "
                f"selama {self.default_spray_duration_ms} ms..."
            )
            self.mqtt.send_apar_spray(self.default_spray_duration_ms)

            # Tunggu semprotan selesai
            self.mqtt.wait_for_apar_completed(timeout_sec=self.default_spray_duration_ms / 1000.0 + 3.0)

            # 6. Fase Verifikasi Pemadaman
            self.set_state(SystemState.VERIFYING)
            logger.info(f"Menunggu {self.verify_wait_sec} detik untuk evaluasi hasil pemadaman...")
            time.sleep(self.verify_wait_sec)

            if fire_verifier_cb is None:
                # Jika tidak ada verifier dinamis, asumsikan 1x semprot cukup pada pengujian kontrol murni
                logger.info("Verifikasi selesai (mode simulasi/tanpa sensor kamera live).")
                break

            is_fire_active = fire_verifier_cb()
            if not is_fire_active:
                logger.info("VERIFIKASI SUKSES: Api terkonfirmasi telah padam sepenuhnya!")
                break
            else:
                self.retry_count += 1
                logger.warning(
                    f"[PERINGATAN] Api masih menyala setelah disemprot! "
                    f"Percobaan gagal: {self.retry_count}/{self.max_spray_retries}"
                )

        # 7. Penanganan jika melebihi batas percobaan semprot
        if self.retry_count >= self.max_spray_retries:
            logger.critical(
                f"[FATAL: STATUS FAULT] Api masih menyala setelah {self.max_spray_retries} kali semprot! "
                "Penyemprotan dihentikan untuk menghindari habisnya tabung APAR. "
                "Sistem terkunci pada status FAULT dan membunyikan alarm kritis operator!"
            )
            self.set_state(SystemState.FAULT)
            return False

        # 8. Kembali ke Posisi Standby
        if self.home_after_extinguish:
            self._return_to_standby()
        else:
            self.set_state(SystemState.IDLE_MONITORING)

        return True

    def _return_to_standby(self):
        """Mengembalikan kedua sumbu railing ke titik parkir/standby (0, 0)."""
        self.set_state(SystemState.RETURN_HOME)
        logger.info("Mengembalikan sistem ke posisi awal (0, 0)...")
        self.mqtt.send_stepper_goto("x", 0)
        self.mqtt.send_stepper_goto("y", 0)
        self.mqtt.wait_for_axes_done(timeout_sec=self.move_timeout_sec)
        self.set_state(SystemState.IDLE_MONITORING)
        logger.info("Nozzle kembali ke posisi standby. Sistem siap memantau kembali.")

    # --- Operasi Manual & Diagnostik ---
    def manual_goto_cm(self, x_cm: float, y_cm: float) -> bool:
        """Menggerakkan nozzle secara manual ke koordinat tertentu tanpa menyemprot APAR."""
        step_x, step_y, (clamped_x, clamped_y), is_clamped = self.kinematics.coordinates_to_steps(
            x_cm, y_cm
        )
        if is_clamped:
            logger.warning(f"[MANUAL CLAMPED] Koordinat dibatasi ke ({clamped_x}, {clamped_y}) cm")

        logger.info(f"[MANUAL GOTO] Menggerakkan ke ({clamped_x}, {clamped_y}) cm -> Step ({step_x}, {step_y})")
        self.mqtt.send_stepper_goto("x", step_x)
        self.mqtt.send_stepper_goto("y", step_y)
        return self.mqtt.wait_for_axes_done(timeout_sec=self.move_timeout_sec)

    def manual_goto_steps(self, step_x: int, step_y: int, wait_done: bool = True) -> bool:
        """
        Menggerakkan KEDUA motor secara simultan menggunakan nilai langkah (step) absolut,
        tanpa konversi koordinat cm.
        """
        # Validasi batas langkah fisik
        max_p = self.kinematics.total_steps_x_p
        max_q = self.kinematics.total_steps_y_q
        target_x = max(0, min(int(step_x), max_p))
        target_y = max(0, min(int(step_y), max_q))

        if target_x != step_x or target_y != step_y:
            logger.warning(
                f"[STEP CLAMPED] Input step ({step_x}, {step_y}) dibatasi dalam rentang "
                f"aman (0-{max_p}, 0-{max_q}) -> ({target_x}, {target_y})"
            )

        eq_x_cm, eq_y_cm = self.kinematics.steps_to_coordinates(target_x, target_y)
        logger.info(
            f"[MANUAL STEP GOTO] Menggerakkan KEDUA motor ke Step (X: {target_x}, Y: {target_y}) "
            f"[Setara fisik: {eq_x_cm:.1f} cm, {eq_y_cm:.1f} cm]..."
        )

        self.set_state(SystemState.MOVING_TO_TARGET)
        self.mqtt.send_stepper_goto("x", target_x)
        self.mqtt.send_stepper_goto("y", target_y)

        if wait_done:
            done = self.mqtt.wait_for_axes_done(timeout_sec=self.move_timeout_sec)
            self.set_state(SystemState.IDLE_MONITORING)
            return done
        return True

    def manual_jog_both_steps(self, dx_steps: int, dy_steps: int, wait_done: bool = False) -> bool:
        """
        Menggerakkan KEDUA motor secara simultan sejumlah langkah relatif (delta step).
        """
        logger.info(f"[MANUAL MOVE BOTH] Menggerakkan relatif Sumbu X: {dx_steps} step, Sumbu Y: {dy_steps} step...")
        self.mqtt.send_stepper_move("x", dx_steps)
        self.mqtt.send_stepper_move("y", dy_steps)
        if wait_done:
            return self.mqtt.wait_for_axes_done(timeout_sec=self.move_timeout_sec)
        return True

    def manual_jog_steps(self, node: str, steps: int) -> bool:
        """Menggerakkan sumbu tertentu sejauh jumlah langkah relatif (jogging)."""
        self.mqtt.send_stepper_move(node, steps)
        return True

    def manual_spray_test(self, duration_ms: int = 2000) -> bool:
        """Uji manual semprotan APAR untuk memeriksa fungsi servo dan tuas mekanik."""
        logger.info(f"[MANUAL SPRAY] Menguji semprot APAR selama {duration_ms} ms...")
        self.mqtt.send_apar_spray(duration_ms)
        return self.mqtt.wait_for_apar_completed(timeout_sec=duration_ms / 1000.0 + 3.0)

    def trigger_emergency_stop(self):
        """Memicu penghentian darurat instan seluruh motor dan aktuator."""
        self.set_state(SystemState.EMERGENCY_STOP)
        self.safety.trigger_emergency_stop(self.mqtt)

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Mengambil data telemetri komprehensif sistem."""
        node_states = self.mqtt.get_node_states()
        pos_x_steps = node_states["x"].get("pos", 0)
        pos_y_steps = node_states["y"].get("pos", 0)
        curr_x_cm, curr_y_cm = self.kinematics.steps_to_coordinates(pos_x_steps, pos_y_steps)

        return {
            "state": self.state.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_spray_retries,
            "current_position_cm": {"x": curr_x_cm, "y": curr_y_cm},
            "current_position_steps": {"x": pos_x_steps, "y": pos_y_steps},
            "nodes": node_states,
            "mqtt_connected": self.mqtt.is_connected,
            "kinematics": self.kinematics.get_kinematic_summary(),
        }
