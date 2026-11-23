"""
=====================================================================
DEEZPRO - TEST HOMOGRAPHY END-TO-END (VIEWFINDER -> HOMOGRAPHY -> ESP32)
=====================================================================
File: test_homography_end_to_end.py
Deskripsi: Program uji end-to-end terintegrasi untuk memvalidasi hasil
            kalibrasi homografi kamera terhadap posisi fisik aktual nozzle.
            
Alur Pengujian:
  1. Menampilkan streaming live camera dengan koreksi distorsi lensa (undistort).
  2. Pengguna menggambar bounding box objek di lantai secara manual menggunakan mouse (drag).
  3. Program mengekstrak titik kontak dasar lantai: (u_b, v_b) = ((x1+x2)/2, y2).
  4. Mentransformasikan koordinat piksel (u, v) ke koordinat fisik lantai (X_w, Y_w) cm
     menggunakan matriks homografi (HomographyEngine).
  5. Menghitung clamping area batas dan konversi langkah motor stepper (KinematicsEngine).
  6. Mengirimkan perintah pergerakan (GOTO) ke ESP32 melalui MQTT (CentralCoordinator)
     sehingga nozzle bergerak ke lokasi target untuk memverifikasi akurasi fisik di dunia nyata.

Kontrol Keyboard:
  [G] / [ENTER] : Kirim perintah GOTO ke ESP32 (Nozzle bergerak ke target tanpa semprot).
  [S]           : Kirim perintah GOTO + SPRAY ke ESP32 (Gerak ke target lalu semprot 2 detik).
  [H]           : Kembalikan nozzle ke koordinat asal fisik (0, 0 steps).
  [C]           : Hapus/Reset Bounding Box yang sedang dipilih.
  [SPACE]       : EMERGENCY STOP (Hentikan seluruh motor seketika).
  [Q] / [ESC]   : Keluar dari program.
"""

import os
import sys
import time
import argparse
import threading
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import yaml

# Pastikan path modul lokal dikenali
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vision.homography import HomographyEngine
from src.integration.kinematics import KinematicsEngine
from src.integration.coordinator import CentralCoordinator, SystemState
from src.integration.system_logger import setup_logger

logger = setup_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Deezpro End-to-End Homography Validation & Hardware Dispatch Test."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Nomor logika kamera: 1 untuk Kamera 1, 2 untuk Kamera 2 (default: 1).",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Override index hardware OpenCV (0, 1, 2...). Jika tidak diisi, membaca dari config/system_config.yaml.",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path berkas matriks kalibrasi lensa (.npz). Default: config/calib_cam<cam>.npz",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=None,
        help="Path berkas matriks homografi (.npz). Default: config/homography_cam<cam>.npz",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path berkas konfigurasi sistem YAML. Default: config/system_config.yaml",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Nonaktifkan koreksi distorsi lensa kamera.",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Jalankan dalam mode Dry-Run tanpa menghubungkan ke broker MQTT / ESP32.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["base", "centroid"],
        default="base",
        help="Titik acuan proyeksi: 'base' (kontak dasar lantai) atau 'centroid' (tengah kotak). Default: base.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Resolusi lebar tangkapan kamera (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Resolusi tinggi tangkapan kamera (default: 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate kamera (default: 30 FPS).",
    )
    return parser.parse_args()


class ManualBBoxSelector:
    """Mengelola interaksi mouse untuk menggambar bounding box pada viewfinder."""

    def __init__(self, target_mode: str = "base"):
        self.target_mode = target_mode
        self.is_drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.curr_pt: Optional[Tuple[int, int]] = None
        self.bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self.target_pixel: Optional[Tuple[float, float]] = None  # (u, v)
        self.centroid_pixel: Optional[Tuple[float, float]] = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_pt = (x, y)
            self.curr_pt = (x, y)
            self.bbox = None
            self.target_pixel = None
            self.centroid_pixel = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.curr_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                if self.start_pt is not None:
                    x0, y0 = self.start_pt
                    x1, y1 = x, y
                    x_min, x_max = min(x0, x1), max(x0, x1)
                    y_min, y_max = min(y0, y1), max(y0, y1)

                    # Pastikan ukuran bounding box cukup signifikan (> 6x6 piksel)
                    if (x_max - x_min) >= 6 and (y_max - y_min) >= 6:
                        self.bbox = (x_min, y_min, x_max, y_max)
                        # Hitung centroid
                        u_c = (x_min + x_max) / 2.0
                        v_c = (y_min + y_max) / 2.0
                        self.centroid_pixel = (u_c, v_c)

                        # Hitung titik acuan proyeksi lantai
                        if self.target_mode == "base":
                            # Dasar bawah bounding box yang bersentuhan dengan lantai
                            u_b = (x_min + x_max) / 2.0
                            v_b = float(y_max)
                            self.target_pixel = (u_b, v_b)
                        else:
                            self.target_pixel = (u_c, v_c)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear()

    def clear(self):
        """Mereset bounding box dan target yang dipilih."""
        self.is_drawing = False
        self.start_pt = None
        self.curr_pt = None
        self.bbox = None
        self.target_pixel = None
        self.centroid_pixel = None


class HomographyEndToEndApp:
    """Aplikasi pengujian end-to-end visual dan pengiriman perintah ke ESP32."""

    def __init__(self, args):
        self.args = args
        self.cam_id = args.camera
        self.cfg_dir = os.path.join(current_dir, "config")
        self.config_path = args.config or os.path.join(self.cfg_dir, "system_config.yaml")

        # 1. Muat Konfigurasi Sistem
        self.config = self._load_system_config()

        # 2. Resolusi Hardware Index Kamera
        self.device_index = self._resolve_camera_index(args.source)

        # 3. Inisialisasi Kinematics Engine
        self.kinematics = KinematicsEngine.from_config(self.config)

        # 4. Inisialisasi Matriks Kalibrasi Lensa (Undistort)
        self.calib_file = args.calib or os.path.join(self.cfg_dir, f"calib_cam{self.cam_id}.npz")
        self.map1, self.map2 = None, None
        if not args.no_undistort:
            self._init_undistort_maps()

        # 5. Inisialisasi Homography Engine
        self.homography_file = args.homography or os.path.join(self.cfg_dir, f"homography_cam{self.cam_id}.npz")
        self.homography_engine, self.is_synthetic_homography = self._init_homography()

        # 6. Inisialisasi Central Coordinator (MQTT)
        self.coordinator: Optional[CentralCoordinator] = None
        self.mqtt_connected = False
        if not args.no_mqtt:
            self._init_coordinator()

        # 7. UI State & Mouse Selector
        self.selector = ManualBBoxSelector(target_mode=args.target_mode)
        self.window_name = f"Deezpro - End-to-End Homography Test (Kamera {self.cam_id})"
        self.action_status = "SIAP (Tarik mouse untuk membuat Bounding Box)"
        self.is_action_busy = False
        self.last_action_time = 0.0

    def _load_system_config(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.warning(f"File konfigurasi tidak ditemukan: {self.config_path}. Menggunakan default fallback.")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Gagal membaca system_config.yaml: {e}")
            return {}

    def _resolve_camera_index(self, cli_source: Optional[int]) -> int:
        if cli_source is not None:
            logger.info(f"Menggunakan override index kamera CLI: {cli_source}")
            return cli_source

        cam_key = f"cam{self.cam_id}"
        cameras_cfg = self.config.get("cameras", {})
        if cam_key in cameras_cfg and "index" in cameras_cfg[cam_key]:
            idx = int(cameras_cfg[cam_key]["index"])
            logger.info(f"Menggunakan index kamera dari konfigurasi: {idx}")
            return idx

        fallback_idx = self.cam_id if self.cam_id in [0, 1] else 0
        logger.info(f"Index kamera tidak terkonfigurasi, menggunakan fallback: {fallback_idx}")
        return fallback_idx

    def _init_undistort_maps(self):
        """Memuat matriks distorsi intrinsik dan menyiapkan remap map."""
        if not os.path.exists(self.calib_file):
            alt_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                "Development", "callibration", f"calib_cam{self.cam_id}.npz"
            )
            if os.path.exists(alt_path):
                self.calib_file = alt_path

        if os.path.exists(self.calib_file):
            try:
                data = np.load(self.calib_file)
                K = data["camera_matrix"]
                if "dist_coeffs" in data:
                    D = data["dist_coeffs"]
                elif "dist_coeff" in data:
                    D = data["dist_coeff"]
                elif "D" in data:
                    D = data["D"]
                else:
                    D = np.zeros(5)

                w, h = self.args.width, self.args.height
                new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
                self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
                logger.info(f"Matriks distorsi lensa dimuat: {os.path.basename(self.calib_file)} (Undistort Aktif)")
            except Exception as e:
                logger.error(f"Gagal menginisialisasi koreksi distorsi: {e}")
        else:
            logger.warning(
                f"Berkas kalibrasi lensa ({self.calib_file}) tidak ditemukan! "
                "Berjalan tanpa koreksi distorsi."
            )

    def _init_homography(self) -> Tuple[HomographyEngine, bool]:
        """Memuat matriks homografi dari berkas atau membuat matriks sintetis sementara."""
        if os.path.exists(self.homography_file):
            try:
                engine = HomographyEngine.load(self.homography_file)
                logger.info(
                    f"Matriks homografi berhasil dimuat dari: {self.homography_file} "
                    f"(Mean error: {engine.reprojection_error_cm:.2f} cm)"
                )
                return engine, False
            except Exception as e:
                logger.error(f"Gagal memuat berkas homografi {self.homography_file}: {e}")

        # Fallback homografi sintetis jika file belum ada
        logger.warning(
            f"[PERINGATAN] Berkas homografi '{self.homography_file}' belum ditemukan! "
            "Membuat matriks homografi aproksimasi default (0..Width -> 0..200 cm) "
            "agar pengujian alur tetap dapat dieksekusi. "
            "Jalankan 'calibrate_homography.py' untuk mendapatkan kalibrasi presisi."
        )
        engine = HomographyEngine()
        w, h = float(self.args.width), float(self.args.height)
        # 4 titik aproksimasi perspektif default lantai 200x200 cm
        engine.add_point_pair((0.15 * w, 0.85 * h), (10.0, 10.0))
        engine.add_point_pair((0.85 * w, 0.85 * h), (190.0, 10.0))
        engine.add_point_pair((0.70 * w, 0.30 * h), (190.0, 190.0))
        engine.add_point_pair((0.30 * w, 0.30 * h), (10.0, 190.0))
        engine.compute()
        return engine, True

    def _init_coordinator(self):
        """Menghubungkan CentralCoordinator ke MQTT Broker tanpa auto-homing."""
        try:
            logger.info("Menghubungkan Central Coordinator ke MQTT Broker...")
            self.coordinator = CentralCoordinator(self.config)
            # auto_home=False karena prototipe fisik belum terpasang limit switch
            started = self.coordinator.start(auto_home=False)
            if started:
                self.mqtt_connected = True
                logger.info("Koneksi MQTT ke ESP32 aktif & siap menerima perintah.")
            else:
                logger.warning("Gagal terhubung ke MQTT Broker! Beralih ke mode offline.")
        except Exception as e:
            logger.error(f"Error inisialisasi coordinator: {e}")
            self.coordinator = None
            self.mqtt_connected = False

    # --- Eksekusi Aksi Hardware dalam Thread Terpisah (Non-blocking GUI) ---
    def _execute_async_action(self, target_fn, *args):
        """Menjalankan fungsi perintah hardware di background thread agar GUI tidak membeku."""
        if self.is_action_busy:
            logger.warning("Perintah sebelumnya masih berjalan! Harap tunggu.")
            return

        self.is_action_busy = True
        self.last_action_time = time.time()

        def worker():
            try:
                target_fn(*args)
            except Exception as e:
                self.action_status = f"ERROR AKSI: {e}"
                logger.error(f"Error eksekusi aksi hardware: {e}")
            finally:
                self.is_action_busy = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def send_goto_command(self, x_cm: float, y_cm: float):
        """Mengirim perintah gerak ke koordinat cm target."""
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO ({x_cm:.1f}, {y_cm:.1f}) cm (MQTT Offline)"
            logger.info(self.action_status)
            return

        def _goto():
            self.action_status = f"Menggerakkan nozzle ke ({x_cm:.1f}, {y_cm:.1f}) cm..."
            logger.info(self.action_status)
            success = self.coordinator.manual_goto_cm(x_cm, y_cm)
            if success:
                self.action_status = f"Nozzle TIBA di koordinat ({x_cm:.1f}, {y_cm:.1f}) cm!"
                logger.info(self.action_status)
            else:
                self.action_status = "Pergerakan timeout atau dibatalkan!"
                logger.warning(self.action_status)

        self._execute_async_action(_goto)

    def send_goto_and_spray_command(self, x_cm: float, y_cm: float, duration_ms: int = 2000):
        """Mengirim perintah gerak ke target lalu menyemprotkan APAR."""
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO ({x_cm:.1f}, {y_cm:.1f}) cm + SPRAY {duration_ms}ms"
            logger.info(self.action_status)
            return

        def _goto_spray():
            self.action_status = f"Menggerakkan nozzle ke ({x_cm:.1f}, {y_cm:.1f}) cm..."
            logger.info(self.action_status)
            success = self.coordinator.manual_goto_cm(x_cm, y_cm)
            if success:
                self.action_status = f"Tiba di target. Menyemprotkan APAR ({duration_ms} ms)..."
                logger.info(self.action_status)
                self.coordinator.manual_spray_test(duration_ms)
                self.action_status = f"Penyemprotan di ({x_cm:.1f}, {y_cm:.1f}) cm selesai!"
                logger.info(self.action_status)
            else:
                self.action_status = "Gagal mencapai target, penyemprotan dibatalkan."
                logger.warning(self.action_status)

        self._execute_async_action(_goto_spray)

    def send_return_zero_command(self):
        """Mengembalikan nozzle ke posisi referensi (0, 0 steps)."""
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = "[DRY-RUN] Kembali ke posisi referensi (0, 0 steps)"
            logger.info(self.action_status)
            return

        def _return_zero():
            self.action_status = "Menggerakkan nozzle kembali ke Step (0, 0)..."
            logger.info(self.action_status)
            self.coordinator.manual_goto_steps(0, 0, wait_done=True)
            self.action_status = "Nozzle berada di posisi awal (0, 0 steps)."
            logger.info(self.action_status)

        self._execute_async_action(_return_zero)

    def trigger_estop(self):
        """Memicu penghentian darurat instan."""
        self.action_status = "EMERGENCY STOP DIAKTIFKAN!"
        logger.critical("EMERGENCY STOP dipicu oleh pengguna!")
        if self.coordinator:
            self.coordinator.trigger_emergency_stop()

    # --- Render HUD & Visualisasi ---
    def _draw_hud(
        self,
        frame: np.ndarray,
        raw_world: Optional[Tuple[float, float]],
        clamped_world: Optional[Tuple[float, float]],
        target_steps: Optional[Tuple[int, int]],
        is_clamped: bool,
    ):
        """Menggambar panel informasi, telemetri, dan instruksi pada viewfinder."""
        h, w = frame.shape[:2]

        # 1. Overlay Header Atas
        cv2.rectangle(frame, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "DEEZPRO END-TO-END HOMOGRAPHY & HARDWARE DISPATCH TEST",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        mqtt_text = "MQTT: ONLINE" if self.mqtt_connected else "MQTT: OFFLINE (DRY-RUN)"
        mqtt_color = (0, 230, 0) if self.mqtt_connected else (0, 165, 255)
        cv2.putText(frame, mqtt_text, (w - 280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mqtt_color, 2)

        # Status badge sintetis vs kalibrasi riil
        if self.is_synthetic_homography:
            cv2.putText(
                frame,
                "[H: SINTETIS / FILE TIDAK DITEMUKAN]",
                (15, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 140, 255),
                1,
            )
        else:
            err_str = f"Reproj Error: {self.homography_engine.reprojection_error_cm:.2f} cm"
            cv2.putText(
                frame,
                f"[H: TERKALIBRASI] {err_str}",
                (15, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 128),
                1,
            )

        # 2. Kotak Status Aksi & Telemetri ESP32
        status_color = (0, 255, 255) if self.is_action_busy else (220, 220, 220)
        cv2.putText(
            frame,
            f"Status: {self.action_status}",
            (w - 600, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
        )

        # 3. Panel Kiri Bawah: Detail Target Bounding Box & Koordinat
        panel_w = 420
        panel_h = 160
        panel_x = 15
        panel_y = h - panel_h - 45

        # Gambar panel semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

        cv2.putText(
            frame,
            "TARGET TRANSFORMATION DATA",
            (panel_x + 10, panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
        )

        if self.selector.target_pixel is not None and raw_world is not None:
            u, v = self.selector.target_pixel
            rx, ry = raw_world
            cx, cy = clamped_world
            sx, sy = target_steps

            cv2.putText(
                frame,
                f"1. Base Pixel (u, v) : ({u:.1f}, {v:.1f}) px",
                (panel_x + 10, panel_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
            )
            cv2.putText(
                frame,
                f"2. Homography World  : ({rx:.1f}, {ry:.1f}) cm",
                (panel_x + 10, panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 255, 100),
                1,
            )

            clamp_badge = "[CLAMPED!]" if is_clamped else "[OK]"
            clamp_col = (0, 165, 255) if is_clamped else (100, 255, 100)
            cv2.putText(
                frame,
                f"3. Clamped Target    : ({cx:.1f}, {cy:.1f}) cm {clamp_badge}",
                (panel_x + 10, panel_y + 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                clamp_col,
                1,
            )

            cv2.putText(
                frame,
                f"4. Target Motor Step : Step X={sx}, Step Y={sy}",
                (panel_x + 10, panel_y + 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

            # Telemetri posisi saat ini dari ESP32
            if self.coordinator:
                try:
                    tele = self.coordinator.get_system_telemetry()
                    curr_x = tele["current_position_steps"]["x"]
                    curr_y = tele["current_position_steps"]["y"]
                    curr_cm = tele["current_position_cm"]
                    cv2.putText(
                        frame,
                        f"5. Posisi ESP32 Saat Ini: Step ({curr_x}, {curr_y}) ~ ({curr_cm['x']:.1f}, {curr_cm['y']:.1f}) cm",
                        (panel_x + 10, panel_y + 142),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (200, 200, 200),
                        1,
                    )
                except Exception:
                    pass
        else:
            cv2.putText(
                frame,
                "Belum ada target yang dipilih.",
                (panel_x + 10, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                frame,
                "Tarik kotak (mouse drag) pada objek di lantai",
                (panel_x + 10, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
            )
            cv2.putText(
                frame,
                "untuk menghitung koordinat fisik & langkah motor.",
                (panel_x + 10, panel_y + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
            )

        # 4. Banner Bawah: Petunjuk Tombol Operasi
        cv2.rectangle(frame, (0, h - 35), (w, h), (10, 10, 10), -1)
        help_text = (
            "[Drag Mouse] Buat Box  |  [G / ENTER] Kirim GOTO ke ESP32  |  "
            "[S] GOTO + Semprot  |  [H] Posisi 0  |  [C] Reset  |  [SPACE] ESTOP  |  [Q] Keluar"
        )
        cv2.putText(frame, help_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def run(self):
        """Looping utama penangkapan video, antarmuka GUI, dan interaksi pengguna."""
        logger.info(f"Membuka kamera device index {self.device_index}...")
        cap = cv2.VideoCapture(self.device_index)

        # Pengaturan resolusi dan framerate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)

        if not cap.isOpened():
            logger.error(
                f"Gagal membuka kamera index {self.device_index}! "
                "Pastikan webcam terpasang atau gunakan parameter --source <index>."
            )
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.args.width, self.args.height)
        cv2.setMouseCallback(self.window_name, self.selector.on_mouse)

        print("\n" + "=" * 70)
        print(" DEEZPRO HOMOGRAPHY END-TO-END VALIDATOR")
        print("=" * 70)
        print(" Petunjuk:")
        print(" 1. Arahkan kamera ke area lantai uji railing 2x2 meter.")
        print(" 2. Letakkan objek/marker di lantai (contoh: cangkir, lakban, kotak).")
        print(" 3. Drag mouse kiri untuk membuat bounding box mengelilingi objek.")
        print(" 4. Periksa titik kontak lantai (Base Crosshair Hijau) dan koordinat (X, Y) cm.")
        print(" 5. Tekan tombol [G] atau [ENTER] untuk menggerakkan nozzle ESP32 ke titik tersebut.")
        print(" 6. Amati apakah corong nozzle tepat berada di atas objek fisik!")
        print("=" * 70 + "\n")

        raw_world: Optional[Tuple[float, float]] = None
        clamped_world: Optional[Tuple[float, float]] = None
        target_steps: Optional[Tuple[int, int]] = None
        is_clamped: bool = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Gagal membaca frame kamera! Mencoba menyambung kembali...")
                    time.sleep(0.1)
                    continue

                # Koreksi distorsi lensa jika map tersedia
                if self.map1 is not None and self.map2 is not None:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                # 1. Gambar Bounding Box Saat Dragging
                if self.selector.is_drawing and self.selector.start_pt and self.selector.curr_pt:
                    cv2.rectangle(frame, self.selector.start_pt, self.selector.curr_pt, (255, 255, 0), 1)

                # 2. Gambar Bounding Box yang Terpilih
                if self.selector.bbox is not None:
                    x1, y1, x2, y2 = self.selector.bbox
                    # Gambar kotak deteksi (oranye terang seperti detektor api)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(
                        frame,
                        f"TEST BBOX [{x2-x1}x{y2-y1}]",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 140, 255),
                        1,
                    )

                    # Titik centroid geometris (biru langit)
                    if self.selector.centroid_pixel is not None:
                        uc, vc = int(self.selector.centroid_pixel[0]), int(self.selector.centroid_pixel[1])
                        cv2.circle(frame, (uc, vc), 3, (255, 200, 0), -1)

                    # Titik kontak dasar lantai (Hijau Terang)
                    if self.selector.target_pixel is not None:
                        ub, vb = self.selector.target_pixel
                        ib_x, ib_y = int(round(ub)), int(round(vb))

                        # Hubungkan garis putus/lurus dari centroid ke base untuk menunjukkan proyeksi ke lantai
                        if self.selector.centroid_pixel is not None:
                            cv2.line(frame, (uc, vc), (ib_x, ib_y), (0, 255, 255), 1)

                        # Gambar crosshair dan target bullseye pada base point
                        cv2.line(frame, (ib_x - 12, ib_y), (ib_x + 12, ib_y), (0, 255, 0), 2)
                        cv2.line(frame, (ib_x, ib_y - 12), (ib_x, ib_y + 12), (0, 255, 0), 2)
                        cv2.circle(frame, (ib_x, ib_y), 6, (0, 255, 0), 1)
                        cv2.circle(frame, (ib_x, ib_y), 12, (0, 255, 0), 1)

                        # Transformasi Homografi (Piksel -> Dunia Fisik cm)
                        try:
                            xw, yw = self.homography_engine.project_image_to_world(ub, vb)
                            raw_world = (xw, yw)

                            # Perhitungan Kinematika & Clamping Sumbu Railing
                            sx, sy, (cx, cy), is_clamped = self.kinematics.coordinates_to_steps(xw, yw)
                            clamped_world = (cx, cy)
                            target_steps = (sx, sy)

                            # Label mengambang di samping base point
                            lbl_coord = f"({xw:.1f}, {yw:.1f}) cm -> Step ({sx}, {sy})"
                            cv2.putText(
                                frame,
                                lbl_coord,
                                (ib_x + 15, ib_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                        except Exception as e:
                            logger.error(f"Error kalkulasi homografi: {e}")
                            raw_world = None

                # 3. Tampilkan HUD Telemetri dan Status
                self._draw_hud(frame, raw_world, clamped_world, target_steps, is_clamped)

                # 4. Tampilkan Frame ke Layar
                cv2.imshow(self.window_name, frame)

                # 5. Baca Masukan Keyboard
                key = cv2.waitKey(20) & 0xFF

                # Periksa apakah jendela ditutup via tombol 'X'
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                    logger.info("Pengguna meminta keluar dari aplikasi.")
                    break

                elif key in [ord("g"), ord("G"), 10, 13]:  # G atau ENTER: Kirim GOTO ke ESP32
                    if clamped_world is not None:
                        cx, cy = clamped_world
                        logger.info(f"[USER COMMAND: GOTO] Menembak target ke ({cx:.1f}, {cy:.1f}) cm")
                        self.send_goto_command(cx, cy)
                    else:
                        self.action_status = "Pilih Bounding Box objek terlebih dahulu!"
                        logger.warning(self.action_status)

                elif key in [ord("s"), ord("S")]:  # S: Kirim GOTO + Semprot
                    if clamped_world is not None:
                        cx, cy = clamped_world
                        logger.info(f"[USER COMMAND: SPRAY] Menembak target ke ({cx:.1f}, {cy:.1f}) cm + Semprot")
                        self.send_goto_and_spray_command(cx, cy)
                    else:
                        self.action_status = "Pilih Bounding Box objek terlebih dahulu!"
                        logger.warning(self.action_status)

                elif key in [ord("h"), ord("H")]:  # H: Kembali ke posisi 0 (tanpa switch limit)
                    logger.info("[USER COMMAND: HOME] Mengembalikan nozzle ke Step (0, 0)")
                    self.send_return_zero_command()

                elif key in [ord("c"), ord("C")]:  # C: Reset/Clear Bounding Box
                    self.selector.clear()
                    raw_world = None
                    clamped_world = None
                    target_steps = None
                    self.action_status = "Bounding Box direset. Tarik mouse untuk memilih target baru."
                    logger.info(self.action_status)

                elif key == 32:  # SPASI: Emergency Stop
                    self.trigger_estop()

        except KeyboardInterrupt:
            logger.info("Program dihentikan melalui KeyboardInterrupt.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.coordinator:
                logger.info("Mematikan koordinator dan memutus koneksi MQTT...")
                self.coordinator.stop()
            logger.info("Aplikasi selesai.")


def main():
    args = parse_arguments()
    app = HomographyEndToEndApp(args)
    app.run()


if __name__ == "__main__":
    main()
