"""
=====================================================================
DEEZPRO - TEST HOMOGRAPHY END-TO-END (SINGLE & DUAL CAMERA FUSION)
=====================================================================
File: test_homography_end_to_end.py
Deskripsi: Program uji end-to-end terintegrasi untuk memvalidasi hasil
            kalibrasi homografi kamera terhadap posisi fisik aktual nozzle.
            Mendukung pengujian kamera tunggal (Single Camera) maupun
            pengujian dua kamera sekaligus dalam 1 sesi (Dual-Camera Fusion).

Fitur Utama:
  1. Single-Camera Mode: Uji mandiri Kamera 1 atau Kamera 2.
  2. Dual-Camera Fusion Mode (--dual):
     - Menampilkan feed Kamera 1 (Kiri) dan Kamera 2 (Kanan) secara berdampingan.
     - User dapat menggambar bounding box pada Kamera 1 dan/atau Kamera 2.
     - Proyeksi Silang (Cross-Projection): Menandai estimasi posisi target di kamera lawan.
     - Menghitung galat selisih spasial (Discrepancy delta_d cm) antara kedua kamera.
     - Melakukan Fusi Koordinat Rata-rata (Late Fusion) untuk memperkecil error.
  3. Translasi Kinematika & Clamping area batas aman railing (KinematicsEngine).
  4. Pengiriman perintah pergerakan (GOTO) non-blocking ke ESP32 via MQTT (CentralCoordinator).
  5. Keamanan Fisik: Auto-Homing dinonaktifkan secara ketat demi keselamatan rel mekanik.

Kontrol Keyboard:
  [G] / [ENTER] : Kirim koordinat target (atau fusi) ke ESP32 (GOTO tanpa semprot).
  [S]           : Kirim koordinat target ke ESP32 lalu semprot APAR 2 detik.
  [1]           : Hapus/Reset Bounding Box Kamera 1 saja (Dual Mode).
  [2]           : Hapus/Reset Bounding Box Kamera 2 saja (Dual Mode).
  [C]           : Hapus/Reset seluruh Bounding Box.
  [SPACE]       : EMERGENCY STOP (Hentikan seluruh motor seketika).
  [Q] / [ESC]   : Keluar dari program.
"""

import os
import sys
import time
import argparse
import threading
from typing import Tuple, Optional, Dict, Any, List
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
        description="Deezpro End-to-End Homography Validation & Hardware Dispatch (Single & Dual Camera)."
    )
    # Mode Operasi
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Aktifkan mode Dual-Camera (menampilkan Kamera 1 dan Kamera 2 berdampingan dalam 1 sesi).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Nomor logika kamera untuk Single Mode: 1 untuk Kamera 1, 2 untuk Kamera 2 (default: 1).",
    )
    # Konfigurasi Hardware Index Kamera
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Override index hardware OpenCV untuk kamera utama / Kamera 1.",
    )
    parser.add_argument(
        "--cam1",
        type=int,
        default=None,
        help="Override index hardware OpenCV khusus Kamera 1 (Dual Mode).",
    )
    parser.add_argument(
        "--cam2",
        type=int,
        default=None,
        help="Override index hardware OpenCV khusus Kamera 2 (Dual Mode).",
    )
    # Berkas Kalibrasi & Konfigurasi
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path berkas matriks kalibrasi lensa (.npz) untuk Single Mode.",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=None,
        help="Path berkas matriks homografi (.npz) untuk Single Mode.",
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
    # Resolusi & Frame Rate
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Resolusi lebar tangkapan kamera individual (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Resolusi tinggi tangkapan kamera individual (default: 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate kamera (default: 30 FPS).",
    )
    return parser.parse_args()


class ThreadedCamera:
    """Pengambil frame video asinkron menggunakan threading untuk menghindari latensi USB."""

    def __init__(self, cam_index: int, width: int = 1280, height: int = 720, fps: int = 30, label: str = "Camera"):
        self.cam_index = cam_index
        self.label = label
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Gagal membuka {label} (Device Index {cam_index})! Pastikan webcam terhubung.")

        # Atur format MJPG agar tidak membebani USB bandwidth
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        logger.info(f"{label} aktif (Index {cam_index}): {self.actual_w}x{self.actual_h}")

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


class ManualBBoxSelector:
    """Mengelola interaksi mouse untuk menggambar bounding box pada satu kamera."""

    def __init__(self, target_mode: str = "base", label: str = "Cam"):
        self.target_mode = target_mode
        self.label = label
        self.is_drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.curr_pt: Optional[Tuple[int, int]] = None
        self.bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self.target_pixel: Optional[Tuple[float, float]] = None  # (u, v)
        self.centroid_pixel: Optional[Tuple[float, float]] = None

    def handle_mouse(self, event, x: int, y: int):
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

                    if (x_max - x_min) >= 6 and (y_max - y_min) >= 6:
                        self.bbox = (x_min, y_min, x_max, y_max)
                        u_c = (x_min + x_max) / 2.0
                        v_c = (y_min + y_max) / 2.0
                        self.centroid_pixel = (u_c, v_c)

                        if self.target_mode == "base":
                            u_b = (x_min + x_max) / 2.0
                            v_b = float(y_max)
                            self.target_pixel = (u_b, v_b)
                        else:
                            self.target_pixel = (u_c, v_c)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear()

    def clear(self):
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
        self.cfg_dir = os.path.join(current_dir, "config")
        self.config_path = args.config or os.path.join(self.cfg_dir, "system_config.yaml")

        # 1. Muat Konfigurasi Sistem
        self.config = self._load_system_config()

        # 2. Tentukan Mode Operasi (Dual vs Single)
        self.is_dual_mode = bool(args.dual)

        # 3. Inisialisasi Kinematics Engine
        self.kinematics = KinematicsEngine.from_config(self.config)

        # 4. Inisialisasi Kamera & Engine
        if self.is_dual_mode:
            self._init_dual_mode()
        else:
            self._init_single_mode()

        # 5. Inisialisasi Central Coordinator (MQTT)
        self.coordinator: Optional[CentralCoordinator] = None
        self.mqtt_connected = False
        if not args.no_mqtt:
            self._init_coordinator()

        # 6. Status Aksi & UI
        self.window_name = (
            "Deezpro - End-to-End Homography Test [DUAL CAMERA FUSION]"
            if self.is_dual_mode
            else f"Deezpro - End-to-End Homography Test [KAMERA {self.cam_id}]"
        )
        self.action_status = "SIAP (Tarik mouse pada objek lantai untuk membuat Bounding Box)"
        self.is_action_busy = False

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

    def _load_undistort_maps(self, calib_file: str, width: int, height: int):
        """Memuat matriks distorsi intrinsik dan menyiapkan remap map."""
        if not os.path.exists(calib_file):
            alt_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                "Development", "callibration", os.path.basename(calib_file)
            )
            if os.path.exists(alt_path):
                calib_file = alt_path

        if os.path.exists(calib_file):
            try:
                data = np.load(calib_file)
                K = data["camera_matrix"]
                if "dist_coeffs" in data:
                    D = data["dist_coeffs"]
                elif "dist_coeff" in data:
                    D = data["dist_coeff"]
                elif "D" in data:
                    D = data["D"]
                else:
                    D = np.zeros(5)

                new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha=0.0, newImgSize=(width, height))
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
                logger.info(f"Kalibrasi lensa dimuat: {os.path.basename(calib_file)} (Undistort Aktif)")
                return map1, map2
            except Exception as e:
                logger.error(f"Gagal menginisialisasi koreksi distorsi {calib_file}: {e}")
        else:
            logger.warning(f"Berkas kalibrasi lensa ({calib_file}) tidak ditemukan! Berjalan tanpa undistort.")
        return None, None

    def _load_homography_engine(self, filepath: str, cam_id: int) -> Tuple[HomographyEngine, bool]:
        """Memuat matriks homografi dari berkas atau membuat matriks sintetis sementara."""
        if os.path.exists(filepath):
            try:
                engine = HomographyEngine.load(filepath)
                logger.info(
                    f"Matriks homografi Kamera {cam_id} dimuat: {os.path.basename(filepath)} "
                    f"(Reproj error: {engine.reprojection_error_cm:.2f} cm)"
                )
                return engine, False
            except Exception as e:
                logger.error(f"Gagal memuat berkas homografi {filepath}: {e}")

        logger.warning(
            f"[PERINGATAN] Berkas homografi '{filepath}' belum ditemukan! "
            f"Membuat matriks aproksimasi default untuk Kamera {cam_id}."
        )
        engine = HomographyEngine()
        w, h = float(self.args.width), float(self.args.height)
        if cam_id == 1:
            engine.add_point_pair((0.15 * w, 0.85 * h), (10.0, 10.0))
            engine.add_point_pair((0.85 * w, 0.85 * h), (190.0, 10.0))
            engine.add_point_pair((0.70 * w, 0.30 * h), (190.0, 190.0))
            engine.add_point_pair((0.30 * w, 0.30 * h), (10.0, 190.0))
        else:
            engine.add_point_pair((0.85 * w, 0.85 * h), (190.0, 10.0))
            engine.add_point_pair((0.15 * w, 0.85 * h), (10.0, 10.0))
            engine.add_point_pair((0.30 * w, 0.30 * h), (10.0, 190.0))
            engine.add_point_pair((0.70 * w, 0.30 * h), (190.0, 190.0))
        engine.compute()
        return engine, True

    def _init_single_mode(self):
        """Inisialisasi mode pengujian kamera tunggal."""
        self.cam_id = self.args.camera
        cameras_cfg = self.config.get("cameras", {})
        cam_key = f"cam{self.cam_id}"

        # Resolve index
        if self.args.source is not None:
            self.dev_idx = self.args.source
        elif cam_key in cameras_cfg and "index" in cameras_cfg[cam_key]:
            self.dev_idx = int(cameras_cfg[cam_key]["index"])
        else:
            self.dev_idx = 0

        calib_f = self.args.calib or os.path.join(self.cfg_dir, f"calib_cam{self.cam_id}.npz")
        homo_f = self.args.homography or os.path.join(self.cfg_dir, f"homography_cam{self.cam_id}.npz")

        self.map1, self.map2 = None, None
        if not self.args.no_undistort:
            self.map1, self.map2 = self._load_undistort_maps(calib_f, self.args.width, self.args.height)

        self.homography_engine, self.is_synthetic = self._load_homography_engine(homo_f, self.cam_id)
        self.selector = ManualBBoxSelector(target_mode=self.args.target_mode, label=f"Cam {self.cam_id}")

    def _init_dual_mode(self):
        """Inisialisasi mode pengujian dual kamera sekaligus (Kamera 1 & Kamera 2)."""
        cameras_cfg = self.config.get("cameras", {})

        # Resolve index Cam 1
        if self.args.cam1 is not None:
            self.dev_idx1 = self.args.cam1
        elif self.args.source is not None:
            self.dev_idx1 = self.args.source
        elif "cam1" in cameras_cfg and "index" in cameras_cfg["cam1"]:
            self.dev_idx1 = int(cameras_cfg["cam1"]["index"])
        else:
            self.dev_idx1 = 0

        # Resolve index Cam 2
        if self.args.cam2 is not None:
            self.dev_idx2 = self.args.cam2
        elif "cam2" in cameras_cfg and "index" in cameras_cfg["cam2"]:
            self.dev_idx2 = int(cameras_cfg["cam2"]["index"])
        else:
            self.dev_idx2 = 1

        calib_f1 = os.path.join(self.cfg_dir, "calib_cam1.npz")
        calib_f2 = os.path.join(self.cfg_dir, "calib_cam2.npz")
        homo_f1 = os.path.join(self.cfg_dir, "homography_cam1.npz")
        homo_f2 = os.path.join(self.cfg_dir, "homography_cam2.npz")

        self.map1_c1, self.map2_c1 = None, None
        self.map1_c2, self.map2_c2 = None, None
        if not self.args.no_undistort:
            self.map1_c1, self.map2_c1 = self._load_undistort_maps(calib_f1, self.args.width, self.args.height)
            self.map1_c2, self.map2_c2 = self._load_undistort_maps(calib_f2, self.args.width, self.args.height)

        self.homo1, self.syn1 = self._load_homography_engine(homo_f1, 1)
        self.homo2, self.syn2 = self._load_homography_engine(homo_f2, 2)

        self.selector_cam1 = ManualBBoxSelector(target_mode=self.args.target_mode, label="Kamera 1")
        self.selector_cam2 = ManualBBoxSelector(target_mode=self.args.target_mode, label="Kamera 2")

    def _init_coordinator(self):
        """Menghubungkan CentralCoordinator ke MQTT Broker tanpa auto-homing."""
        try:
            logger.info("Menghubungkan Central Coordinator ke MQTT Broker...")
            self.coordinator = CentralCoordinator(self.config)
            # auto_home=False demi keamanan mekanis limit switch
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
        if self.is_action_busy:
            logger.warning("Perintah sebelumnya masih berjalan! Harap tunggu.")
            return

        self.is_action_busy = True

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

    def send_goto_command(self, x_cm: float, y_cm: float, source_label: str = "TARGET"):
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}] (MQTT Offline)"
            logger.info(self.action_status)
            return

        def _goto():
            self.action_status = f"Menggerakkan nozzle ke ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}]..."
            logger.info(self.action_status)
            success = self.coordinator.manual_goto_cm(x_cm, y_cm)
            if success:
                self.action_status = f"Nozzle TIBA di koordinat ({x_cm:.1f}, {y_cm:.1f}) cm!"
                logger.info(self.action_status)
            else:
                self.action_status = "Pergerakan timeout atau dibatalkan!"
                logger.warning(self.action_status)

        self._execute_async_action(_goto)

    def send_goto_and_spray_command(self, x_cm: float, y_cm: float, duration_ms: int = 2000, source_label: str = "TARGET"):
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO ({x_cm:.1f}, {y_cm:.1f}) cm + SPRAY [{source_label}]"
            logger.info(self.action_status)
            return

        def _goto_spray():
            self.action_status = f"Menggerakkan nozzle ke ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}]..."
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

    def trigger_estop(self):
        self.action_status = "EMERGENCY STOP DIAKTIFKAN!"
        logger.critical("EMERGENCY STOP dipicu oleh pengguna!")
        if self.coordinator:
            self.coordinator.trigger_emergency_stop()

    # --- Logika Menggambar Anotasi Kamera ---
    def _draw_camera_annotations(
        self,
        frame: np.ndarray,
        selector: ManualBBoxSelector,
        engine: HomographyEngine,
        label: str,
        cross_guide_world: Optional[Tuple[float, float]] = None,
        cross_guide_label: str = "",
    ) -> Optional[Tuple[float, float]]:
        """Menggambar bounding box, target base contact point, dan proyeksi silang kamera."""
        # 1. Gambar Bounding Box Saat Dragging
        if selector.is_drawing and selector.start_pt and selector.curr_pt:
            cv2.rectangle(frame, selector.start_pt, selector.curr_pt, (255, 255, 0), 1)

        raw_world = None

        # 2. Gambar Bounding Box yang Terpilih
        if selector.bbox is not None:
            x1, y1, x2, y2 = selector.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(
                frame,
                f"{label} BBOX [{x2-x1}x{y2-y1}]",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 140, 255),
                1,
            )

            # Titik Centroid
            if selector.centroid_pixel is not None:
                uc, vc = int(selector.centroid_pixel[0]), int(selector.centroid_pixel[1])
                cv2.circle(frame, (uc, vc), 3, (255, 200, 0), -1)

            # Titik Kontak Dasar Lantai
            if selector.target_pixel is not None:
                ub, vb = selector.target_pixel
                ib_x, ib_y = int(round(ub)), int(round(vb))

                if selector.centroid_pixel is not None:
                    cv2.line(frame, (uc, vc), (ib_x, ib_y), (0, 255, 255), 1)

                cv2.line(frame, (ib_x - 12, ib_y), (ib_x + 12, ib_y), (0, 255, 0), 2)
                cv2.line(frame, (ib_x, ib_y - 12), (ib_x, ib_y + 12), (0, 255, 0), 2)
                cv2.circle(frame, (ib_x, ib_y), 6, (0, 255, 0), 1)
                cv2.circle(frame, (ib_x, ib_y), 12, (0, 255, 0), 1)

                # Transformasi ke Dunia Fisik
                try:
                    xw, yw = engine.project_image_to_world(ub, vb)
                    raw_world = (xw, yw)
                    lbl_coord = f"({xw:.1f}, {yw:.1f}) cm"
                    cv2.putText(
                        frame,
                        lbl_coord,
                        (ib_x + 15, ib_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                except Exception:
                    pass

        # 3. Gambar Cross-Projection Guide Point dari Kamera Lawan
        if cross_guide_world is not None and engine.H_inv is not None:
            try:
                gx, gy = cross_guide_world
                u_proj, v_proj = engine.project_world_to_image(gx, gy)
                px, py = int(round(u_proj)), int(round(v_proj))
                h, w = frame.shape[:2]
                if 0 <= px < w and 0 <= py < h:
                    cv2.drawMarker(frame, (px, py), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
                    cv2.circle(frame, (px, py), 10, (255, 255, 0), 1)
                    cv2.putText(
                        frame,
                        f"Target {cross_guide_label} ({gx:.1f},{gy:.1f})",
                        (px + 12, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 0),
                        1,
                    )
            except Exception:
                pass

        # Label Banner Sudut Kamera
        cv2.rectangle(frame, (10, 10), (190, 36), (30, 30, 30), -1)
        cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return raw_world

    # --- Main Loop: Single-Camera Mode ---
    def run_single(self):
        logger.info(f"Menjalankan Single-Camera Mode pada device index {self.dev_idx}...")
        try:
            cam = ThreadedCamera(
                self.dev_idx,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
                label=f"Kamera {self.cam_id}"
            )
        except Exception as e:
            logger.error(f"Gagal membuka kamera: {e}")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.args.width, self.args.height)

        def on_mouse(event, x, y, flags, param):
            self.selector.handle_mouse(event, x, y)

        cv2.setMouseCallback(self.window_name, on_mouse)

        print("\n" + "=" * 70)
        print(f" DEEZPRO HOMOGRAPHY VALIDATOR - SINGLE CAMERA (CAM {self.cam_id})")
        print("=" * 70)
        print(" [Drag Mouse] Tarik Bounding Box objek di lantai.")
        print(" [G / ENTER]  Kirim perintah GOTO ke ESP32.")
        print(" [S]          Kirim perintah GOTO + SPRAY (2 detik).")
        print(" [C]          Reset Bounding Box.")
        print(" [SPACE]      Emergency Stop (ESTOP).")
        print(" [Q / ESC]    Keluar.")
        print("=" * 70 + "\n")

        try:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                if self.map1 is not None and self.map2 is not None:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                raw_world = self._draw_camera_annotations(
                    frame, self.selector, self.homography_engine, f"Kamera {self.cam_id}"
                )

                clamped_world = None
                target_steps = None
                is_clamped = False
                if raw_world is not None:
                    sx, sy, (cx, cy), is_clamped = self.kinematics.coordinates_to_steps(raw_world[0], raw_world[1])
                    clamped_world = (cx, cy)
                    target_steps = (sx, sy)

                # Render HUD Single
                self._draw_single_hud(frame, raw_world, clamped_world, target_steps, is_clamped)

                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(20) & 0xFF

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in [ord("q"), ord("Q"), 27]:
                    break
                elif key in [ord("g"), ord("G"), 10, 13]:
                    if clamped_world is not None:
                        self.send_goto_command(clamped_world[0], clamped_world[1])
                    else:
                        self.action_status = "Pilih Bounding Box objek terlebih dahulu!"
                elif key in [ord("s"), ord("S")]:
                    if clamped_world is not None:
                        self.send_goto_and_spray_command(clamped_world[0], clamped_world[1])
                    else:
                        self.action_status = "Pilih Bounding Box objek terlebih dahulu!"
                elif key in [ord("c"), ord("C")]:
                    self.selector.clear()
                    self.action_status = "Bounding Box direset. Tarik mouse untuk memilih target baru."
                elif key == 32:
                    self.trigger_estop()

        finally:
            cam.release()
            cv2.destroyAllWindows()
            if self.coordinator:
                self.coordinator.stop()

    # --- Main Loop: Dual-Camera Mode ---
    def run_dual(self):
        logger.info(f"Menjalankan Dual-Camera Mode (Cam1: Index {self.dev_idx1}, Cam2: Index {self.dev_idx2})...")
        cam1, cam2 = None, None
        try:
            cam1 = ThreadedCamera(
                self.dev_idx1,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
                label="Kamera 1"
            )
        except Exception as e:
            logger.error(f"Gagal membuka Kamera 1: {e}")
            return

        try:
            cam2 = ThreadedCamera(
                self.dev_idx2,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
                label="Kamera 2"
            )
        except Exception as e:
            logger.warning(f"Gagal membuka Kamera 2: {e}. Beralih ke Single-Camera Mode.")
            cam1.release()
            self.cam_id = 1
            self.is_dual_mode = False
            self._init_single_mode()
            self.run_single()
            return

        # Target ukuran pratinjau per kamera agar muat di layar horizontal (misal 800x450)
        view_w, view_h = 800, 450
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, view_w * 2, view_h + 160)

        def on_mouse_dual(event, x, y, flags, param):
            # Jika di bagian atas (area video)
            if y < view_h:
                if x < view_w:
                    scale_x = float(self.args.width) / view_w
                    scale_y = float(self.args.height) / view_h
                    self.selector_cam1.handle_mouse(event, int(x * scale_x), int(y * scale_y))
                else:
                    scale_x = float(self.args.width) / view_w
                    scale_y = float(self.args.height) / view_h
                    local_x = x - view_w
                    self.selector_cam2.handle_mouse(event, int(local_x * scale_x), int(y * scale_y))

        cv2.setMouseCallback(self.window_name, on_mouse_dual)

        print("\n" + "=" * 75)
        print(" DEEZPRO HOMOGRAPHY VALIDATOR - DUAL CAMERA FUSION MODE")
        print("=" * 75)
        print(" [Drag Mouse Kiri]  Tarik BBox pada feed Kamera 1 (Kiri).")
        print(" [Drag Mouse Kanan] Tarik BBox pada feed Kamera 2 (Kanan).")
        print(" [G / ENTER]        Kirim koordinat FUSI (atau single) ke ESP32.")
        print(" [S]                Kirim koordinat FUSI + SPRAY (2 detik).")
        print(" [1]                Reset BBox Kamera 1 saja.")
        print(" [2]                Reset BBox Kamera 2 saja.")
        print(" [C]                Reset kedua BBox.")
        print(" [SPACE]            Emergency Stop (ESTOP).")
        print(" [Q / ESC]          Keluar.")
        print("=" * 75 + "\n")

        try:
            while True:
                ret1, f1 = cam1.read()
                ret2, f2 = cam2.read()

                if not ret1 or f1 is None or not ret2 or f2 is None:
                    time.sleep(0.01)
                    continue

                # Undistort Kamera 1 & 2
                if self.map1_c1 is not None and self.map2_c1 is not None:
                    f1 = cv2.remap(f1, self.map1_c1, self.map2_c1, cv2.INTER_LINEAR)
                if self.map1_c2 is not None and self.map2_c2 is not None:
                    f2 = cv2.remap(f2, self.map1_c2, self.map2_c2, cv2.INTER_LINEAR)

                # Evaluasi titik dunia awal untuk cross-projection
                w1_est = None
                if self.selector_cam1.target_pixel is not None:
                    try:
                        w1_est = self.homo1.project_image_to_world(*self.selector_cam1.target_pixel)
                    except Exception:
                        pass

                w2_est = None
                if self.selector_cam2.target_pixel is not None:
                    try:
                        w2_est = self.homo2.project_image_to_world(*self.selector_cam2.target_pixel)
                    except Exception:
                        pass

                # Anotasi pada masing-masing kamera
                raw_w1 = self._draw_camera_annotations(
                    f1, self.selector_cam1, self.homo1, "KAMERA 1",
                    cross_guide_world=w2_est, cross_guide_label="Cam 2"
                )
                raw_w2 = self._draw_camera_annotations(
                    f2, self.selector_cam2, self.homo2, "KAMERA 2",
                    cross_guide_world=w1_est, cross_guide_label="Cam 1"
                )

                # Resize ke tampilan panel berdampingan
                f1_res = cv2.resize(f1, (view_w, view_h))
                f2_res = cv2.resize(f2, (view_w, view_h))

                # Logika Fusi & Evaluasi Galat Spasial
                active_target_cm = None
                target_mode_desc = "TIDAK ADA TARGET"
                discrepancy_cm = None
                consistency_status = "N/A"

                if raw_w1 is not None and raw_w2 is not None:
                    # KEDUA KAMERA MENDETEKSI: LATE FUSION (RATA-RATA KOORDINAT)
                    dist = np.sqrt((raw_w1[0] - raw_w2[0]) ** 2 + (raw_w1[1] - raw_w2[1]) ** 2)
                    discrepancy_cm = float(dist)

                    fused_x = (raw_w1[0] + raw_w2[0]) / 2.0
                    fused_y = (raw_w1[1] + raw_w2[1]) / 2.0
                    active_target_cm = (fused_x, fused_y)
                    target_mode_desc = "DUAL-CAMERA FUSION (Rata-rata)"

                    if discrepancy_cm <= 15.0:
                        consistency_status = "SANGAT TINGGI (Selisih <= 15 cm)"
                    elif discrepancy_cm <= 30.0:
                        consistency_status = "MODERAT (Selisih <= 30 cm)"
                    else:
                        consistency_status = "PERBEDAAN TINGGI (Cek Kalibrasi!)"

                elif raw_w1 is not None:
                    active_target_cm = raw_w1
                    target_mode_desc = "SINGLE-CAMERA 1 (Fallback)"
                elif raw_w2 is not None:
                    active_target_cm = raw_w2
                    target_mode_desc = "SINGLE-CAMERA 2 (Fallback)"

                # Kinematika & Clamping
                clamped_cm = None
                target_steps = None
                is_clamped = False
                if active_target_cm is not None:
                    sx, sy, (cx, cy), is_clamped = self.kinematics.coordinates_to_steps(
                        active_target_cm[0], active_target_cm[1]
                    )
                    clamped_cm = (cx, cy)
                    target_steps = (sx, sy)

                # Gabungkan feed video secara horizontal
                combined_feed = np.hstack([f1_res, f2_res])
                cv2.line(combined_feed, (view_w, 0), (view_w, view_h), (80, 80, 80), 2)

                # Bangun Panel Telemetri Fusi di Bawah
                hud_panel = self._create_dual_hud_panel(
                    view_w * 2, 160,
                    raw_w1, raw_w2,
                    active_target_cm, clamped_cm, target_steps,
                    discrepancy_cm, consistency_status, target_mode_desc, is_clamped
                )

                # Gabungkan feed atas dan panel bawah
                full_canvas = np.vstack([combined_feed, hud_panel])

                cv2.imshow(self.window_name, full_canvas)
                key = cv2.waitKey(20) & 0xFF

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in [ord("q"), ord("Q"), 27]:
                    break
                elif key in [ord("g"), ord("G"), 10, 13]:
                    if clamped_cm is not None:
                        self.send_goto_command(clamped_cm[0], clamped_cm[1], source_label=target_mode_desc)
                    else:
                        self.action_status = "Pilih Bounding Box pada salah satu atau kedua kamera terlebih dahulu!"
                elif key in [ord("s"), ord("S")]:
                    if clamped_cm is not None:
                        self.send_goto_and_spray_command(clamped_cm[0], clamped_cm[1], source_label=target_mode_desc)
                    else:
                        self.action_status = "Pilih Bounding Box pada kamera terlebih dahulu!"
                elif key == ord("1"):
                    self.selector_cam1.clear()
                    self.action_status = "Bounding Box Kamera 1 direset."
                elif key == ord("2"):
                    self.selector_cam2.clear()
                    self.action_status = "Bounding Box Kamera 2 direset."
                elif key in [ord("c"), ord("C")]:
                    self.selector_cam1.clear()
                    self.selector_cam2.clear()
                    self.action_status = "Seluruh Bounding Box direset."
                elif key == 32:
                    self.trigger_estop()

        finally:
            if cam1: cam1.release()
            if cam2: cam2.release()
            cv2.destroyAllWindows()
            if self.coordinator:
                self.coordinator.stop()

    # --- Render Panel HUD Dual-Camera ---
    def _create_dual_hud_panel(
        self,
        width: int,
        height: int,
        w1: Optional[Tuple[float, float]],
        w2: Optional[Tuple[float, float]],
        active_target: Optional[Tuple[float, float]],
        clamped_target: Optional[Tuple[float, float]],
        target_steps: Optional[Tuple[int, int]],
        discrepancy: Optional[float],
        consistency: str,
        target_mode: str,
        is_clamped: bool,
    ) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        cv2.line(panel, (0, 0), (width, 0), (100, 100, 100), 1)

        # Kolom 1: Hasil Masing-Masing Kamera
        cv2.putText(panel, "[1] HASIL PER KAMERA", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        c1_str = f"Kamera 1 : ({w1[0]:.1f}, {w1[1]:.1f}) cm" if w1 else "Kamera 1 : (Belum dipilih)"
        c2_str = f"Kamera 2 : ({w2[0]:.1f}, {w2[1]:.1f}) cm" if w2 else "Kamera 2 : (Belum dipilih)"
        cv2.putText(panel, c1_str, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)
        cv2.putText(panel, c2_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        if discrepancy is not None:
            disc_col = (0, 255, 0) if discrepancy <= 15.0 else ((0, 165, 255) if discrepancy <= 30.0 else (0, 0, 255))
            cv2.putText(panel, f"Selisih : {discrepancy:.1f} cm [{consistency}]", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.44, disc_col, 1)
        else:
            cv2.putText(panel, "Selisih : (Pilih kedua kamera untuk evaluasi fusi)", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

        # Garis pemisah kolom 1 ke 2
        cv2.line(panel, (width // 3 + 20, 10), (width // 3 + 20, height - 35), (50, 50, 50), 1)

        # Kolom 2: Fusi & Target Kinematika
        x_col2 = width // 3 + 40
        cv2.putText(panel, "[2] FUSI & TRANSLASI MOTOR", (x_col2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        cv2.putText(panel, f"Mode Target : {target_mode}", (x_col2, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 128), 1)

        if active_target and clamped_target and target_steps:
            clamp_b = "[CLAMPED!]" if is_clamped else "[OK]"
            clamp_col = (0, 165, 255) if is_clamped else (100, 255, 100)
            cv2.putText(panel, f"Koordinat Clamped: ({clamped_target[0]:.1f}, {clamped_target[1]:.1f}) cm {clamp_b}", (x_col2, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, clamp_col, 1)
            cv2.putText(panel, f"Langkah Motor    : Step X={target_steps[0]}, Step Y={target_steps[1]}", (x_col2, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        else:
            cv2.putText(panel, "Tarik kotak pada minimal 1 kamera untuk target.", (x_col2, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

        # Garis pemisah kolom 2 ke 3
        cv2.line(panel, ((width // 3) * 2 + 40, 10), ((width // 3) * 2 + 40, height - 35), (50, 50, 50), 1)

        # Kolom 3: Telemetri ESP32 & Status Sistem
        x_col3 = (width // 3) * 2 + 60
        cv2.putText(panel, "[3] STATUS HARDWARE ESP32", (x_col3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        mqtt_text = "MQTT: ONLINE" if self.mqtt_connected else "MQTT: OFFLINE (DRY-RUN)"
        mqtt_color = (0, 230, 0) if self.mqtt_connected else (0, 165, 255)
        cv2.putText(panel, mqtt_text, (x_col3, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mqtt_color, 1)

        if self.coordinator:
            try:
                tele = self.coordinator.get_system_telemetry()
                cx = tele["current_position_steps"]["x"]
                cy = tele["current_position_steps"]["y"]
                pos_cm = tele["current_position_cm"]
                cv2.putText(panel, f"Posisi ESP32: ({cx}, {cy}) steps ~ ({pos_cm['x']:.1f}, {pos_cm['y']:.1f}) cm", (x_col3, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            except Exception:
                pass

        cv2.putText(panel, f"Status: {self.action_status}", (x_col3, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Banner Bawah: Petunjuk Tombol
        cv2.rectangle(panel, (0, height - 28), (width, height), (10, 10, 10), -1)
        help_t = (
            "[Drag Kiri: Cam 1]  [Drag Kanan: Cam 2]  |  [G / ENTER] Kirim Target Fusi  |  "
            "[S] Kirim + Semprot  |  [1] Reset Cam 1  |  [2] Reset Cam 2  |  [C] Reset Semua  |  "
            "[SPACE] ESTOP  |  [Q] Keluar"
        )
        cv2.putText(panel, help_t, (20, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        return panel

    # --- Render HUD Single Mode ---
    def _draw_single_hud(self, frame, raw_world, clamped_world, target_steps, is_clamped):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.putText(frame, f"DEEZPRO HOMOGRAPHY TEST [KAMERA {self.cam_id}]", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        mqtt_text = "MQTT: ONLINE" if self.mqtt_connected else "MQTT: OFFLINE (DRY-RUN)"
        mqtt_color = (0, 230, 0) if self.mqtt_connected else (0, 165, 255)
        cv2.putText(frame, mqtt_text, (w - 280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mqtt_color, 2)

        err_str = f"Reproj Error: {self.homography_engine.reprojection_error_cm:.2f} cm"
        cv2.putText(frame, f"[H: TERKALIBRASI] {err_str}", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
        cv2.putText(frame, f"Status: {self.action_status}", (w - 600, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Panel Kiri Bawah
        panel_w = 420
        panel_h = 145
        panel_x = 15
        panel_y = h - panel_h - 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

        cv2.putText(frame, "TARGET COORDINATE DATA", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        if raw_world and clamped_world and target_steps:
            u, v = self.selector.target_pixel
            rx, ry = raw_world
            cx, cy = clamped_world
            sx, sy = target_steps
            cv2.putText(frame, f"1. Base Pixel (u, v) : ({u:.1f}, {v:.1f}) px", (panel_x + 10, panel_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)
            cv2.putText(frame, f"2. Homography World  : ({rx:.1f}, {ry:.1f}) cm", (panel_x + 10, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1)
            clamp_b = "[CLAMPED!]" if is_clamped else "[OK]"
            clamp_c = (0, 165, 255) if is_clamped else (100, 255, 100)
            cv2.putText(frame, f"3. Clamped Target    : ({cx:.1f}, {cy:.1f}) cm {clamp_b}", (panel_x + 10, panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, clamp_c, 1)
            cv2.putText(frame, f"4. Target Motor Step : Step X={sx}, Step Y={sy}", (panel_x + 10, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Belum ada target yang dipilih.", (panel_x + 10, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
            cv2.putText(frame, "Tarik mouse pada objek lantai untuk kalkulasi.", (panel_x + 10, panel_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

        cv2.rectangle(frame, (0, h - 32), (w, h), (10, 10, 10), -1)
        help_text = "[Drag Mouse] Buat Box  |  [G / ENTER] Kirim GOTO  |  [S] GOTO + Semprot  |  [C] Reset Box  |  [SPACE] ESTOP  |  [Q] Keluar"
        cv2.putText(frame, help_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

    def run(self):
        if self.is_dual_mode:
            self.run_dual()
        else:
            self.run_single()


def main():
    args = parse_arguments()
    app = HomographyEndToEndApp(args)
    app.run()


if __name__ == "__main__":
    main()
