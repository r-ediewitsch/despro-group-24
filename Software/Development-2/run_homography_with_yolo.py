"""
=====================================================================
DEEZPRO - AUTO-DETECTION HOMOGRAPHY END-TO-END (SINGLE & DUAL CAMERA)
=====================================================================
File: run_homography_with_yolo.py
Deskripsi: Program uji end-to-end terintegrasi otomatis di mana Bounding Box
            diperoleh secara REAL-TIME dari model YOLO (bukan input manual mouse).
            Mendukung Single-Camera maupun Dual-Camera Late Fusion, penghitungan
            koordinat fisik (X, Y cm) via Homografi, pembatasan rail aman
            (Kinematics Clamping), serta pengiriman perintah ke ESP32.

Fitur Utama:
  1. Auto Bounding Box dari YOLO:
     - Mendeteksi 'fire' dan 'smoke' secara otomatis.
     - Mengambil target api dengan confidence tertinggi.
     - Titik kontak acuan: 'base' (dasar api menyentuh lantai) atau 'centroid'.
  2. Single & Dual Camera Fusion:
     - Proyeksi otomatis dari piksel ke koordinat dunia nyata (X, Y cm).
     - Jika kedua kamera mendeteksi api, dilakukan Late Fusion (rata-rata)
       dan evaluasi galat spasial (Discrepancy delta_d cm).
  3. Visualisasi Lengkap:
     - Tampilan feed kamera dengan overlay BBox YOLO, titik base/centroid,
       proyeksi silang (Cross-Projection), dan panel telemetri status.
  4. Integrasi Hardware & Safety:
     - Pengiriman perintah [G] (GOTO) dan [S] (GOTO + SPRAY) ke ESP32 via MQTT.
     - Emergency Stop [SPACE] instan.
     - Auto-Homing dinonaktifkan secara ketat demi keselamatan mekanis gantry.

Kontrol Keyboard:
  [G] / [ENTER] : Kirim koordinat target otomatis ke ESP32 (GOTO).
  [S]           : Kirim koordinat target otomatis ke ESP32 + SPRAY APAR (2s).
  [M]           : Toggle titik target: 'base' (dasar api) <-> 'centroid' (tengah).
  [SPACE]       : EMERGENCY STOP (Hentikan seluruh motor seketika).
  [Q] / [ESC]   : Keluar dari program.
"""

import os
import sys
import time
import argparse
import threading
from collections import deque
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vision.detector import FireSmokeDetector, PixelDetection
from src.vision.homography import HomographyEngine
from src.integration.kinematics import KinematicsEngine
from src.integration.coordinator import CentralCoordinator, SystemState
from src.integration.system_logger import setup_logger

logger = setup_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Deezpro Auto YOLO End-to-End Homography Validation & Hardware Dispatch."
    )
    # Mode Operasi
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Aktifkan mode Dual-Camera (menampilkan Kamera 1 dan Kamera 2 berdampingan).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Nomor logika kamera untuk Single Mode: 1 untuk Kamera 1, 2 untuk Kamera 2 (default: 1).",
    )
    # Model YOLO & Deteksi
    parser.add_argument(
        "--model",
        type=str,
        default="models/best-v8m.pt",
        help="Path berkas bobot model YOLO (.pt). Default: models/best-v8m.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold deteksi YOLO (default: 0.40).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Resolusi input inferensi YOLO (default: 800).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["base", "centroid"],
        default="base",
        help="Titik acuan proyeksi: 'base' (kontak dasar lantai) atau 'centroid' (tengah). Default: base.",
    )
    # Temporal Buffer & Stabilizer
    parser.add_argument(
        "--buffer-window",
        type=int,
        default=30,
        help="Jumlah frame pada sliding window buffer (default: 30 frame ~1.0 detik @ 30 FPS).",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.50,
        help="Persentase minimum deteksi dalam window untuk status TARGET_LOCKED (default: 0.50 = 50%%).",
    )
    parser.add_argument(
        "--spatial-radius",
        type=float,
        default=25.0,
        help="Radius maksimum (cm) untuk inlier spatial filtering di sekitar median titik api (default: 25.0 cm).",
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
        "--alpha",
        type=float,
        default=1.0,
        help="Parameter free scaling alpha OpenCV undistort (0.0=crop, 1.0=full FoV tanpa crop). Default: 1.0",
    )
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
    # Resolusi & Frame Rate Tangkapan
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Resolusi lebar tangkapan kamera individual (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Resolusi tinggi tangkapan kamera individual (default: 1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate kamera (default: 30 FPS).",
    )
    return parser.parse_args()


class ThreadedCamera:
    """Pengambil frame video asinkron menggunakan threading untuk menghindari bottleneck buffer USB."""

    def __init__(self, cam_index: int, width: int = 1280, height: int = 720, fps: int = 30, label: str = "Camera"):
        self.cam_index = cam_index
        self.label = label
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Gagal membuka {label} (Device Index {cam_index})! Pastikan webcam terhubung.")

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
        time.sleep(0.05)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info(f"{self.label} dimatikan.")


class TargetStabilizer:
    """
    Buffer Temporal & Stabilizer Spasial untuk Deteksi Api:
    1. Detection Rate Window (Percentage buffer):
       Menghitung persentase frame positif dalam N frame terakhir (default: 30 frame ~1.0s).
       Mencegah false positive dari refleksi sekejap, sekaligus toleran terhadap asap/flicker.
    2. Spatial Inlier Averaging:
       Menghitung median posisi (X, Y cm) dari deteksi positif, membuang outlier ekstrim,
       dan menghitung rata-rata inlier agar koordinat target benar-benar stabil & terpusat.
    3. Status Kunci Target:
       is_locked = True jika detection_rate >= min_rate.
    """

    def __init__(self, window_size: int = 30, min_rate: float = 0.50, spatial_radius_cm: float = 25.0):
        self.window_size = window_size
        self.min_rate = min_rate
        self.spatial_radius_cm = spatial_radius_cm

        # Buffer biner: 1 jika frame ini terdeteksi api, 0 jika tidak
        self.detection_history = deque(maxlen=window_size)
        # Buffer koordinat fisik (X, Y cm) dari deteksi positif
        self.coord_history = deque(maxlen=window_size)

        self.is_locked = False
        self.detection_rate = 0.0
        self.stabilized_coord: Optional[Tuple[float, float]] = None
        self.last_valid_coord: Optional[Tuple[float, float]] = None

    def update(self, raw_coord: Optional[Tuple[float, float]]) -> Tuple[bool, Optional[Tuple[float, float]], float]:
        """
        Memperbarui status stabilizer dengan koordinat raw frame saat ini.

        Returns:
            Tuple[is_locked, stabilized_coord, detection_rate]
        """
        if raw_coord is not None:
            self.detection_history.append(1)
            self.coord_history.append(raw_coord)
            self.last_valid_coord = raw_coord
        else:
            self.detection_history.append(0)

        # 1. Hitung Rasio Deteksi (%) dalam window
        total_frames = len(self.detection_history)
        if total_frames > 0:
            positive_count = sum(self.detection_history)
            self.detection_rate = float(positive_count) / float(total_frames)
        else:
            self.detection_rate = 0.0

        # 2. Evaluasi Kondisi Kunci Target (Target Lock)
        self.is_locked = (self.detection_rate >= self.min_rate and len(self.coord_history) >= max(3, int(self.window_size * 0.2)))

        # 3. Robust Spatial Averaging jika ada sampel koordinat
        if len(self.coord_history) > 0 and self.is_locked:
            coords = list(self.coord_history)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            med_x = float(np.median(xs))
            med_y = float(np.median(ys))

            # Filter inlier: hanya sampel yang berjarak <= spatial_radius_cm dari median
            inliers = [
                c for c in coords
                if np.sqrt((c[0] - med_x) ** 2 + (c[1] - med_y) ** 2) <= self.spatial_radius_cm
            ]

            if inliers:
                avg_x = float(np.mean([c[0] for c in inliers]))
                avg_y = float(np.mean([c[1] for c in inliers]))
                self.stabilized_coord = (avg_x, avg_y)
            else:
                self.stabilized_coord = (med_x, med_y)
        elif raw_coord is not None:
            # Belum terkunci penuh, tampilkan koordinat mentah instan sementara
            self.stabilized_coord = raw_coord
        else:
            self.stabilized_coord = None

        return self.is_locked, self.stabilized_coord, self.detection_rate

    def reset(self):
        self.detection_history.clear()
        self.coord_history.clear()
        self.is_locked = False
        self.detection_rate = 0.0
        self.stabilized_coord = None


class AutoHomographyYoloRunner:
    """Koordinator visualisasi & validasi end-to-end dengan input otomatis dari YOLO."""

    def __init__(self, args):
        self.args = args
        self.is_dual_mode = args.dual
        self.cam_id = args.camera
        self.target_mode = args.target_mode
        self.window_name = "Deezpro Auto-YOLO Homography Validator"

        # 1. Muat Konfigurasi Sistem
        self.config_path = args.config or os.path.join(current_dir, "config", "system_config.yaml")
        self.config = self._load_yaml(self.config_path)

        # 2. Inisialisasi Detektor YOLO
        model_path = args.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, model_path)
        if not os.path.exists(model_path):
            # Fallback ke models/best.pt jika target model tidak ditemukan
            fallback_model = os.path.join(current_dir, "models", "best.pt")
            if os.path.exists(fallback_model):
                logger.warning(f"Model '{model_path}' tidak ditemukan. Menggunakan fallback: '{fallback_model}'")
                model_path = fallback_model
            else:
                logger.error(f"Berkas model YOLO tidak ditemukan di: {model_path}")
                sys.exit(1)

        logger.info(f"Menginisialisasi YOLO Detector: {model_path} (imgsz={args.imgsz}, conf={args.conf})...")
        self.detector = FireSmokeDetector(
            model_path=model_path,
            conf_threshold=args.conf,
            imgsz=args.imgsz
        )

        # 3. Inisialisasi Kinematics Engine
        self.kinematics = KinematicsEngine(self.config)

        # 4. Inisialisasi Temporal & Spatial Stabilizer (HitL Protection)
        self.stabilizer = TargetStabilizer(
            window_size=args.buffer_window,
            min_rate=args.min_detection_rate,
            spatial_radius_cm=args.spatial_radius
        )

        # 5. Status Eksekusi Hardware
        self.coordinator: Optional[CentralCoordinator] = None
        self.mqtt_connected = False
        self.is_action_busy = False
        self.action_status = "Siaga (Tekan G untuk GOTO, S untuk GOTO+SPRAY)"

        # 6. Hubungkan ke MQTT jika diizinkan
        if not args.no_mqtt:
            self._init_coordinator()
        else:
            self.action_status = "Mode DRY-RUN (Koneksi MQTT dinonaktifkan via --no-mqtt)"

        # 7. Setup Kamera & Homografi
        if self.is_dual_mode:
            self._init_dual_mode()
        else:
            self._init_single_mode()

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.warning(f"File konfigurasi {path} tidak ditemukan, menggunakan nilai default!")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Gagal membaca konfigurasi YAML {path}: {e}")
            return {}

    def _init_coordinator(self):
        try:
            logger.info("Menghubungkan Central Coordinator ke MQTT Broker...")
            self.coordinator = CentralCoordinator(self.config)
            # auto_home=False secara ketat demi keselamatan limit switch
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

    def send_goto_command(self, x_cm: float, y_cm: float, source_label: str = "AUTO-TARGET"):
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}] (MQTT Offline)"
            logger.info(self.action_status)
            return

        def _goto():
            self.action_status = f"Menggerakkan nozzle ke ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}]..."
            logger.info(self.action_status)
            success = self.coordinator.manual_goto_cm(x_cm, y_cm)
            if success:
                self.action_status = f"Nozzle TIBA di ({x_cm:.1f}, {y_cm:.1f}) cm!"
                logger.info(self.action_status)
            else:
                self.action_status = "Pergerakan timeout atau dibatalkan!"
                logger.warning(self.action_status)

        self._execute_async_action(_goto)

    def send_goto_and_spray_command(self, x_cm: float, y_cm: float, source_label: str = "AUTO-TARGET"):
        if not self.mqtt_connected or self.coordinator is None:
            self.action_status = f"[DRY-RUN] GOTO+SPRAY ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}] (MQTT Offline)"
            logger.info(self.action_status)
            return

        def _goto_and_spray():
            self.action_status = f"Navigasi ke ({x_cm:.1f}, {y_cm:.1f}) cm [{source_label}]..."
            logger.info(self.action_status)
            success = self.coordinator.manual_goto_cm(x_cm, y_cm)
            if success:
                self.action_status = "Menyemprotkan APAR (2 detik)..."
                logger.info(self.action_status)
                self.coordinator.manual_spray(duration_ms=2000)
                self.action_status = f"Pemadaman di ({x_cm:.1f}, {y_cm:.1f}) cm SELESAI!"
                logger.info(self.action_status)
            else:
                self.action_status = "Gagal mencapai target, semprot dibatalkan!"
                logger.warning(self.action_status)

        self._execute_async_action(_goto_and_spray)

    def trigger_estop(self):
        self.action_status = "EMERGENCY STOP DIAKTIFKAN!"
        logger.warning(self.action_status)
        if self.coordinator:
            self.coordinator.emergency_stop()

    # --- Setup Single-Camera ---
    def _init_single_mode(self):
        cam_cfg = self.config.get("camera", {})
        default_dev = cam_cfg.get("camera_1_index", 0) if self.cam_id == 1 else cam_cfg.get("camera_2_index", 1)
        self.dev_idx = self.args.source if self.args.source is not None else default_dev

        homo_path = self.args.homography or os.path.join(current_dir, "config", f"homography_cam{self.cam_id}.npz")
        if not os.path.exists(homo_path):
            logger.error(f"Berkas homografi tidak ditemukan: {homo_path}")
            sys.exit(1)

        self.homo = HomographyEngine(homo_path)
        logger.info(f"Homografi Kamera {self.cam_id} dimuat dari: {homo_path}")

        # Undistort Maps
        self.map1, self.map2 = None, None
        if not self.args.no_undistort:
            calib_path = self.args.calib or os.path.join(current_dir, "config", f"calib_cam{self.cam_id}.npz")
            if os.path.exists(calib_path):
                try:
                    data = np.load(calib_path)
                    K = data["camera_matrix"] if "camera_matrix" in data else data.get("mtx")
                    D = data["dist_coeffs"] if "dist_coeffs" in data else data.get("dist")
                    w, h = self.args.width, self.args.height
                    alpha = getattr(self.args, "alpha", 1.0)
                    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha, newImgSize=(w, h))
                    self.map1, self.map2 = cv2.initUndistortRectifyMap(
                        K, D, None, new_K, (w, h), cv2.CV_16SC2
                    )
                    logger.info(f"Kalibrasi lensa Kamera {self.cam_id} aktif: {calib_path} (alpha={alpha})")
                except Exception as e:
                    logger.warning(f"Gagal memuat peta undistort Kamera {self.cam_id}: {e}")

    # --- Setup Dual-Camera ---
    def _init_dual_mode(self):
        cam_cfg = self.config.get("camera", {})
        self.dev_idx1 = self.args.cam1 if self.args.cam1 is not None else cam_cfg.get("camera_1_index", 0)
        self.dev_idx2 = self.args.cam2 if self.args.cam2 is not None else cam_cfg.get("camera_2_index", 1)

        h1_path = os.path.join(current_dir, "config", "homography_cam1.npz")
        h2_path = os.path.join(current_dir, "config", "homography_cam2.npz")

        if not os.path.exists(h1_path) or not os.path.exists(h2_path):
            logger.error(f"Berkas homografi dual kamera tidak lengkap: {h1_path} atau {h2_path}")
            sys.exit(1)

        self.homo1 = HomographyEngine(h1_path)
        self.homo2 = HomographyEngine(h2_path)
        logger.info("Homografi Kamera 1 dan Kamera 2 berhasil dimuat.")

        # Undistort Maps untuk Cam 1 & Cam 2
        self.map1_c1, self.map2_c1 = None, None
        self.map1_c2, self.map2_c2 = None, None

        if not self.args.no_undistort:
            w, h = self.args.width, self.args.height
            alpha = getattr(self.args, "alpha", 1.0)
            c1_path = os.path.join(current_dir, "config", "calib_cam1.npz")
            if os.path.exists(c1_path):
                try:
                    d = np.load(c1_path)
                    K1 = d["camera_matrix"] if "camera_matrix" in d else d.get("mtx")
                    D1 = d["dist_coeffs"] if "dist_coeffs" in d else d.get("dist")
                    new_K1, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (w, h), alpha=alpha, newImgSize=(w, h))
                    self.map1_c1, self.map2_c1 = cv2.initUndistortRectifyMap(
                        K1, D1, None, new_K1, (w, h), cv2.CV_16SC2
                    )
                except Exception as e:
                    logger.warning(f"Gagal memuat kalibrasi Cam 1: {e}")

            c2_path = os.path.join(current_dir, "config", "calib_cam2.npz")
            if os.path.exists(c2_path):
                try:
                    d = np.load(c2_path)
                    K2 = d["camera_matrix"] if "camera_matrix" in d else d.get("mtx")
                    D2 = d["dist_coeffs"] if "dist_coeffs" in d else d.get("dist")
                    new_K2, _ = cv2.getOptimalNewCameraMatrix(K2, D2, (w, h), alpha=alpha, newImgSize=(w, h))
                    self.map1_c2, self.map2_c2 = cv2.initUndistortRectifyMap(
                        K2, D2, None, new_K2, (w, h), cv2.CV_16SC2
                    )
                except Exception as e:
                    logger.warning(f"Gagal memuat kalibrasi Cam 2: {e}")

    # --- Ekstraksi Target Terbaik dari Deteksi YOLO ---
    def _extract_primary_target(self, detections: List[PixelDetection]) -> Optional[PixelDetection]:
        """Memilih deteksi api ('fire') dengan confidence tertinggi."""
        fire_dets = [d for d in detections if d.is_fire]
        if fire_dets:
            return max(fire_dets, key=lambda d: d.confidence)
        # Jika tidak ada api, fallback ke deteksi apa pun dengan conf tertinggi
        if detections:
            return max(detections, key=lambda d: d.confidence)
        return None

    # --- Rendering Visualisasi Per Kamera ---
    def _draw_camera_annotations(
        self,
        frame: np.ndarray,
        detections: List[PixelDetection],
        primary_det: Optional[PixelDetection],
        homo: HomographyEngine,
        cam_label: str,
        cross_guide_world: Optional[Tuple[float, float]] = None,
        cross_guide_label: str = "Cam Lawan"
    ) -> Optional[Tuple[float, float]]:
        """Menggambar bounding box YOLO, target point, dan cross-projection pada frame."""
        h, w = frame.shape[:2]

        # 1. Gambar seluruh deteksi
        for det in detections:
            is_primary = (primary_det is not None and det == primary_det)
            color = (0, 0, 255) if det.is_fire else (180, 180, 180)
            thickness = 3 if is_primary else 1

            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            lbl = f"{det.cls_name.upper()} {det.confidence*100:.1f}%"
            cv2.putText(frame, lbl, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Gambar Cross-Projection dari kamera lawan (jika ada)
        if cross_guide_world is not None:
            try:
                proj_px = homo.project_world_to_image(cross_guide_world[0], cross_guide_world[1])
                px_u, px_v = int(proj_px[0]), int(proj_px[1])
                if 0 <= px_u < w and 0 <= px_v < h:
                    cv2.drawMarker(frame, (px_u, px_v), (0, 255, 255),
                                   cv2.MARKER_TILTED_CROSS, 24, 2)
                    cv2.circle(frame, (px_u, px_v), 16, (0, 255, 255), 1)
                    cv2.putText(frame, f"Cross-Proj ({cross_guide_label})", (px_u + 12, px_v - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            except Exception:
                pass

        # 3. Hitung koordinat dunia untuk primary detection
        world_pt = None
        if primary_det is not None:
            target_pt = primary_det.base_center if self.target_mode == "base" else primary_det.centroid
            try:
                world_x, world_y = homo.project_image_to_world(target_pt[0], target_pt[1])
                world_pt = (world_x, world_y)

                # Gambar Titik Target Acuan
                color_target = (0, 255, 0) if self.target_mode == "base" else (255, 100, 0)
                cv2.circle(frame, target_pt, 7, color_target, -1)
                cv2.circle(frame, target_pt, 12, (255, 255, 255), 2)

                # Tampilkan info koordinat lokal di dekat bounding box
                bx1, by1, _, by2 = primary_det.bbox
                txt_y = min(h - 10, by2 + 25)
                coord_str = f"Lokal: ({world_x:.1f}, {world_y:.1f}) cm [{self.target_mode.upper()}]"
                cv2.putText(frame, coord_str, (bx1, txt_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            except Exception as e:
                logger.warning(f"Gagal proyeksi homografi {cam_label}: {e}")

        # Label Header Kamera
        cv2.putText(frame, f"[{cam_label}]", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return world_pt

    # --- Mode Kamera Tunggal ---
    def run_single(self):
        logger.info(f"Menjalankan Auto YOLO Single-Camera Mode (Kamera {self.cam_id}, Dev Index {self.dev_idx})...")
        cam = None
        try:
            cam = ThreadedCamera(
                self.dev_idx,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
                label=f"Kamera {self.cam_id}"
            )
        except Exception as e:
            logger.error(f"Gagal membuka Kamera {self.cam_id}: {e}")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 680)

        print("\n" + "=" * 70)
        print(f" DEEZPRO AUTO-YOLO HOMOGRAPHY VALIDATOR - SINGLE CAMERA {self.cam_id}")
        print("=" * 70)
        print(" [G / ENTER] : Kirim koordinat target YOLO ke ESP32 (GOTO).")
        print(" [S]         : Kirim target YOLO + SPRAY APAR (2 detik).")
        print(" [M]         : Ganti target mode (base <-> centroid).")
        print(" [SPACE]     : EMERGENCY STOP (Hentikan seluruh motor).")
        print(" [Q / ESC]   : Keluar.")
        print("=" * 70 + "\n")

        fps_prev_time = time.time()
        fps_val = 0.0

        try:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Koreksi Distorsi Lensa
                if self.map1 is not None and self.map2 is not None:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                # Deteksi Otomatis YOLO
                detections = self.detector.detect(frame)
                primary_det = self._extract_primary_target(detections)

                # Anotasi & Hitung Koordinat Fisik Instan
                raw_world = self._draw_camera_annotations(
                    frame, detections, primary_det, self.homo, f"KAMERA {self.cam_id}"
                )

                # Update Temporal & Spatial Stabilizer
                is_locked, stabilized_world, det_rate = self.stabilizer.update(raw_world)

                # Target yang digunakan untuk kinematika: utamakan stabilized_world jika terkunci, atau raw_world
                effective_world = stabilized_world if is_locked else raw_world

                # Kinematics Clamping
                clamped_world = None
                clamped_step = None
                in_safe_bounds = False

                if effective_world is not None:
                    in_safe_bounds = self.kinematics.is_within_soft_limits(effective_world[0], effective_world[1])
                    cx, cy = self.kinematics.clamp_world_coordinates(effective_world[0], effective_world[1])
                    clamped_world = (cx, cy)
                    try:
                        clamped_step = self.kinematics.world_to_steps(cx, cy)
                    except Exception:
                        clamped_step = None

                # Hitung FPS
                cur_t = time.time()
                dt = cur_t - fps_prev_time
                fps_prev_time = cur_t
                if dt > 0:
                    fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt)

                # Overlay Status Panel
                info_overlay = np.zeros((140, frame.shape[1], 3), dtype=np.uint8)
                color_bounds = (0, 255, 0) if in_safe_bounds else (0, 140, 255)
                lock_badge = "[TARGET LOCKED - STABLE]" if is_locked else f"[BUFFERING... {det_rate*100:.0f}%]"
                lock_color = (0, 255, 0) if is_locked else (0, 200, 255)

                if effective_world is not None and clamped_world is not None and primary_det is not None:
                    status_line1 = (
                        f"Target Api: ({effective_world[0]:.1f}, {effective_world[1]:.1f}) cm {lock_badge} | "
                        f"Clamped: ({clamped_world[0]:.1f}, {clamped_world[1]:.1f}) cm | Rate: {det_rate*100:.1f}%"
                    )
                    step_str = f"Step X: {clamped_step[0]} | Step Y: {clamped_step[1]}" if clamped_step else "Step: N/A"
                    status_line2 = (
                        f"Kinematika: {step_str} | Status Batas: {'AMAN' if in_safe_bounds else 'CLAMPED'} | "
                        f"Conf: {primary_det.confidence*100:.1f}% | Mode: {self.target_mode.upper()}"
                    )
                else:
                    status_line1 = f"Mencari Api... (Window Buffer: {self.args.buffer_window} fr | Det Rate: {det_rate*100:.0f}%)"
                    status_line2 = "Arahkan kamera ke lidah api atau pemantik gas."

                status_line3 = f"HitL Control: [G/ENTER]=GOTO | [S]=GOTO+SPRAY | [M]=MODE | FPS: {fps_val:.1f}"
                status_line4 = f"Status Aksi: {self.action_status}"

                cv2.putText(info_overlay, status_line1, (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, lock_color, 1)
                cv2.putText(info_overlay, status_line2, (15, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_bounds, 1)
                cv2.putText(info_overlay, status_line3, (15, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)
                cv2.putText(info_overlay, status_line4, (15, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

                display_img = np.vstack([frame, info_overlay])
                cv2.imshow(self.window_name, display_img)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    break
                elif key in [ord("g"), ord("G"), 10, 13]:
                    if clamped_world is not None:
                        self.send_goto_command(clamped_world[0], clamped_world[1], f"Cam {self.cam_id}")
                    else:
                        self.action_status = "Tidak ada api terdeteksi untuk GOTO!"
                elif key in [ord("s"), ord("S")]:
                    if clamped_world is not None:
                        self.send_goto_and_spray_command(clamped_world[0], clamped_world[1], f"Cam {self.cam_id}")
                    else:
                        self.action_status = "Tidak ada api terdeteksi untuk GOTO+SPRAY!"
                elif key in [ord("m"), ord("M")]:
                    self.target_mode = "centroid" if self.target_mode == "base" else "base"
                    self.action_status = f"Mode titik target diubah ke: {self.target_mode.upper()}"
                elif key == 32:
                    self.trigger_estop()

        finally:
            cam.release()
            cv2.destroyAllWindows()
            if self.coordinator:
                self.coordinator.stop()

    # --- Mode Dua Kamera (Dual-Camera Late Fusion) ---
    def run_dual(self):
        logger.info(f"Menjalankan Auto YOLO Dual-Camera Mode (Cam1: Index {self.dev_idx1}, Cam2: Index {self.dev_idx2})...")
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

        view_w, view_h = 800, 450
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, view_w * 2, view_h + 160)

        print("\n" + "=" * 75)
        print(" DEEZPRO AUTO-YOLO HOMOGRAPHY VALIDATOR - DUAL CAMERA FUSION MODE")
        print("=" * 75)
        print(" [G / ENTER] : Kirim koordinat FUSI (atau single) ke ESP32 (GOTO).")
        print(" [S]         : Kirim koordinat FUSI + SPRAY (2 detik).")
        print(" [M]         : Ganti target mode (base <-> centroid).")
        print(" [SPACE]     : Emergency Stop (ESTOP).")
        print(" [Q / ESC]   : Keluar.")
        print("=" * 75 + "\n")

        fps_prev_time = time.time()
        fps_val = 0.0

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

                # Deteksi YOLO Otomatis pada kedua kamera
                dets1 = self.detector.detect(f1)
                dets2 = self.detector.detect(f2)

                p1 = self._extract_primary_target(dets1)
                p2 = self._extract_primary_target(dets2)

                # Estimasi titik dunia awal untuk Cross-Projection
                w1_est = None
                if p1 is not None:
                    t1 = p1.base_center if self.target_mode == "base" else p1.centroid
                    try:
                        w1_est = self.homo1.project_image_to_world(*t1)
                    except Exception:
                        pass

                w2_est = None
                if p2 is not None:
                    t2 = p2.base_center if self.target_mode == "base" else p2.centroid
                    try:
                        w2_est = self.homo2.project_image_to_world(*t2)
                    except Exception:
                        pass

                # Anotasi visual masing-masing kamera
                raw_w1 = self._draw_camera_annotations(
                    f1, dets1, p1, self.homo1, "KAMERA 1",
                    cross_guide_world=w2_est, cross_guide_label="Cam 2"
                )
                raw_w2 = self._draw_camera_annotations(
                    f2, dets2, p2, self.homo2, "KAMERA 2",
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
                    # KEDUA KAMERA DETEKSI: LATE FUSION (RATA-RATA)
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
                        consistency_status = "PERBEDAAN TINGGI (Periksa Kalibrasi!)"

                elif raw_w1 is not None:
                    active_target_cm = raw_w1
                    target_mode_desc = "SINGLE: KAMERA 1 SAJA (Kamera 2 Tertutup/Tidak Terdeteksi)"
                elif raw_w2 is not None:
                    active_target_cm = raw_w2
                    target_mode_desc = "SINGLE: KAMERA 2 SAJA (Kamera 1 Tertutup/Tidak Terdeteksi)"

                # Update Temporal & Spatial Stabilizer pada hasil fusi/single
                is_locked, stabilized_target_cm, det_rate = self.stabilizer.update(active_target_cm)
                effective_target_cm = stabilized_target_cm if is_locked else active_target_cm

                # Kinematics Clamping
                clamped_target_cm = None
                clamped_steps = None
                in_safe_bounds = False

                if effective_target_cm is not None:
                    in_safe_bounds = self.kinematics.is_within_soft_limits(effective_target_cm[0], effective_target_cm[1])
                    cx, cy = self.kinematics.clamp_world_coordinates(effective_target_cm[0], effective_target_cm[1])
                    clamped_target_cm = (cx, cy)
                    try:
                        clamped_steps = self.kinematics.world_to_steps(cx, cy)
                    except Exception:
                        clamped_steps = None

                # Hitung FPS
                cur_t = time.time()
                dt = cur_t - fps_prev_time
                fps_prev_time = cur_t
                if dt > 0:
                    fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt)

                # Gabungkan Feed Kamera Kiri + Kanan
                dual_feed = np.hstack([f1_res, f2_res])

                # Panel Status Telemetri Bawah
                dash_h = 160
                dash_w = view_w * 2
                dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)

                col_w = dash_w // 3
                cv2.line(dashboard, (col_w, 10), (col_w, dash_h - 10), (50, 50, 50), 1)
                cv2.line(dashboard, (col_w * 2, 10), (col_w * 2, dash_h - 10), (50, 50, 50), 1)

                lock_badge = "LOCKED" if is_locked else f"BUFFER {det_rate*100:.0f}%"
                lock_color = (0, 255, 0) if is_locked else (0, 200, 255)

                # Kolom 1: Status Sumber Deteksi & Fusi
                cv2.putText(dashboard, f"[1] STATUS FUSI & DETEKSI [{lock_badge}]", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, lock_color, 1)
                cv2.putText(dashboard, f"Mode: {target_mode_desc[:38]}", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

                if discrepancy_cm is not None:
                    color_disc = (0, 255, 0) if discrepancy_cm <= 15.0 else (0, 165, 255)
                    cv2.putText(dashboard, f"Selisih Spasial : {discrepancy_cm:.2f} cm", (15, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_disc, 1)
                    cv2.putText(dashboard, f"Konsistensi     : {consistency_status}", (15, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_disc, 1)
                else:
                    cv2.putText(dashboard, f"Detection Rate  : {det_rate*100:.1f}% (Min: {self.args.min_detection_rate*100:.0f}%)", (15, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lock_color, 1)
                    cv2.putText(dashboard, f"Target Acuan    : {self.target_mode.upper()} point", (15, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # Kolom 2: Koordinat Target & Kinematika
                cv2.putText(dashboard, "[2] KOORDINAT FISIK & KINEMATIKA", (col_w + 15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                if effective_target_cm is not None and clamped_target_cm is not None:
                    stabilized_note = " (Stabil)" if is_locked else " (Instan)"
                    cv2.putText(dashboard, f"Target Fisik: ({effective_target_cm[0]:.1f}, {effective_target_cm[1]:.1f}) cm{stabilized_note}",
                                (col_w + 15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    color_b = (0, 255, 0) if in_safe_bounds else (0, 140, 255)
                    cv2.putText(dashboard, f"Target Rail : ({clamped_target_cm[0]:.1f}, {clamped_target_cm[1]:.1f}) cm [{ 'AMAN' if in_safe_bounds else 'CLAMPED' }]",
                                (col_w + 15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color_b, 1)
                    if clamped_steps is not None:
                        cv2.putText(dashboard, f"Motor Steps : X={clamped_steps[0]} | Y={clamped_steps[1]}",
                                    (col_w + 15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1)
                else:
                    cv2.putText(dashboard, "Menunggu deteksi objek api...",
                                (col_w + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

                # Kolom 3: Status Hardware & Kontrol
                cv2.putText(dashboard, "[3] KONTROL & HARDWARE ESP32", (col_w * 2 + 15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                mqtt_str = "TERHUBUNG" if self.mqtt_connected else "TERPUTUS / DRY-RUN"
                color_m = (0, 255, 0) if self.mqtt_connected else (0, 0, 255)
                cv2.putText(dashboard, f"MQTT Broker : {mqtt_str}", (col_w * 2 + 15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_m, 1)

                cv2.putText(dashboard, f"Status Aksi : {self.action_status[:38]}", (col_w * 2 + 15, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(dashboard, f"[G]=GOTO | [S]=GOTO+SPRAY | [M]=MODE | FPS={fps_val:.1f}",
                            (col_w * 2 + 15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                final_view = np.vstack([dual_feed, dashboard])
                cv2.imshow(self.window_name, final_view)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    break
                elif key in [ord("g"), ord("G"), 10, 13]:
                    if clamped_target_cm is not None:
                        self.send_goto_command(clamped_target_cm[0], clamped_target_cm[1], target_mode_desc)
                    else:
                        self.action_status = "Tidak ada api terdeteksi untuk GOTO!"
                elif key in [ord("s"), ord("S")]:
                    if clamped_target_cm is not None:
                        self.send_goto_and_spray_command(clamped_target_cm[0], clamped_target_cm[1], target_mode_desc)
                    else:
                        self.action_status = "Tidak ada api terdeteksi untuk GOTO+SPRAY!"
                elif key in [ord("m"), ord("M")]:
                    self.target_mode = "centroid" if self.target_mode == "base" else "base"
                    self.action_status = f"Mode titik target diubah ke: {self.target_mode.upper()}"
                elif key == 32:
                    self.trigger_estop()

        finally:
            cam1.release()
            cam2.release()
            cv2.destroyAllWindows()
            if self.coordinator:
                self.coordinator.stop()


def main():
    args = parse_arguments()
    runner = AutoHomographyYoloRunner(args)
    if args.dual:
        runner.run_dual()
    else:
        runner.run_single()


if __name__ == "__main__":
    main()
