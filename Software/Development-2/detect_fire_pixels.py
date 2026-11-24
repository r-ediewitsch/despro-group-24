"""
=====================================================================
DEEZPRO VISION - DETEKSI API & ASAP BERBASIS KOORDINAT PIKSEL
=====================================================================
File: detect_fire_pixels.py
Deskripsi: Program visualisasi real-time deteksi api dan asap menggunakan
            model best.pt. Mengidentifikasi bounding box, titik centroid
            geometris (u_c, v_c), dan titik kontak dasar lantai (u_b, v_b)
            dalam satuan piksel citra kamera.
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vision.detector import FireSmokeDetector
from src.integration.system_logger import setup_logger

logger = setup_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Deteksi Api & Asap Berbasis Koordinat Piksel menggunakan YOLO best.pt."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Nomor logika kamera: 1 untuk Kamera 1, 2 untuk Kamera 2.",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Override index hardware webcam (0, 1, 2...). Jika tidak diisi, membaca dari config/system_config.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path berkas bobot model YOLO (.pt). Default: models/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold deteksi (default: 0.40).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran resolusi input inferensi YOLO (default: 640).",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path berkas matriks kalibrasi lensa (.npz). Default: config/calib_cam<cam>.npz",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Nonaktifkan proses koreksi distorsi lensa.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["base", "centroid"],
        default="base",
        help="Titik acuan target piksel: 'base' (kontak lantai) atau 'centroid' (pusat api).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Jalankan deteksi pada satu berkas citra statis.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Resolusi lebar tangkapan kamera (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Resolusi tinggi tangkapan kamera (default: 1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate kamera (default: 30 FPS).",
    )
    return parser.parse_args()


def load_lens_calibration(calib_file: str, width: int, height: int):
    """Memuat matriks K & D untuk koreksi distorsi lensa."""
    if not os.path.exists(calib_file):
        return None, None

    data = np.load(calib_file)
    K = data["camera_matrix"]
    D = data["dist_coeffs"]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha=0.0, newImgSize=(width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
    return map1, map2


def draw_hud(
    frame: np.ndarray,
    fps: float,
    fire_count: int,
    smoke_count: int,
    target_mode: str,
    is_undistort: bool,
    cam_id: int,
    model_name: str,
):
    """Menggambar banner HUD status dan panduan keyboard."""
    h, w = frame.shape[:2]

    # Bar banner atas semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # 1. Status Bahaya Api & Asap
    if fire_count > 0 and smoke_count > 0:
        badge_text = f" [!] API & ASAP TERDETEKSI (Api: {fire_count}, Asap: {smoke_count}) "
        badge_color = (0, 0, 255)  # Merah terang
    elif fire_count > 0:
        badge_text = f" [!] API TERDETEKSI ({fire_count} Titik Api) "
        badge_color = (0, 0, 255)  # Merah terang
    elif smoke_count > 0:
        badge_text = f" [!] ASAP TERDETEKSI ({smoke_count} Titik Asap) "
        badge_color = (0, 165, 255)  # Oranye / Amber
    else:
        badge_text = " [OK] MONITORING AMAN (Tidak Ada Bahaya) "
        badge_color = (0, 255, 0)  # Hijau

    cv2.putText(frame, badge_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, badge_color, 2)

    # 2. Metadata Sistem
    undist_str = "AKTIF" if is_undistort else "OFF"
    tgt_str = "Dasar Lantai (u_b, v_b)" if target_mode == "base" else "Centroid (u_c, v_c)"
    info_text = (
        f"FPS: {fps:.1f} | Cam: {cam_id} | Model: {model_name} | "
        f"Target: [{tgt_str}] | Undistort: [{undist_str}]"
    )
    cv2.putText(frame, info_text, (500, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # 3. Bar Bawah Panduan Kontrol
    cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
    help_text = (
        "[P]: Ubah Titik Target (Base <-> Centroid) | [U]: Toggle Undistort | "
        "[S]: Simpan Screenshot | [Q/ESC]: Keluar"
    )
    cv2.putText(frame, help_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)


def main():
    args = parse_arguments()

    print("\n" + "=" * 70)
    print(" SISTEM DETEKSI API & ASAP (KOORDINAT PIKSEL) - DEEZPRO")
    print("=" * 70)
    print(f" Model YOLO     : {args.model}")
    print(f" Threshold Conf : {args.conf}")
    print(f" Target Mode    : {'Base-Center (Kontak Lantai)' if args.target_mode == 'base' else 'Centroid (Pusat BBox)'}")
    print("=" * 70 + "\n")

    # Inisialisasi Detector
    try:
        detector = FireSmokeDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
        )
        logger.info(f"Detector siap! Kelas model: {detector.class_names}")
    except Exception as e:
        logger.error(f"Gagal memuat model YOLO: {e}")
        return

    # Inisialisasi Kalibrasi Undistort
    calib_file = args.calib
    if calib_file is None:
        calib_file = os.path.join(current_dir, "config", f"calib_cam{args.camera}.npz")

    map1, map2 = None, None
    is_undistort = not args.no_undistort
    if is_undistort:
        map1, map2 = load_lens_calibration(calib_file, args.width, args.height)
        if map1 is not None and map2 is not None:
            logger.info(f"Koreksi distorsi lensa aktif menggunakan: {os.path.basename(calib_file)}")
        else:
            logger.warning(f"Berkas kalibrasi lensa ({calib_file}) tidak ditemukan. Berjalan tanpa undistort.")
            is_undistort = False

    target_mode = args.target_mode
    window_name = f"Deezpro - Deteksi Api YOLO ({os.path.basename(detector.model_path)})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Skenario 1: Inferensi pada citra statis
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Berkas citra tidak ditemukan: {args.image}")
            return
        frame = cv2.imread(args.image)
        if is_undistort and map1 is not None and map2 is not None:
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        detections = detector.detect(frame)
        fire_cnt = sum(1 for d in detections if d.is_fire)
        smoke_cnt = sum(1 for d in detections if d.is_smoke)

        print("\n--- HASIL DETEKSI PIKSEL ---")
        for i, d in enumerate(detections):
            print(
                f" [{i+1}] {d.cls_name.upper()} (conf: {d.confidence:.2f}) | "
                f"BBox: {d.bbox} | Centroid: {d.centroid} px | Base: {d.base_center} px"
            )
        print("----------------------------\n")

        annotated = detector.draw_detections(frame, detections, target_mode=target_mode)
        draw_hud(annotated, 0.0, fire_cnt, smoke_cnt, target_mode, is_undistort, args.camera, os.path.basename(detector.model_path))

        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Resolusi hardware index dari system_config.yaml jika tidak dioverride
    device_index = args.source
    if device_index is None:
        cfg_path = os.path.join(current_dir, "config", "system_config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path, "r") as f:
                    cfg_y = yaml.safe_load(f) or {}
                    cam_key = f"cam{args.camera}"
                    if "cameras" in cfg_y and cam_key in cfg_y["cameras"]:
                        device_index = int(cfg_y["cameras"][cam_key].get("index", args.camera))
            except Exception:
                pass
        if device_index is None:
            device_index = args.camera

    logger.info(f"Kamera Logika {args.camera} terhubung ke Hardware Device Index {device_index}")

    # Skenario 2: Streaming Webcam Live
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        logger.error(f"Gagal membuka webcam kamera indeks {device_index} (Kamera {args.camera})!")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.005)
                continue

            # Putar 180 derajat karena posisi kamera pada prototype terbalik
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_180)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # 1. Undistort Lensa (Opsional)
            if is_undistort and map1 is not None and map2 is not None:
                processed_frame = cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            else:
                processed_frame = raw_frame

            # 2. Eksekusi Inferensi YOLO
            detections = detector.detect(processed_frame)

            fire_cnt = sum(1 for d in detections if d.is_fire)
            smoke_cnt = sum(1 for d in detections if d.is_smoke)

            # 3. Gambar Anotasi Bounding Box & Titik Piksel
            display_frame = detector.draw_detections(processed_frame, detections, target_mode=target_mode)

            # 4. Gambar Status HUD
            draw_hud(
                display_frame,
                fps,
                fire_cnt,
                smoke_cnt,
                target_mode,
                is_undistort,
                args.camera,
                os.path.basename(detector.model_path),
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("p"):  # Toggle mode titik target (Base vs Centroid)
                target_mode = "centroid" if target_mode == "base" else "base"
                mode_label = "Base-Center (Kontak Lantai)" if target_mode == "base" else "Centroid (Pusat BBox)"
                print(f"[TOGGLE] Mode Titik Target diubah ke: {mode_label}")
            elif key == ord("u"):  # Toggle Undistort
                is_undistort = not is_undistort
                print(f"[TOGGLE] Koreksi Lensa Undistort: {'AKTIF' if is_undistort else 'NONAKTIF'}")
            elif key == ord("s"):  # Simpan screenshot
                fname = f"capture_fire_pixels_{int(time.time())}.jpg"
                cv2.imwrite(fname, display_frame)
                print(f"[SCREENSHOT] Tangkapan layar disimpan: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Program deteksi api koordinat piksel ditutup.")


if __name__ == "__main__":
    main()
