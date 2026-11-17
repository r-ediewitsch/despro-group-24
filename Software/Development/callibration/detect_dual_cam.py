import os
import sys

# Redam log peringatan backend OpenCV agar output terminal bersih
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import time
import argparse
import zipfile
import threading
import numpy as np
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi Api Stereo Dual-Webcam (Kamera 1 & Kamera 2) dengan Kalibrasi Zhang & YOLO."
    )
    parser.add_argument(
        "--cam1",
        type=int,
        default=1,
        help="Indeks webcam untuk Kamera 1 (default: 1).",
    )
    parser.add_argument(
        "--cam2",
        type=int,
        default=2,
        help="Indeks webcam untuk Kamera 2 (default: 2).",
    )
    parser.add_argument(
        "--calib1",
        type=str,
        default="calib_cam1.npz",
        help="File kalibrasi untuk Kamera 1 (default: calib_cam1.npz).",
    )
    parser.add_argument(
        "--calib2",
        type=str,
        default="calib_cam2.npz",
        help="File kalibrasi untuk Kamera 2 (default: calib_cam2.npz).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Path file bobot model YOLO (.pt). Default: yolo26n.pt",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate untuk kedua kamera (default: 30 FPS).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi input inferensi YOLO (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold deteksi (default: 0.4).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Resolusi lebar frame kamera yang diminta (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Resolusi tinggi frame kamera yang diminta (default: 1080).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Parameter scaling undistort (0.0: buang tepi hitam, 1.0: simpan semua piksel).",
    )
    parser.add_argument(
        "--fire-only",
        action="store_true",
        default=True,
        help="Hanya tampilkan deteksi api (sembunyikan manusia, barang, dll).",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Tampilkan semua objek yang terdeteksi.",
    )
    parser.add_argument(
        "--treat-cls0-as-fire",
        action="store_true",
        help="Paksa kelas ID 0 diperlakukan sebagai 'Fire/Api'.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Nonaktifkan proses koreksi undistort lensa.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Pindai dan tampilkan seluruh indeks kamera yang tersedia di komputer lalu keluar.",
    )
    return parser.parse_args()


def scan_available_cameras(max_tested=8):
    """
    Memindai indeks kamera yang tersedia di sistem untuk membantu pengguna
    mengidentifikasi indeks kamera fisik.
    """
    print("\n" + "=" * 65)
    print(" PEMINDAIAN WEBCAM TERHUBUNG (SCANNER)")
    print("=" * 65)
    found_any = False
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f" [TERDETEKSI] Indeks {idx} : Aktif (Default Res: {w}x{h} @ {fps:.1f} FPS)")
            found_any = True
            cap.release()
        else:
            print(f" [KOSONG]     Indeks {idx} : Tidak ada perangkat")

    if not found_any:
        print("\nTidak ada kamera yang terdeteksi. Pastikan kabel USB terpasang dengan baik.")
    print("=" * 65 + "\n")


def resolve_path(path_str):
    """
    Menyelesaikan path relatif baik dari CWD maupun dari folder skrip ini berada.
    """
    if os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path_str)
    if os.path.exists(candidate):
        return candidate
    return path_str


def load_calibration(calib_file, target_w, target_h, alpha=0.0):
    """
    Memuat matriks kalibrasi dan menghitung map undistort/rectify untuk efisiensi komputasi real-time.
    """
    calib_file = resolve_path(calib_file)
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {calib_file}")

    data = np.load(calib_file)
    K = data["camera_matrix"]
    D = data["dist_coeffs"]
    mean_err = data.get("mean_error", None)

    err_str = f"{float(mean_err):.4f} px" if mean_err is not None else "N/A"
    print(f"[INFO] Kalibrasi dimuat: {os.path.basename(calib_file)} (Mean Error: {err_str})")

    # Hitung matriks optimal dan remap grid CV_16SC2 (akselerasi hardware)
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (target_w, target_h), alpha=alpha, newImgSize=(target_w, target_h)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (target_w, target_h), cv2.CV_16SC2
    )
    return K, D, new_K, map1, map2, roi


class ThreadedCamera:
    """
    Kamera real-time berbasis multi-threading dengan akselerasi format MJPG.
    Menghilangkan masalah drop FPS ke 5 FPS akibat saturasi bandwidth USB 2.0
    dan memisahkan I/O kamera dari komputasi inferensi YOLO.
    """

    def __init__(self, cam_index, target_w=1920, target_h=1080, target_fps=30, label="Cam"):
        self.cam_index = cam_index
        self.label = label
        print(f"[INFO] Inisialisasi {label} pada indeks {cam_index} (MJPG 30 FPS)...")

        # 1. Buka device dengan backend DirectShow (paling stabil di Windows)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Gagal membuka {label} (Indeks {cam_index})! Pastikan webcam terhubung."
            )

        # 2. PENTING: Set FOURCC ke MJPG TERLEBIH DAHULU sebelum set resolusi!
        # Tanpa MJPG, driver USB kamera beralih ke format uncompressed (YUY2/YUV).
        # USB 2.0 (maks ~35 MB/s) tidak sanggup membawa 1080p uncompressed (124 MB/s @ 30fps),
        # sehingga hardware membatasi ke 5 FPS. MJPG mengompres di kamera sehingga tembus 30 FPS.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

        print(
            f"[INFO] {label} Siap (Indeks {self.cam_index}): "
            f"{self.actual_w}x{self.actual_h} @ {reported_fps:.1f} FPS (MJPG Enabled)"
        )

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.003)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


def setup_yolo_model(model_path, treat_cls0_as_fire=False):
    """
    Memuat model YOLO dan memvalidasi bobot model.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] Modul 'ultralytics' belum terpasang!")
        print("Silakan jalankan: pip install ultralytics\n")
        sys.exit(1)

    model_path = resolve_path(model_path)

    if not os.path.isfile(model_path) and os.path.isdir(model_path):
        packed_pt = model_path.rstrip("/\\") + "_packed.pt"
        if not os.path.exists(packed_pt):
            print(f"[INFO] Mengemas direktori menjadi arsip PyTorch: {packed_pt} ...")
            with zipfile.ZipFile(packed_pt, "w", compression=zipfile.ZIP_STORED) as zf:
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        rp = os.path.relpath(fp, model_path)
                        zf.write(fp, rp)
        model_path = packed_pt

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model = YOLO(model_path)
    class_names = dict(model.names) if hasattr(model, "names") else {}

    fire_keywords = {"fire", "api", "flame", "smoke"}
    exclude_keywords = {"hydrant"}

    fire_class_ids = []
    if treat_cls0_as_fire and 0 in class_names:
        fire_class_ids.append(0)
        class_names[0] = "fire"
    else:
        for cid, name in class_names.items():
            nl = str(name).lower()
            if any(kw in nl for kw in fire_keywords) and not any(ex in nl for ex in exclude_keywords):
                fire_class_ids.append(cid)

    print("\n" + "=" * 65)
    print(" [VERIFIKASI BOBOT MODEL YOLO]")
    print(f" File        : {os.path.basename(model_path)} ({model_size_mb:.2f} MB)")
    print(f" Total Kelas : {len(class_names)}")
    if fire_class_ids:
        print(f" Target Api  : Kelas {fire_class_ids} ({[class_names[i] for i in fire_class_ids]})")
    else:
        print(" Target Api  : Kelas 'fire' tidak ditemukan pada bobot ini (basis COCO).")
        print("               Tekan tombol '0' untuk memetakan Kelas 0 sebagai Api.")
    print("=" * 65 + "\n")

    return model, class_names, fire_class_ids, os.path.basename(model_path)


def process_detections(boxes, class_names, fire_class_ids, fire_only_mode):
    """
    Ekstraksi bounding box, centroid, dan filter kelas api.
    """
    detections = []
    fire_count = 0
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            is_fire = cls_id in fire_class_ids

            if fire_only_mode and not is_fire:
                continue

            if is_fire:
                fire_count += 1

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf,
                    "cls_id": cls_id,
                    "cls_name": class_names.get(cls_id, str(cls_id)),
                    "is_fire": is_fire,
                    "centroid": (int(cx), int(cy)),
                }
            )
    return detections, fire_count


def draw_camera_panel(frame, panel_title, calib_title, detections, is_undistort):
    """
    Render anotasi bounding box, centroid crosshair, dan label kamera pada panel tunggal.
    """
    annotated = frame.copy()

    # Gambar Bounding Boxes & Centroids
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        conf = det["conf"]
        cname = det["cls_name"]
        is_fire = det["is_fire"]

        color = (0, 0, 255) if is_fire else (0, 255, 128)
        thickness = 3 if is_fire else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        # Centroid Crosshair
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
        cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 2)
        cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 2)

        # Label Text
        label = f"{cname} {conf:.2f} ({cx}, {cy})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - 24)),
            (x1 + tw + 6, max(24, y1)),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Panel Header Overlay
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w, 42), (25, 25, 25), -1)
    status_str = "UNDISTORT ON" if is_undistort else "RAW (NO CALIB)"
    header_text = f"{panel_title} | Matriks: {calib_title} [{status_str}]"
    cv2.putText(annotated, header_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    return annotated


def draw_stereo_hud(
    dashboard,
    fps,
    fire1_cnt,
    fire2_cnt,
    c1_centroid,
    c2_centroid,
    cam1_idx,
    cam2_idx,
    model_name,
    fire_only,
):
    """
    Render HUD bar global pada dashboard stereo gabungan.
    """
    dh, dw = dashboard.shape[:2]

    # Bar banner atas semi-transparan
    overlay = dashboard.copy()
    cv2.rectangle(overlay, (0, 0), (dw, 60), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, dashboard, 0.25, 0, dashboard)

    # Status Deteksi Api
    if fire1_cnt > 0 or fire2_cnt > 0:
        badge = f" [!] API TERDETEKSI: Cam1={fire1_cnt} | Cam2={fire2_cnt} "
        cv2.putText(dashboard, badge, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        badge = "[OK] STEREO MONITORING AMAN"
        cv2.putText(dashboard, badge, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Perhitungan Disparitas Stereo (jika kedua kamera mendeteksi titik pusat api)
    stereo_info = ""
    if c1_centroid is not None and c2_centroid is not None:
        u1, v1 = c1_centroid
        u2, v2 = c2_centroid
        disparity_u = u1 - u2
        vert_err = abs(v1 - v2)
        stereo_info = f" | Disparitas: dU={disparity_u:+.1f}px (dV={vert_err:.1f}px)"

    # Info Status Sistem
    filter_str = "HANYA API" if fire_only else "SEMUA OBJEK"
    hud_text = (
        f"FPS: {fps:.1f} | Cam1: [{cam1_idx}] | Cam2: [{cam2_idx}] | "
        f"Filter: [{filter_str}]{stereo_info}"
    )
    cv2.putText(dashboard, hud_text, (480, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)

    # Bar panduan kontrol bawah
    cv2.rectangle(dashboard, (0, dh - 30), (dw, dh), (15, 15, 15), -1)
    controls = (
        "[Q]: Keluar | [X]: TUKAR KAMERA 1 <-> 2 | [F]: Filter Api/Semua | "
        "[0]: Toggle Kelas 0 sbg Api | [U]: Toggle Undistort | [S]: Screenshot Stereo"
    )
    cv2.putText(dashboard, controls, (20, dh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)


def main():
    args = parse_arguments()

    if args.scan:
        scan_available_cameras()
        return

    print("\n" + "=" * 65)
    print(" MEMULAI SISTEM DETEKSI API DUAL-WEBCAM (STEREO)")
    print("=" * 65)
    print(f" Kamera 1 Target: Indeks {args.cam1} (Kalibrasi: {args.calib1})")
    print(f" Kamera 2 Target: Indeks {args.cam2} (Kalibrasi: {args.calib2})")
    print(f" Model YOLO     : {args.model}")
    print("=" * 65 + "\n")

    # 1. Buka Kedua Webcam dengan Akselerasi Hardware MJPG (Multi-threaded)
    try:
        cam1 = ThreadedCamera(
            args.cam1, args.width, args.height, target_fps=args.fps, label="Kamera 1"
        )
    except Exception as e:
        print(f"[FATAL] Gagal membuka Kamera 1: {e}")
        scan_available_cameras()
        return

    try:
        cam2 = ThreadedCamera(
            args.cam2, args.width, args.height, target_fps=args.fps, label="Kamera 2"
        )
    except Exception as e:
        print(f"[FATAL] Gagal membuka Kamera 2: {e}")
        cam1.release()
        scan_available_cameras()
        return

    # 2. Muat Kalibrasi untuk Masing-masing Kamera
    is_undistort = not args.no_undistort
    map1_c1, map2_c1 = None, None
    map1_c2, map2_c2 = None, None

    try:
        K1, D1, new_K1, map1_c1, map2_c1, _ = load_calibration(
            args.calib1, cam1.actual_w, cam1.actual_h, alpha=args.alpha
        )
    except Exception as e:
        print(f"[WARNING] Gagal memuat kalibrasi Kamera 1 ({e}).")

    try:
        K2, D2, new_K2, map1_c2, map2_c2, _ = load_calibration(
            args.calib2, cam2.actual_w, cam2.actual_h, alpha=args.alpha
        )
    except Exception as e:
        print(f"[WARNING] Gagal memuat kalibrasi Kamera 2 ({e}).")

    # 3. Muat Model YOLO
    try:
        model, class_names, fire_class_ids, model_name = setup_yolo_model(
            args.model, treat_cls0_as_fire=args.treat_cls0_as_fire
        )
    except Exception as e:
        print(f"[FATAL] Gagal memuat model YOLO: {e}")
        cam1.release()
        cam2.release()
        return

    # Status State Runtime (dapat ditukar secara langsung tanpa restart)
    cam1_idx = cam1.cam_index
    cam2_idx = cam2.cam_index
    calib1_name = os.path.basename(args.calib1)
    calib2_name = os.path.basename(args.calib2)
    fire_only_mode = not args.show_all
    is_swapped = False

    # 4. Buat Jendela GUI Stereo
    window_name = "Sistem Deteksi Api Stereo - Kamera 1 & Kamera 2 (Dual Calibrated)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 520)

    print("\n" + "=" * 65)
    print(" KONTROL RUNTIME:")
    print("   'x'           : TUKAR Kamera 1 <-> Kamera 2 (Secara instan di layar)")
    print("   'f'           : Toggle Filter (Hanya Api <-> Semua Objek)")
    print("   '0' (nol)     : Toggle Kelas 0 sbg Api (jika bobot mengindeks api di kelas 0)")
    print("   'u'           : Toggle Koreksi Undistort Lensa")
    print("   's'           : Simpan screenshot stereo gabungan")
    print("   'q' atau 'ESC': Keluar")
    print("=" * 65 + "\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                time.sleep(0.003)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Logika Swap Kamera jika pengguna menekan 'x'
            if not is_swapped:
                f_left, f_right = frame1, frame2
                m1_left, m2_left = map1_c1, map2_c1
                m1_right, m2_right = map1_c2, map2_c2
                title_left = f"KAMERA 1 (Idx {cam1_idx})"
                title_right = f"KAMERA 2 (Idx {cam2_idx})"
                cal_left = calib1_name
                cal_right = calib2_name
            else:
                f_left, f_right = frame2, frame1
                m1_left, m2_left = map1_c2, map2_c2
                m1_right, m2_right = map1_c1, map2_c1
                title_left = f"KAMERA 2 (Idx {cam2_idx} -> Ditukar ke Kiri)"
                title_right = f"KAMERA 1 (Idx {cam1_idx} -> Ditukar ke Kanan)"
                cal_left = calib2_name
                cal_right = calib1_name

            # 5. Undistort Masing-masing Aliran Kamera
            if is_undistort and m1_left is not None and m2_left is not None:
                undist_left = cv2.remap(f_left, m1_left, m2_left, interpolation=cv2.INTER_LINEAR)
            else:
                undist_left = f_left.copy()

            if is_undistort and m1_right is not None and m2_right is not None:
                undist_right = cv2.remap(f_right, m1_right, m2_right, interpolation=cv2.INTER_LINEAR)
            else:
                undist_right = f_right.copy()

            # 6. Inferensi YOLO Batch (Sekaligus 2 frame untuk kecepatan maksimal)
            batch_results = model.predict(
                source=[undist_left, undist_right],
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            dets_left, fire_cnt_left = process_detections(
                batch_results[0].boxes, class_names, fire_class_ids, fire_only_mode
            )
            dets_right, fire_cnt_right = process_detections(
                batch_results[1].boxes, class_names, fire_class_ids, fire_only_mode
            )

            # Ambil centroid api pertama untuk perhitungan stereo disparity
            c1_centroid = next((d["centroid"] for d in dets_left if d["is_fire"]), None)
            c2_centroid = next((d["centroid"] for d in dets_right if d["is_fire"]), None)

            # 7. Gambar Panel Kiri & Kanan
            panel_left = draw_camera_panel(
                undist_left, title_left, cal_left, dets_left, is_undistort
            )
            panel_right = draw_camera_panel(
                undist_right, title_right, cal_right, dets_right, is_undistort
            )

            # 8. Gabungkan Secara Horizontal Menjadi Dashboard Stereo
            combined = np.hstack((panel_left, panel_right))

            # Resize proporsional agar muat dengan nyaman di monitor (target lebar ~1600px)
            target_w = 1600
            scale = target_w / float(combined.shape[1])
            target_h = int(combined.shape[0] * scale)
            display_dashboard = cv2.resize(combined, (target_w, target_h))

            # Tambahkan Global HUD Bar
            draw_stereo_hud(
                display_dashboard,
                fps,
                fire_cnt_left,
                fire_cnt_right,
                c1_centroid,
                c2_centroid,
                cam1_idx,
                cam2_idx,
                model_name,
                fire_only_mode,
            )

            cv2.imshow(window_name, display_dashboard)

            # 9. Penanganan Input Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[INFO] Menutup sistem deteksi dual-kamera...")
                break
            elif key == ord("x"):  # Tukar posisi Kamera 1 dan Kamera 2
                is_swapped = not is_swapped
                status = "DITUKAR (Swapped)" if is_swapped else "DEFAULT (Normal)"
                print(f"[TUKAR KAMERA] Status Penempatan: {status}")
            elif key == ord("f"):  # Toggle Filter
                fire_only_mode = not fire_only_mode
                mode_str = "HANYA API" if fire_only_mode else "SEMUA OBJEK"
                print(f"[FILTER] Mode: {mode_str}")
            elif key == ord("0"):  # Toggle Kelas 0 sbg Api
                if 0 in fire_class_ids:
                    fire_class_ids.remove(0)
                    class_names[0] = "person"
                    print("[TOGGLE] Kelas 0 dikembalikan ke label 'person'.")
                else:
                    fire_class_ids.append(0)
                    class_names[0] = "fire"
                    print("[TOGGLE] Kelas 0 sekarang DIPERLAKUKAN SEBAGAI 'fire/api'!")
            elif key == ord("u"):  # Toggle Undistort
                is_undistort = not is_undistort
                status = "AKTIF" if is_undistort else "NONAKTIF"
                print(f"[UNDISTORT] Status Koreksi: {status}")
            elif key == ord("s"):  # Simpan Screenshot
                fname = f"stereo_capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, display_dashboard)
                print(f"[SCREENSHOT] Dashboard stereo tersimpan: {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Interupsi terdeteksi. Menghentikan program...")

    finally:
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()
        print("[INFO] Kedua webcam dan jendela OpenCV berhasil ditutup dengan bersih.")


if __name__ == "__main__":
    main()
