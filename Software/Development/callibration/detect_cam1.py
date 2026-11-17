import os
import sys
import time
import argparse
import zipfile
import threading
import numpy as np
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Deteksi Objek Real-Time YOLO menggunakan Webcam Kamera 1 dengan Kalibrasi calib_cam1.npz."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Indeks webcam (default: 1 untuk webcam Kamera 1).",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="calib_cam1.npz",
        help="File kalibrasi kamera (default: calib_cam1.npz).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path file model YOLO (.pt). Default: best.pt",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate kamera (default: 30 FPS).",
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
        help="Confidence threshold deteksi YOLO (default: 0.4).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Resolusi lebar frame kamera (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Resolusi tinggi frame kamera (default: 1080).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Parameter crop undistort (0.0: buang tepi hitam, 1.0: pertahankan seluruh piksel).",
    )
    parser.add_argument(
        "--fire-only",
        action="store_true",
        help="Hanya tampilkan deteksi api jika model memiliki kelas api.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Tampilkan seluruh kelas objek yang terdeteksi.",
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
    return parser.parse_args()


def resolve_path(path_str):
    """
    Menyelesaikan path relatif baik dari CWD maupun dari direktori skrip ini berada.
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
    Memuat matriks kalibrasi calib_cam1.npz dan menghitung map undistort/rectify CV_16SC2
    untuk efisiensi komputasi real-time.
    """
    calib_file = resolve_path(calib_file)
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {calib_file}")

    data = np.load(calib_file)
    K = data["camera_matrix"]
    D = data["dist_coeffs"]
    mean_err = data.get("mean_error", None)

    print(f"[INFO] Kalibrasi dimuat: {os.path.basename(calib_file)}")
    print(f"       Camera Matrix (K):\n{K}")
    print(f"       Distortion Coeffs (D): {D.ravel()}")
    if mean_err is not None:
        print(f"       Mean Reprojection Error: {float(mean_err):.4f} px")

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (target_w, target_h), alpha=alpha, newImgSize=(target_w, target_h)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (target_w, target_h), cv2.CV_16SC2
    )
    return K, D, new_K, map1, map2, roi


class ThreadedCamera:
    """
    Kamera real-time berbasis multi-threading dengan akselerasi hardware MJPG.
    Menghilangkan masalah drop FPS ke 5 FPS pada USB 2.0 dengan kompresi hardware.
    """

    def __init__(self, cam_index, target_w=1920, target_h=1080, target_fps=30):
        self.cam_index = cam_index
        print(f"[INFO] Inisialisasi Kamera 1 pada indeks {cam_index} (MJPG @ {target_fps} FPS)...")

        # 1. Buka device dengan backend DirectShow (Windows)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        # Fallback ke kamera 0 jika indeks target gagal terbuka
        if not self.cap.isOpened() and cam_index != 0:
            print(f"[WARNING] Kamera {cam_index} gagal dibuka! Fallback ke kamera 0...")
            self.cam_index = 0
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError(f"Gagal membuka webcam! Pastikan kabel USB kamera terhubung.")

        # 2. PENTING: Set FOURCC ke MJPG TERLEBIH DAHULU sebelum set resolusi & FPS!
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
            f"[INFO] Kamera 1 Siap (Indeks {self.cam_index}) - "
            f"Resolusi: {self.actual_w}x{self.actual_h} @ {reported_fps:.1f} FPS (Codec: MJPG)"
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
    Memuat model YOLO dan menganalisis daftar kelas yang tersedia.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n" + "=" * 60)
        print("[ERROR] Modul 'ultralytics' belum terpasang!")
        print("Silakan jalankan: pip install ultralytics")
        print("=" * 60 + "\n")
        sys.exit(1)

    model_path = resolve_path(model_path)

    # Kemas otomatis jika path berupa direktori terekstrak
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
        raise FileNotFoundError(f"File model YOLO tidak ditemukan: {model_path}")

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model = YOLO(model_path)
    class_names = dict(model.names) if hasattr(model, "names") else {}

    # Identifikasi kelas Api dan Asap secara terpisah
    fire_keywords = {"fire", "api", "flame"}
    smoke_keywords = {"smoke", "asap"}
    exclude_keywords = {"hydrant"}

    fire_class_ids = []
    smoke_class_ids = []

    if treat_cls0_as_fire and 0 in class_names and "smoke" not in str(class_names[0]).lower():
        fire_class_ids.append(0)
        class_names[0] = "fire"

    for cls_id, name in class_names.items():
        name_lower = str(name).lower()
        if any(ex in name_lower for ex in exclude_keywords):
            continue
        if any(kw in name_lower for kw in fire_keywords):
            if cls_id not in fire_class_ids:
                fire_class_ids.append(cls_id)
        elif any(kw in name_lower for kw in smoke_keywords):
            if cls_id not in smoke_class_ids:
                smoke_class_ids.append(cls_id)

    print("\n" + "=" * 65)
    print(" [VERIFIKASI MODEL YOLO]")
    print(f" File Model    : {os.path.basename(model_path)} ({model_size_mb:.2f} MB)")
    print(f" Jumlah Kelas  : {len(class_names)}")
    if fire_class_ids or smoke_class_ids:
        if fire_class_ids:
            f_names = ", ".join([f"{cid}: '{class_names[cid]}'" for cid in fire_class_ids])
            print(f" Target Api    : Kelas ID [{f_names}]")
        if smoke_class_ids:
            s_names = ", ".join([f"{cid}: '{class_names[cid]}'" for cid in smoke_class_ids])
            print(f" Target Asap   : Kelas ID [{s_names}]")
    else:
        print(" Target Bahaya : Model tidak memiliki label 'fire'/'smoke' (menggunakan label COCO).")
        print("                 Tekan '0' pada keyboard jika Kelas 0 ingin diperlakukan sbg Api.")
    print("=" * 65 + "\n")

    return model, class_names, fire_class_ids, smoke_class_ids, os.path.basename(model_path)


def draw_hud(
    frame,
    fps,
    is_undistort,
    view_mode,
    fire_count,
    smoke_count,
    total_detections,
    cam_id,
    model_name,
    fire_only_mode,
    calib_name="calib_cam1.npz",
):
    """
    Menggambar informasi status (HUD) pada window OpenCV.
    """
    h, w = frame.shape[:2]

    # Bar banner atas semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # 1. Status Deteksi Api & Asap
    if fire_count > 0 and smoke_count > 0:
        badge_text = f" [!] API & ASAP TERDETEKSI (Api:{fire_count}, Asap:{smoke_count}) "
        badge_color = (0, 0, 255)  # Merah terang
    elif fire_count > 0:
        badge_text = f" [!] API TERDETEKSI ({fire_count}) "
        badge_color = (0, 0, 255)  # Merah terang
    elif smoke_count > 0:
        badge_text = f" [!] ASAP TERDETEKSI ({smoke_count}) "
        badge_color = (0, 165, 255)  # Oranye / Amber
    else:
        badge_text = "[OK] MONITORING AMAN"
        badge_color = (0, 255, 0)  # Hijau

    cv2.putText(frame, badge_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, badge_color, 2)

    # 2. Status Kalibrasi & Filter
    filter_str = "HANYA API & ASAP" if fire_only_mode else "SEMUA OBJEK"
    undist_str = f"AKTIF ({calib_name})" if is_undistort else "OFF (Raw)"
    info_text = (
        f"FPS: {fps:.1f} | Cam: {cam_id} | Model: {model_name} | "
        f"Filter: [{filter_str}] | Undistort: [{undist_str}]"
    )
    cv2.putText(frame, info_text, (430, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

    # Bar panduan kontrol bawah
    cv2.rectangle(frame, (0, h - 28), (w, h), (15, 15, 15), -1)
    help_text = (
        "[Q/ESC]: Keluar | [F]: Filter Api-Asap/Semua | [0]: Toggle Kelas-0 sbg Api | "
        "[U]: Toggle Undistort | [C]: Dual View (Raw vs Undistort) | [S]: Screenshot"
    )
    cv2.putText(frame, help_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def main():
    args = parse_arguments()

    # 1. Buka Webcam Kamera 1 (Hardware MJPG 30 FPS, Multi-threaded)
    try:
        cap = ThreadedCamera(
            args.camera, args.width, args.height, target_fps=args.fps
        )
        cam_index = cap.cam_index
        actual_w = cap.actual_w
        actual_h = cap.actual_h
    except Exception as e:
        print(f"[FATAL] Gagal membuka kamera: {e}")
        return

    # 2. Muat Kalibrasi Undistort calib_cam1.npz
    is_undistort = not args.no_undistort
    map1, map2 = None, None
    calib_file = args.calib

    try:
        K, D, new_K, map1, map2, roi = load_calibration(
            calib_file, actual_w, actual_h, alpha=args.alpha
        )
    except Exception as e:
        print(f"[WARNING] Gagal memuat kalibrasi ({e}). Melanjutkan tanpa undistort.")
        is_undistort = False

    # 3. Muat Model YOLO (best.pt / yolo26n.pt)
    try:
        model, class_names, fire_class_ids, smoke_class_ids, model_name = setup_yolo_model(
            args.model, treat_cls0_as_fire=args.treat_cls0_as_fire
        )
    except Exception as e:
        print(f"[FATAL] Gagal memuat model YOLO: {e}")
        cap.release()
        return

    # Mode filter: jika fire_only aktif, hanya tampilkan bahaya (api & asap)
    fire_only_mode = args.fire_only and not args.show_all

    # 4. Inisialisasi Jendela GUI OpenCV
    window_name = f"Sistem Deteksi YOLO - Kamera 1 (Undistort: {os.path.basename(calib_file)} & Model: {model_name})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 65)
    print(" SISTEM DETEKSI KAMERA 1 BERJALAN")
    print(f" Kamera Aktif : Indeks {cam_index} (MJPG @ {args.fps} FPS)")
    print(f" Matriks Calib: {os.path.basename(calib_file)}")
    print(f" Model YOLO   : {model_name}")
    print(f" Mode Filter  : {'HANYA API & ASAP' if fire_only_mode else 'SEMUA OBJEK'}")
    print("=" * 65)
    print(" Kontrol Keyboard:")
    print("   'q' atau 'ESC' : Keluar dari program")
    print("   'f'           : Toggle Filter (Hanya Api & Asap <-> Tampilkan Semua)")
    print("   '0' (nol)     : Toggle Kelas 0 diperlakukan sebagai 'fire/api'")
    print("   'u'           : Mengaktifkan/menonaktifkan proses undistort")
    print("   'c'           : Berpindah tampilan (Single View vs Side-by-Side)")
    print("   's'           : Simpan tangkapan layar saat ini")
    print("=" * 65 + "\n")

    view_mode = 0  # 0: Single View (Undistorted), 1: Side-by-Side (Raw vs Undistorted)
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.005)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # 5. Koreksi Distorsi Lensa (Undistort calib_cam1)
            if is_undistort and map1 is not None and map2 is not None:
                processed_frame = cv2.remap(
                    raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR
                )
            else:
                processed_frame = raw_frame.copy()

            # 6. Inferensi Deteksi Objek Murni YOLO
            results = model.predict(
                source=processed_frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            detections = []
            fire_count = 0
            smoke_count = 0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    is_fire = cls_id in fire_class_ids
                    is_smoke = cls_id in smoke_class_ids
                    is_hazard = is_fire or is_smoke

                    # Jika mode filter aktif dan objek bukan api/asap, lewati
                    if fire_only_mode and not is_hazard:
                        continue

                    if is_fire:
                        fire_count += 1
                    if is_smoke:
                        smoke_count += 1

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    detections.append(
                        {
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "conf": conf,
                            "cls_id": cls_id,
                            "cls_name": class_names.get(cls_id, str(cls_id)),
                            "is_fire": is_fire,
                            "is_smoke": is_smoke,
                            "centroid": (int(cx), int(cy)),
                        }
                    )

            # 7. Gambar Bounding Box & Centroid Crosshair
            display_frame = processed_frame.copy()

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["centroid"]
                conf = det["conf"]
                cname = det["cls_name"]
                is_fire = det["is_fire"]
                is_smoke = det["is_smoke"]

                # Warna: Merah untuk api, Oranye/Amber untuk asap, Hijau neon untuk objek lain
                if is_fire:
                    box_color = (0, 0, 255)      # Merah terang
                    crosshair_color = (0, 255, 255)  # Kuning
                    line_thickness = 3
                elif is_smoke:
                    box_color = (0, 165, 255)    # Oranye / Amber
                    crosshair_color = (0, 200, 255)  # Emas cerah
                    line_thickness = 2
                else:
                    box_color = (0, 255, 128)    # Hijau neon
                    crosshair_color = (0, 255, 255)
                    line_thickness = 2

                # Bounding Box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, line_thickness)

                # Titik Pusat Centroid & Crosshair (untuk targeting motor stepper/nozzle)
                cv2.circle(display_frame, (cx, cy), 5, crosshair_color, -1)
                cv2.line(display_frame, (cx - 10, cy), (cx + 10, cy), crosshair_color, 1)
                cv2.line(display_frame, (cx, cy - 10), (cx, cy + 10), crosshair_color, 1)

                # Label Bounding Box
                label = f"{cname} {conf:.2f} ({cx}, {cy})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    display_frame,
                    (x1, max(0, y1 - 22)),
                    (x1 + tw + 6, max(22, y1)),
                    box_color,
                    -1,
                )
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 3, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 8. Tampilan Layar: Single View atau Side-by-Side (Raw vs Undistort)
            if view_mode == 1:
                raw_labeled = raw_frame.copy()
                cv2.putText(
                    raw_labeled,
                    "CITRA ASLI (DISTORSI KAMERA 1)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "HASIL UNDISTORT & DETEKSI YOLO",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                combined = np.hstack((raw_labeled, display_frame))
                target_disp_w = 1600
                scale_ratio = target_disp_w / float(combined.shape[1])
                target_disp_h = int(combined.shape[0] * scale_ratio)
                out_frame = cv2.resize(combined, (target_disp_w, target_disp_h))
            else:
                out_frame = display_frame

            # Tambahkan HUD di bagian atas dan bawah
            draw_hud(
                out_frame,
                fps,
                is_undistort,
                view_mode,
                fire_count,
                smoke_count,
                len(detections),
                cam_index,
                model_name,
                fire_only_mode,
                calib_name=os.path.basename(calib_file),
            )

            # 9. Tampilkan Jendela OpenCV
            cv2.imshow(window_name, out_frame)

            # 10. Penanganan Event Tombol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' atau ESC untuk keluar
                print("\n[INFO] Menutup program atas permintaan pengguna...")
                break
            elif key == ord("f"):  # Toggle Filter Api & Asap vs Semua Objek
                fire_only_mode = not fire_only_mode
                mode_str = "HANYA API & ASAP" if fire_only_mode else "SEMUA OBJEK"
                print(f"[TOGGLE] Mode Filter: {mode_str}")
            elif key == ord("0"):  # Toggle Kelas 0 sbg Fire/Api
                if 0 in fire_class_ids:
                    fire_class_ids.remove(0)
                    class_names[0] = "person"
                    print("[TOGGLE] Kelas 0 dikembalikan ke label 'person' (COCO).")
                else:
                    fire_class_ids.append(0)
                    class_names[0] = "fire"
                    print("[TOGGLE] Kelas 0 sekarang DIPERLAKUKAN SEBAGAI 'fire/api'!")
            elif key == ord("u"):  # Toggle Undistort
                is_undistort = not is_undistort
                status = "AKTIF" if is_undistort else "NONAKTIF"
                print(f"[TOGGLE] Undistort: {status}")
            elif key == ord("c"):  # Toggle View Mode
                view_mode = 1 if view_mode == 0 else 0
                mode_str = "Berdampingan (Dual)" if view_mode == 1 else "Normal (Single)"
                print(f"[TOGGLE] Mode Tampilan: {mode_str}")
            elif key == ord("s"):  # Simpan Screenshot
                fname = f"capture_cam1_{int(time.time())}.jpg"
                cv2.imwrite(fname, out_frame)
                print(f"[SCREENSHOT] Berhasil disimpan sebagai: {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Interupsi terdeteksi. Keluar program...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Kamera 1 dan jendela OpenCV berhasil ditutup dengan bersih.")


if __name__ == "__main__":
    main()
