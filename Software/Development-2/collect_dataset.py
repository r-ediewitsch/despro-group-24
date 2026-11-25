"""
collect_dataset.py
------------------
Tool akuisisi data visual lapangan sesuai panduan STEPS.md Fase 1:
- Mendukung kamera live (webcam/USB cam) atau file video.
- Koreksi distorsi lensa otomatis menggunakan calib_cam1.npz / calib_cam2.npz (undistort).
- Mode sampling otomatis berbasis interval (default 1.5 detik per frame) untuk mencegah redundansi.
- Mode manual snapshot dengan tombol SPACE.
- Pengkategorian on-the-fly: 'positive' (api & asap) vs 'negative' (hard negative background).
- Menghasilkan direktori mentah:
    dataset_raw/positive/
    dataset_raw/negative/
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2


class DatasetCollector:
    def __init__(self, source=0, calib_file=None, output_dir="dataset_raw",
                 interval_sec=1.5, width=1280, height=720, max_samples=None):
        self.source = source
        self.output_dir = output_dir
        self.interval_sec = interval_sec
        self.width = width
        self.height = height
        self.max_samples = max_samples
        self.should_stop = False

        # Output folders
        self.pos_dir = os.path.join(output_dir, "positive")
        self.neg_dir = os.path.join(output_dir, "negative")
        os.makedirs(self.pos_dir, exist_ok=True)
        os.makedirs(self.neg_dir, exist_ok=True)

        # Counting existing files
        self.pos_count = len([f for f in os.listdir(self.pos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.neg_count = len([f for f in os.listdir(self.neg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Calibration setup
        self.map1 = None
        self.map2 = None
        if calib_file and os.path.exists(calib_file):
            print(f"[Collector] Memuat parameter kalibrasi: {calib_file}")
            data = np.load(calib_file)
            mtx = data["camera_matrix"] if "camera_matrix" in data else data.get("K", None)
            dist = (
                data["dist_coeffs"] if "dist_coeffs" in data 
                else (data["dist_coeff"] if "dist_coeff" in data else data.get("D", None))
            )

            if mtx is not None and dist is not None:
                new_K, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), alpha=0.0, newImgSize=(width, height))
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    mtx, dist, None, new_K, (width, height), cv2.CV_32FC1
                )
                print("[Collector] Koreksi distorsi lensa aktif (Undistort OK).")
            else:
                print("[Collector WARNING] Format matriks kalibrasi tidak lengkap. Berjalan tanpa undistort.")
        else:
            if calib_file:
                print(f"[Collector] Peringatan: File {calib_file} tidak ditemukan. Berjalan tanpa undistort.")

        # State
        self.current_mode = "positive"  # 'positive' or 'negative'
        self.auto_capture = True
        self.last_capture_time = time.time()
        self.last_saved_info = ""
        self.flash_timer = 0.0

    def undistort(self, frame):
        if self.map1 is not None and self.map2 is not None:
            return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        return frame

    def save_frame(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        msec = int((time.time() % 1) * 1000)
        filename = f"{self.current_mode}_{timestamp}_{msec:03d}.jpg"

        if self.current_mode == "positive":
            target_path = os.path.join(self.pos_dir, filename)
            self.pos_count += 1
        else:
            target_path = os.path.join(self.neg_dir, filename)
            self.neg_count += 1

        cv2.imwrite(target_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.last_saved_info = f"Tersimpan: {filename} ({self.current_mode})"
        self.flash_timer = time.time() + 0.3
        self.last_capture_time = time.time()
        print(f"[Capture] {target_path} (Pos: {self.pos_count}, Neg: {self.neg_count})")

        total = self.pos_count + self.neg_count
        if self.max_samples and total >= self.max_samples:
            print(f"\n[TARGET TERCAPAI] Mencapai kuota batas {self.max_samples} sampel.")
            self.should_stop = True

    def run(self):
        # Open capture device
        try:
            cam_idx = int(self.source)
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        except ValueError:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[Collector ERROR] Tidak dapat membuka sumber video: {self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        win_name = "Pengumpul Dataset - Deezpro (Fase 1 STEPS.md)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1024, 600)

        print("\n" + "=" * 60)
        print("PENGUMPUL DATASET AKTIF")
        print("Kontrol Keyboard:")
        print("  [SPACE]     : Ambil snapshot frame saat ini secara manual")
        print("  [A]         : Toggle Auto-Interval Capture (ON / OFF)")
        print("  [M]         : Ganti Mode antara POSITIVE (Api/Asap) & NEGATIVE (Background)")
        print("  [+/-]       : Tambah / Kurangi interval waktu otomatis")
        print("  [Q / ESC]   : Keluar")
        print("=" * 60 + "\n")

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[Collector] Aliran video selesai atau terputus.")
                break

            # Undistort lens geometry
            frame = self.undistort(raw_frame)
            display = frame.copy()
            now = time.time()

            # Auto interval capture logic
            time_since_last = now - self.last_capture_time
            countdown = max(0.0, self.interval_sec - time_since_last)

            if self.auto_capture and time_since_last >= self.interval_sec:
                self.save_frame(frame)

            if self.should_stop:
                print(f"[Collector] Selesai otomatis: Kuota {self.max_samples} sampel terpenuhi.")
                break

            # Visual feedback (flash on capture)
            if now < self.flash_timer:
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), 8)

            # Draw HUD
            self._draw_hud(display, countdown)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(10) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key == 32:  # SPACE
                self.save_frame(frame)
            elif key in (ord('m'), ord('M')):
                self.current_mode = "negative" if self.current_mode == "positive" else "positive"
                print(f"[Collector] Mode diubah ke: {self.current_mode.upper()}")
            elif key in (ord('a'), ord('A')):
                self.auto_capture = not self.auto_capture
                self.last_capture_time = time.time()
                print(f"[Collector] Auto capture: {'ON' if self.auto_capture else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                self.interval_sec = round(self.interval_sec + 0.5, 1)
                print(f"[Collector] Interval dinaikkan: {self.interval_sec}s")
            elif key in (ord('-'), ord('_')):
                self.interval_sec = max(0.5, round(self.interval_sec - 0.5, 1))
                print(f"[Collector] Interval diturunkan: {self.interval_sec}s")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nPengumpulan selesai! Total tersimpan: {self.pos_count} positif, {self.neg_count} negatif.")

    def _draw_hud(self, img, countdown):
        h, w = img.shape[:2]

        # Top banner overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Mode indicator badge
        mode_color = (0, 140, 255) if self.current_mode == "positive" else (180, 180, 50)
        mode_text = f"MODE: [{self.current_mode.upper()}]"
        cv2.putText(img, mode_text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 2)

        # Sampling stats
        auto_str = f"AUTO: ON ({countdown:.1f}s / {self.interval_sec}s)" if self.auto_capture else "AUTO: OFF (Manual)"
        cv2.putText(img, auto_str, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # Stats ratio (Target: 70% positive, 30% negative)
        total = self.pos_count + self.neg_count
        pos_pct = (self.pos_count / total * 100) if total > 0 else 0
        neg_pct = (self.neg_count / total * 100) if total > 0 else 0
        stats_str = f"Pos: {self.pos_count} ({pos_pct:.0f}%) | Neg: {self.neg_count} ({neg_pct:.0f}%) [Total: {total}]"
        cv2.putText(img, stats_str, (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)

        # Subtitle info
        if self.last_saved_info:
            cv2.putText(img, self.last_saved_info, (15, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

        # Bottom help bar
        help_bar = "[M] Switch Mode  |  [A] Auto On/Off  |  [SPACE] Capture  |  [+/-] Interval  |  [Q] Keluar"
        cv2.putText(img, help_bar, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="Tool Akuisisi Dataset Kamera Deezpro (Fase 1 STEPS.md)")
    parser.add_argument("--source", default="0", help="Index webcam (0, 1, ...) atau path file video")
    parser.add_argument("--calib", default=None, help="Path ke file matriks kalibrasi (.npz), misal config/calib_cam1.npz")
    parser.add_argument("--out", default="dataset_raw", help="Folder output data mentah (default: dataset_raw)")
    parser.add_argument("--interval", type=float, default=1.5, help="Interval pengambilan otomatis dalam detik (default: 1.5)")
    parser.add_argument("--width", type=int, default=1280, help="Lebar resolusi kamera (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Tinggi resolusi kamera (default: 720)")
    parser.add_argument("--max-samples", type=int, default=None, help="Batas maksimal jumlah sampel sebelum berhenti otomatis (opsional)")
    args = parser.parse_args()

    collector = DatasetCollector(
        source=args.source,
        calib_file=args.calib,
        output_dir=args.out,
        interval_sec=args.interval,
        width=args.width,
        height=args.height,
        max_samples=args.max_samples
    )
    collector.run()


if __name__ == "__main__":
    main()
