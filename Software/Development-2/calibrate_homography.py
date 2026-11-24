"""
=====================================================================
DEEZPRO - INTERACTIVE HOMOGRAPHY CALIBRATOR & VERIFIER
=====================================================================
File: calibrate_homography.py
Deskripsi: Program interaktif untuk melakukan kalibrasi homografi planar
            pada Kamera 1 atau Kamera 2. Mengoreksi distorsi lensa
            (undistort) secara real-time, memungkinkan penandaan titik
            marker via klik mouse, menghitung matriks H, menampilkan
            pratinjau Bird's-Eye View, serta memverifikasi koordinat fisik
            lantai secara langsung.
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional
import numpy as np
import cv2
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vision.homography import HomographyEngine
from src.integration.system_logger import setup_logger

logger = setup_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Kalibrasi Homografi Interaktif Kamera -> Bidang Fisik Lantai (Deezpro)."
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
        help="Override index hardware OpenCV (0, 1, 2...). Jika tidak diisi, membaca dari config/system_config.yaml.",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path berkas kalibrasi intrinsik Zhang (.npz). Default: config/calib_cam<camera>.npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path berkas output penyimpanan homografi (.npz). Default: config/homography_cam<camera>.npz",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Gunakan berkas citra statis alih-alih streaming live webcam.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Aktifkan langsung mode verifikasi menggunakan berkas homografi yang sudah ada.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Resolusi lebar kamera (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Resolusi tinggi kamera (default: 1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS kamera (default: 30).",
    )
    return parser.parse_args()


class HomographyGuiApp:
    def __init__(self, args):
        self.args = args
        self.cam_id = args.camera

        # Tentukan path kalibrasi intrinsik dan output
        cfg_dir = os.path.join(current_dir, "config")
        if args.calib:
            self.calib_path = args.calib
        else:
            self.calib_path = os.path.join(cfg_dir, f"calib_cam{self.cam_id}.npz")

        if args.output:
            self.output_path = args.output
        else:
            self.output_path = os.path.join(cfg_dir, f"homography_cam{self.cam_id}.npz")

        # Resolusi index hardware kamera (OpenCV VideoCapture)
        self.device_index = args.source
        if self.device_index is None:
            sys_cfg_path = os.path.join(cfg_dir, "system_config.yaml")
            if os.path.exists(sys_cfg_path):
                try:
                    with open(sys_cfg_path, "r") as f:
                        cfg_yaml = yaml.safe_load(f) or {}
                        cam_key = f"cam{self.cam_id}"
                        if "cameras" in cfg_yaml and cam_key in cfg_yaml["cameras"]:
                            self.device_index = int(cfg_yaml["cameras"][cam_key].get("index", self.cam_id))
                except Exception:
                    pass
            if self.device_index is None:
                self.device_index = self.cam_id

        logger.info(f"Kamera {self.cam_id} dipetakan ke Hardware Device Index {self.device_index}")

        self.engine = HomographyEngine()

        # Load parameter undistort
        self.map1, self.map2 = None, None
        self._load_lens_calibration()

        # State GUI
        self.is_frozen = False
        self.frozen_frame = None
        self.live_frame = None
        self.points_img: List[Tuple[int, int]] = []
        self.points_world: List[Tuple[float, float]] = []
        self.window_name = f"Deezpro - Kalibrasi Homografi Kamera {self.cam_id}"
        self.is_verified_mode = args.verify

        # Load existing homography jika mode verifikasi diaktifkan
        if self.is_verified_mode:
            if os.path.exists(self.output_path):
                self.engine = HomographyEngine.load(self.output_path)
                logger.info(f"Mode Verifikasi Aktif! Berkas dimuat: {self.output_path}")
            else:
                logger.warning(f"Berkas homografi tidak ditemukan: {self.output_path}. Memulai kalibrasi normal.")
                self.is_verified_mode = False

        self.last_clicked_pos: Optional[Tuple[int, int]] = None
        self.last_predicted_world: Optional[Tuple[float, float]] = None

    def _load_lens_calibration(self):
        """Memuat matriks K & D dari calib_cam.npz untuk koreksi distorsi lensa."""
        if not os.path.exists(self.calib_path):
            # Coba cari di folder Development/callibration
            alt_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                "Development", "callibration", f"calib_cam{self.cam_id}.npz"
            )
            if os.path.exists(alt_path):
                self.calib_path = alt_path

        if os.path.exists(self.calib_path):
            data = np.load(self.calib_path)
            K = data["camera_matrix"]
            D = data["dist_coeffs"]
            w, h = self.args.width, self.args.height
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0, newImgSize=(w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
            logger.info(f"Kalibrasi lensa dimuat: {os.path.basename(self.calib_path)} (Undistort Aktif)")
        else:
            logger.warning(f"Berkas kalibrasi lensa ({self.calib_path}) tidak ditemukan. Berjalan tanpa undistort.")

    def on_mouse_click(self, event, x, y, flags, param):
        """Handler event klik mouse pada jendela OpenCV."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.is_verified_mode:
            # Mode Verifikasi: Hitung koordinat fisik dari titik yang diklik
            self.last_clicked_pos = (x, y)
            if self.engine.H is not None:
                xw, yw = self.engine.project_image_to_world(float(x), float(y))
                self.last_predicted_world = (xw, yw)
                logger.info(f"[PROYEKSI] Piksel ({x}, {y}) -> Koordinat Lantai: X={xw:.1f} cm, Y={yw:.1f} cm")
            return

        # Mode Kalibrasi: Mendaftarkan titik baru
        print(f"\n[TITIK {len(self.points_img) + 1} TERPILIH] Piksel Citra: ({x}, {y})")
        print("Pilih acuan koordinat dunia fisik lantai (cm):")
        print("  [1] Sudut Kiri-Bawah  : (10, 10)")
        print("  [2] Sudut Kanan-Bawah : (190, 10)")
        print("  [3] Sudut Kanan-Atas  : (190, 190)")
        print("  [4] Sudut Kiri-Atas   : (10, 190)")
        print("  [5] Titik Tengah      : (100, 100)")
        print("  [M] Input manual kustom (X Y)")

        choice = input("Pilihan (1-5 atau ketik 'X Y' langsung) > ").strip()

        preset_coords = {
            "1": (10.0, 10.0),
            "2": (190.0, 10.0),
            "3": (190.0, 190.0),
            "4": (10.0, 190.0),
            "5": (100.0, 100.0),
        }

        if choice in preset_coords:
            wx, wy = preset_coords[choice]
        else:
            try:
                parts = choice.split()
                if len(parts) >= 2:
                    wx, wy = float(parts[0]), float(parts[1])
                else:
                    print("[BATAL] Input tidak valid. Titik diabaikan.")
                    return
            except ValueError:
                print("[BATAL] Koordinat harus berupa angka. Titik diabaikan.")
                return

        self.points_img.append((x, y))
        self.points_world.append((wx, wy))
        self.engine.add_point_pair((float(x), float(y)), (wx, wy))
        logger.info(f"Titik #{len(self.points_img)} terdaftar: Piksel ({x}, {y}) <-> Dunia ({wx:.1f}, {wy:.1f}) cm")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

        cap = None
        if self.args.image:
            if not os.path.exists(self.args.image):
                logger.error(f"Berkas citra tidak ditemukan: {self.args.image}")
                return
            img = cv2.imread(self.args.image)
            if self.map1 is not None and self.map2 is not None:
                img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            self.frozen_frame = img
            self.is_frozen = True
        else:
            # Buka kamera dengan DirectShow
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.device_index)

            if not cap.isOpened():
                logger.error(f"Gagal membuka kamera indeks {self.device_index} (Kamera {self.cam_id})!")
                return

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            cap.set(cv2.CAP_PROP_FPS, self.args.fps)

        self._print_instructions()

        try:
            while True:
                if not self.is_frozen and cap is not None:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue

                    # Putar 180 derajat karena posisi kamera pada prototype terbalik
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                    # Undistort frame secara real-time
                    if self.map1 is not None and self.map2 is not None:
                        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                    self.live_frame = frame
                    display_frame = frame.copy()
                else:
                    display_frame = self.frozen_frame.copy()

                # Render visualisasi overlay
                self._draw_overlay(display_frame)

                cv2.imshow(self.window_name, display_frame)
                key = cv2.waitKey(20) & 0xFF

                if key in [ord("q"), 27]:  # Q atau ESC
                    break

                elif key == 32:  # SPASI (Freeze / Unfreeze)
                    if not self.args.image and self.live_frame is not None:
                        self.is_frozen = not self.is_frozen
                        if self.is_frozen:
                            self.frozen_frame = self.live_frame.copy()
                            print("\n[FREEZE] Frame dibekukan untuk penandaan titik yang presisi.")
                        else:
                            print("\n[UNFREEZE] Melanjutkan streaming video langsung.")

                elif key == ord("z"):  # Undo titik terakhir
                    if len(self.points_img) > 0:
                        removed_img = self.points_img.pop()
                        removed_w = self.points_world.pop()
                        # Rebuild engine points
                        self.engine.clear_points()
                        for p_i, p_w in zip(self.points_img, self.points_world):
                            self.engine.add_point_pair(p_i, p_w)
                        print(f"\n[UNDO] Menghapus titik terakhir: Piksel {removed_img} <-> Dunia {removed_w}")

                elif key == ord("r"):  # Reset semua titik
                    self.points_img.clear()
                    self.points_world.clear()
                    self.engine.clear_points()
                    print("\n[RESET] Seluruh titik korespondensi telah direset.")

                elif key == ord("c"):  # Hitung Homografi
                    self._compute_and_preview(display_frame)

                elif key == ord("s"):  # Simpan ke .npz
                    if self.engine.H is not None:
                        meta = {
                            "camera_id": self.cam_id,
                            "lens_calib_file": os.path.basename(self.calib_path),
                            "timestamp": time.time(),
                        }
                        self.engine.save(self.output_path, extra_meta=meta)
                        print(f"\n[SUKSES] Matriks homografi berhasil disimpan ke: {self.output_path}")
                    else:
                        print("\n[ERROR] Matriks belum dihitung! Tekan 'C' terlebih dahulu.")

                elif key == ord("v"):  # Toggle mode verifikasi
                    if self.engine.H is not None:
                        self.is_verified_mode = not self.is_verified_mode
                        status = "AKTIF" if self.is_verified_mode else "NONAKTIF"
                        print(f"\n[TOGGLE] Mode Verifikasi Klik: {status}")
                    else:
                        print("\n[ERROR] Matriks homografi belum dihitung!")

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Program kalibrasi homografi ditutup.")

    def _compute_and_preview(self, current_frame: np.ndarray):
        """Menghitung matriks dan menampilkan pratinjau Bird's-Eye View."""
        if len(self.points_img) < 4:
            print(f"\n[PERINGATAN] Dibutuhkan minimal 4 titik (saat ini: {len(self.points_img)} titik).")
            return

        try:
            H, err, mask = self.engine.compute()
            print("\n" + "=" * 60)
            print(" HASIL KALIBRASI HOMOGRAFI:")
            print("=" * 60)
            print(f" Matriks H:\n{H}")
            print(f" Mean Reprojection Error: {err:.2f} cm")
            status_err = "SANGAT BAIK (< 2.0 cm)" if err <= 2.0 else "CUKUP (< 5.0 cm)" if err <= 5.0 else "BURUK (> 5.0 cm)"
            print(f" Kualitas Kalibrasi      : {status_err}")
            print("=" * 60)

            # Tampilkan Pratinjau Bird's-Eye View
            top_down = self.engine.generate_top_down_view(
                current_frame, view_size_px=(600, 600), max_world_cm=(200.0, 200.0)
            )
            # Gambar grid koordinat referensi pada top-down view
            for grid_val in range(50, 200, 50):
                px = int(grid_val * (600 / 200))
                cv2.line(top_down, (px, 0), (px, 600), (80, 80, 80), 1)
                cv2.line(top_down, (0, px), (600, px), (80, 80, 80), 1)
                cv2.putText(top_down, f"{grid_val}cm", (px + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(top_down, f"{grid_val}cm", (5, px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Pratinjau Bird's-Eye View (Top-Down Lantai)", top_down)
            print("[INFO] Pratinjau Bird's-Eye View ditampilkan di jendela terpisah.")
            print("[INFO] Tekan 'S' untuk menyimpan matriks ke file .npz.")

        except Exception as e:
            print(f"[ERROR] Gagal menghitung homografi: {e}")

    def _draw_overlay(self, frame: np.ndarray):
        """Menggambar visualisasi titik-titik marker dan banner panduan."""
        h, w = frame.shape[:2]

        # 1. Gambar garis polygon penghubung titik marker
        if len(self.points_img) >= 2:
            pts = np.array(self.points_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=(len(self.points_img) >= 4), color=(0, 255, 255), thickness=2)

        # 2. Gambar titik marker dan label koordinatnya
        for idx, ((u, v), (wx, wy)) in enumerate(zip(self.points_img, self.points_world)):
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
            cv2.circle(frame, (u, v), 10, (255, 255, 255), 2)
            label = f"P{idx+1}: ({wx:.0f}, {wy:.0f})cm"
            cv2.putText(frame, label, (u + 12, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 3. Tampilan hasil verifikasi klik mouse jika aktif
        if self.last_clicked_pos and self.last_predicted_world:
            cx, cy = self.last_clicked_pos
            px_w, py_w = self.last_predicted_world
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
            v_label = f"KLIK TERUKUR: X = {px_w:.1f} cm, Y = {py_w:.1f} cm"
            cv2.rectangle(frame, (cx + 10, cy - 30), (cx + 340, cy + 5), (20, 20, 20), -1)
            cv2.putText(frame, v_label, (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 4. Banner Atas
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        mode_str = "VERIFIKASI KLIK UKUR" if self.is_verified_mode else "PENANDAAN TITIK"
        freeze_str = "FROZEN (TERKUNCI)" if self.is_frozen else "LIVE STREAM"
        banner_text = (
            f"Kamera {self.cam_id} | Mode: [{mode_str}] | Feed: [{freeze_str}] | "
            f"Titik Terdaftar: {len(self.points_img)} | Error: {self.engine.reprojection_error_cm:.2f} cm"
        )
        cv2.putText(frame, banner_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 5. Banner Bawah Panduan Tombol
        cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
        help_text = (
            "[KLIK]: Pilih Titik | [SPASI]: Freeze/Unfreeze | [C]: Hitung Homografi | "
            "[S]: Simpan | [V]: Toggle Verifikasi | [Z]: Undo | [R]: Reset | [Q]: Keluar"
        )
        cv2.putText(frame, help_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    def _print_instructions(self):
        print("\n" + "=" * 70)
        print(f" ALAT KALIBRASI HOMOGRAFI INTERAKTIF - KAMERA {self.cam_id}")
        print("=" * 70)
        print(" Panduan Penggunaan:")
        print("  1. Arahkan kamera ke area bidang kerja lantai 2x2 meter.")
        print("  2. Tekan [SPASI] untuk membekukan citra (Freeze).")
        print("  3. KLIK MOUSE KIRI pada titik-titik marker/sudut yang diketahui koordinatnya.")
        print("     (Pilih preset 1-5 atau ketik koordinat kustom cm di terminal).")
        print("  4. Setelah minimal 4 titik terdaftar, tekan [C] untuk menghitung matriks Homografi.")
        print("     (Jendela pratinjau Top-Down Bird's-Eye View akan muncul).")
        print("  5. Tekan [S] untuk menyimpan ke berkas .npz.")
        print("  6. Tekan [V] untuk menguji langsung: klik sembarang titik di lantai untuk melihat cm!")
        print("  7. Tekan [Z] untuk Undo titik terakhir, [R] untuk Reset, atau [Q] untuk Keluar.")
        print("=" * 70 + "\n")


def main():
    args = parse_arguments()
    app = HomographyGuiApp(args)
    app.run()


if __name__ == "__main__":
    main()
