"""
inspect_cameras.py
------------------
Tool diagnostik & verifikasi index kamera (OpenCV VideoCapture):
- Memindai seluruh webcam yang terhubung ke komputer/laptop (index 0, 1, 2, dst).
- Menampilkan live feed seluruh kamera secara berdampingan (side-by-side) dalam satu jendela.
- Membantu Anda mengidentifikasi kamera fisik (tutup lensa salah satu kamera dengan tangan).
- Menetapkan Kamera 1 dan Kamera 2 secara interaktif:
    * Tekan '1' lalu pilih index untuk Cam 1
    * Tekan '2' lalu pilih index untuk Cam 2
    * Tekan 'S' untuk langsung menyimpan konfigurasi ke config/system_config.yaml!
"""

import os
import sys
import yaml
import numpy as np
import cv2


def scan_available_cameras(max_tested=5):
    """Memindai index kamera yang valid dan dapat dibuka."""
    available = []
    print(f"[Scanner] Memindai perangkat kamera (index 0 sampai {max_tested-1})...")

    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append({"index": i, "cap": cap, "width": w, "height": h})
                print(f"  [+] Kamera Ditemukan di Index {i}: Resolusi default {w}x{h}")
            else:
                cap.release()
        else:
            cap.release()
    return available


def update_system_config(cam1_idx, cam2_idx, config_path="config/system_config.yaml"):
    """Menyimpan pemetaan index kamera ke berkas konfigurasi YAML."""
    if not os.path.exists(config_path):
        print(f"[ERROR] File konfigurasi tidak ditemukan: {config_path}")
        return False

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "cameras" not in cfg:
        cfg["cameras"] = {}

    cfg["cameras"]["cam1"] = {
        "index": int(cam1_idx),
        "name": "Kamera 1",
        "calib_file": "config/calib_cam1.npz",
        "homography_file": "config/homography_cam1.npz",
        "resolution": [1280, 720]
    }
    cfg["cameras"]["cam2"] = {
        "index": int(cam2_idx),
        "name": "Kamera 2",
        "calib_file": "config/calib_cam2.npz",
        "homography_file": "config/homography_cam2.npz",
        "resolution": [1280, 720]
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n[SUKSES] Konfigurasi kamera berhasil disimpan ke {config_path}:")
    print(f"  - Kamera 1 (Zhang: calib_cam1.npz) -> Menggunakan Index {cam1_idx}")
    print(f"  - Kamera 2 (Zhang: calib_cam2.npz) -> Menggunakan Index {cam2_idx}")
    return True


def run_camera_inspector(config_path="config/system_config.yaml"):
    cams = scan_available_cameras(max_tested=6)

    if not cams:
        print("\n[ERROR] Tidak ada kamera yang terdeteksi! Pastikan kabel USB webcam terpasang.")
        return

    # Baca mapping saat ini dari config jika ada
    cam1_idx = 1
    cam2_idx = 2
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                if "cameras" in cfg:
                    cam1_idx = cfg["cameras"].get("cam1", {}).get("index", 1)
                    cam2_idx = cfg["cameras"].get("cam2", {}).get("index", 2)
        except Exception:
            pass

    win_name = "Deezpro - Pemeriksa Index Kamera (Visual Inspector)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 500)

    print("\n" + "=" * 65)
    print("PEMERIKSA KAMERA AKTIF")
    print("Tips Identifikasi:")
    print("  Tutup lensa salah satu webcam USB Anda dengan tangan.")
    print("  Perhatikan di layar, index mana yang menjadi gelap!")
    print("\nKontrol Pemetaan:")
    print("  [1] Tetapkan kamera yang sedang dipilih sebagai Kamera 1")
    print("  [2] Tetapkan kamera yang sedang dipilih sebagai Kamera 2")
    print("  [S] Simpan pemetaan index ke config/system_config.yaml")
    print("  [Q / ESC] Keluar")
    print("=" * 65 + "\n")

    selected_cam_slot = 0
    thumb_w, thumb_h = 480, 320

    while True:
        frames = []
        for i, item in enumerate(cams):
            cap = item["cap"]
            idx = item["index"]
            ret, frame = cap.read()
            if not ret or frame is None:
                # Placeholder frame jika gagal baca
                disp = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                cv2.putText(disp, f"CAM INDEX {idx} (NO FEED)", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Putar 180 derajat karena kamera terpasang terbalik pada prototype
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                disp = cv2.resize(frame, (thumb_w, thumb_h))

            # Beri label status pada masing-masing feed
            is_cam1 = (idx == cam1_idx)
            is_cam2 = (idx == cam2_idx)
            is_active_selection = (i == selected_cam_slot)

            # Border penanda
            border_color = (0, 255, 255) if is_active_selection else (60, 60, 60)
            if is_cam1:
                border_color = (0, 255, 0)   # Hijau = Cam 1
            elif is_cam2:
                border_color = (255, 120, 0) # Biru terang = Cam 2

            cv2.rectangle(disp, (0, 0), (thumb_w - 1, thumb_h - 1), border_color, 4 if is_active_selection else 2)

            # Banner info atas
            cv2.rectangle(disp, (0, 0), (thumb_w, 45), (20, 20, 20), -1)
            title = f"INDEX {idx}"
            if idx == 0:
                title += " (Laptop/Internal?)"

            cv2.putText(disp, title, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

            # Banner info bawah (Penugasan)
            cv2.rectangle(disp, (0, thumb_h - 40), (thumb_w, thumb_h), (20, 20, 20), -1)
            if is_cam1:
                role_text = ">> KAMERA 1 (calib_cam1.npz) <<"
                role_color = (0, 255, 0)
            elif is_cam2:
                role_text = ">> KAMERA 2 (calib_cam2.npz) <<"
                role_color = (255, 180, 0)
            else:
                role_text = "[TIDAK DIGUNAKAN / BUILT-IN]"
                role_color = (150, 150, 150)

            cv2.putText(disp, role_text, (10, thumb_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, role_color, 2)
            frames.append(disp)

        # Gabungkan secara horizontal
        grid = np.hstack(frames)

        # Tambahkan instruksi di bagian paling bawah
        hud = np.zeros((60, grid.shape[1], 3), dtype=np.uint8)
        cur_idx = cams[selected_cam_slot]["index"]
        instruction = (
            f"Kamera Dipilih: INDEX {cur_idx} | [TAB] Ganti Kamera | "
            f"[1] Set CAM 1 | [2] Set CAM 2 | [S] SIMPAN CONFIG | [Q] Keluar"
        )
        cv2.putText(hud, instruction, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        final_view = np.vstack([grid, hud])

        cv2.imshow(win_name, final_view)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 9:  # TAB -> Ganti kamera yang dipilih
            selected_cam_slot = (selected_cam_slot + 1) % len(cams)
            print(f"[Inspector] Fokus pada slot index: {cams[selected_cam_slot]['index']}")
        elif key == ord('1'):
            cam1_idx = cams[selected_cam_slot]["index"]
            print(f"[Inspector] Index {cam1_idx} Ditetapkan sebagai KAMERA 1")
        elif key == ord('2'):
            cam2_idx = cams[selected_cam_slot]["index"]
            print(f"[Inspector] Index {cam2_idx} Ditetapkan sebagai KAMERA 2")
        elif key in (ord('s'), ord('S')):
            update_system_config(cam1_idx, cam2_idx, config_path)

    # Release seluruh kamera
    for item in cams:
        item["cap"].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/system_config.yaml"
    run_camera_inspector(config_file)
