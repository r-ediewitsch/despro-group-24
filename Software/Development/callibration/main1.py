import glob
import os
import cv2
import numpy as np

# ==========================================
# 1. KONFIGURASI PARAMETER
# ==========================================
CHECKERBOARD = (12, 9)  # Sudut internal (kolom, baris)
SQUARE_SIZE_MM = 19.0  # Ukuran kotak fisik (mm)
IMAGE_DIR = "dataset_cam1/*.jpg"  # Folder citra input
OUTPUT_FILE = "calib_cam1.npz"  # File penyimpanan hasil
DEBUG_DIR = "debug_cam1"  # Folder penyimpanan hasil debug visual

# Pengaturan tampilan
SHOW_WINDOW = True  # True: tampilkan jendela pop-up; False: hanya proses di balik layar
WINDOW_DELAY_MS = 500  # Durasi tampil per gambar (ms). Set ke 0 jika ingin tekan tombol untuk lanjut.

os.makedirs(DEBUG_DIR, exist_ok=True)

# Kriteria terminasi sub-piksel: 30 iterasi atau toleransi 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Menyiapkan koordinat 3D fisik titik acuan dunia (Z = 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # Titik 3D dunia nyata
imgpoints = []  # Titik 2D bidang piksel
valid_images = []  # Menyimpan nama file gambar yang berhasil diproses

# ==========================================
# 2. EKSTRAKSI SUDUT DARI DATASET CITRA
# ==========================================
images = sorted(glob.glob(IMAGE_DIR))
gray_shape = None

if not images:
    raise FileNotFoundError(
        f"Tidak ada file citra yang ditemukan pada direktori: {IMAGE_DIR}"
    )

print(f"Ditemukan {len(images)} gambar. Memulai deteksi sudut...\n")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"[GAGAL BUKA] File rusak atau tidak terbaca: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Cari sudut internal papan catur
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners_refined)
        valid_images.append(fname)

        # Gambar titik sudut berwarna pada citra
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)

        # Simpan citra visualisasi ke folder debug
        debug_path = os.path.join(DEBUG_DIR, os.path.basename(fname))
        cv2.imwrite(debug_path, img)

        print(f"[BERHASIL] {fname} -> Tersimpan di {debug_path}")

        # Tampilkan di jendela layar
        if SHOW_WINDOW:
            cv2.putText(
                img,
                f"{os.path.basename(fname)} (Tekan 'q' untuk keluar)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Debug Sudut Papan Catur", img)
            key = cv2.waitKey(WINDOW_DELAY_MS) & 0xFF
            if key == ord("q") or key == 27:  # 'q' atau Esc untuk menutup jendela
                SHOW_WINDOW = False
    else:
        print(f"[GAGAL DETEKSI] Sudut tidak terbaca pada: {fname}")

if SHOW_WINDOW:
    cv2.destroyAllWindows()

if len(objpoints) < 3:
    raise ValueError(
        "Jumlah gambar valid kurang dari 3. Kalibrasi Zhang membutuhkan minimal 3 sudut pandang berbeda."
    )

# ==========================================
# 3. KALKULASI PARAMETER KALIBRASI ZHANG
# ==========================================
print(
    f"\nMenghitung kalibrasi berdasarkan {len(objpoints)} gambar valid..."
)
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

# ==========================================
# 4. EVALUASI REPROJECTION ERROR PER CITRA
# ==========================================
print("\n--- Evaluasi Galat per Gambar ---")
total_error = 0

for i in range(len(objpoints)):
    projected_points, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )

    # Menyamakan bentuk kedua array menjadi (N, 2) bertipe float32
    pts_true = imgpoints[i].reshape(-1, 2).astype(np.float32)
    pts_proj = projected_points.reshape(-1, 2).astype(np.float32)

    # Perhitungan L2 norm yang identik secara matematis dan aman dari konflik channel OpenCV
    error = np.linalg.norm(pts_true - pts_proj) / len(pts_proj)
    total_error += error

    # Beri tanda jika ada foto individual yang galatnya melebihi 1 piksel
    status_error = "OK" if error < 1.0 else "BURUK (>1.0 px)"
    print(
        f"{os.path.basename(valid_images[i]):<25} : {error:.4f} px [{status_error}]"
    )

mean_error = total_error / len(objpoints)
print(f"\nMean Reprojection Error Total : {mean_error:.4f} piksel")

if mean_error < 1.0:
    print("Status: Kalibrasi memenuhi standar akurasi (< 1.0 piksel).")
else:
    print(
        "Peringatan: Reprojection error terlalu tinggi. Pertimbangkan menghapus file bertanda [BURUK] lalu jalankan ulang."
    )

# ==========================================
# 5. SIMPAN PARAMETER KE FILE
# ==========================================
np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    mean_error=mean_error,
)
print(f"\nParameter berhasil disimpan ke {OUTPUT_FILE}")