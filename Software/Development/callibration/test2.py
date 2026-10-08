import cv2
import numpy as np

# ==========================================
# 1. KONFIGURASI FILE & PARAMETER
# ==========================================
CALIB_FILE = "calib_cam1.npz"  # File hasil kalibrasi Zhang
TEST_IMAGE = "dataset_cam1/test.jpg"  # Path citra uji (atau pilih salah satu foto)
OUTPUT_RESULT = "hasil_undistort_perbandingan2.jpg"
DRAW_GRID = True  # True: tambahkan garis grid merah untuk uji kelurusan

# ==========================================
# 2. MUAT PARAMETER KALIBRASI
# ==========================================
try:
    calib_data = np.load(CALIB_FILE)
    K = calib_data["camera_matrix"]
    D = calib_data["dist_coeffs"]
    print("Parameter kalibrasi berhasil dimuat.")
    print(f"Matriks K:\n{K}")
    print(f"Koefisien Distorsi D:\n{D.ravel()}\n")
except Exception as e:
    raise FileNotFoundError(
        f"Gagal memuat file kalibrasi {CALIB_FILE}. Error: {e}"
    )

# ==========================================
# 3. BACA CITRA UJI
# ==========================================
img = cv2.imread(TEST_IMAGE)
if img is None:
    raise FileNotFoundError(
        f"Citra uji tidak ditemukan pada path: {TEST_IMAGE}"
    )

h, w = img.shape[:2]

# ==========================================
# 4. PROSES UNDISTORTION
# ==========================================
# Opsi 1: Undistort langsung (mempertahankan dimensi piksel asli)
undistorted_basic = cv2.undistort(img, K, D, None, K)

# Opsi 2: Menggunakan matriks kamera optimal (alpha=0 membuang tepi hitam)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    K, D, (w, h), alpha=0, newImgSize=(w, h)
)
undistorted_optimal = cv2.undistort(img, K, D, None, new_camera_matrix)

# ==========================================
# 5. VISUALISASI DENGAN GARIS BANTU (GRID)
# ==========================================
display_orig = img.copy()
display_undist = undistorted_basic.copy()

if DRAW_GRID:
    # Gambar garis panduan horizontal dan vertikal
    step_y = h // 8
    step_x = w // 8

    # Garis horizontal
    for y in range(step_y, h, step_y):
        cv2.line(display_orig, (0, y), (w, y), (0, 0, 255), 1)
        cv2.line(display_undist, (0, y), (w, y), (0, 0, 255), 1)

    # Garis vertikal
    for x in range(step_x, w, step_x):
        cv2.line(display_orig, (x, 0), (x, h), (0, 0, 255), 1)
        cv2.line(display_undist, (x, 0), (x, h), (0, 0, 255), 1)

# Tambahkan label teks
cv2.putText(
    display_orig,
    "CITRA ASLI (DISTORSI)",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
cv2.putText(
    display_undist,
    "HASIL UNDISTORT",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

# Gabungkan secara horizontal untuk perbandingan visual
comparison = np.hstack((display_orig, display_undist))

# Simpan dan tampilkan hasil
cv2.imwrite(OUTPUT_RESULT, comparison)
print(f"Hasil perbandingan disimpan ke: {OUTPUT_RESULT}")

# Tampilkan jendela perbandingan (tekan 'q' atau 'Esc' untuk menutup)
# Resize tampilan jika resolusi monitor tidak mencukupi
scale = 0.5 if (w * 2) > 1920 else 1.0
preview = (
    cv2.resize(comparison, (0, 0), fx=scale, fy=scale)
    if scale != 1.0
    else comparison
)

cv2.imshow("Pengujian Undistort (Kiri: Asli | Kanan: Terkoreksi)", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()