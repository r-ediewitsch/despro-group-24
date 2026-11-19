# Subsistem Kontrol Deezpro (Python Coordinator <-> 3x ESP32)

Folder ini berisi implementasi kode modular untuk **Tahap Kontrol Perangkat Keras dan Integrasi** sistem pemadam api otomatis Deezpro, mengacu pada arsitektur [`ARCH_PLAN.md`](file:///d:/Tugas/Desain%20Proyek/Deezpro/ARCH_PLAN.md) dan perancangan [`control_plan.md`](file:///d:/Tugas/Desain%20Proyek/Deezpro/Software/Development/esp32-src/control_plan.md).

---

## 1. Struktur Direktori `Software/Development-2/`

```text
Software/Development-2/
├── config/
│   └── system_config.yaml          # Konfigurasi frame (2x2m), clearance, resolusi step P & Q, dan MQTT
│
├── firmware/                       # KODE SOURCE MIKROKONTROLER (ESP32)
│   ├── common/
│   │   └── deezpro_protocol.h      # Definisi topik MQTT dan string status bersama
│   ├── esp32_stepper_x/
│   │   └── esp32_stepper_x.ino     # Firmware ESP #1: Motor Sumbu X + Limit Switch X (Pin 27)
│   ├── esp32_stepper_y/
│   │   └── esp32_stepper_y.ino     # Firmware ESP #2: Motor Sumbu Y + Limit Switch Y (Pin 27)
│   └── esp32_apar_servo/
│       └── esp32_apar_servo.ino    # Firmware ESP #3: Servo Penekan Tuas APAR (Pin 18)
│
├── src/
│   ├── integration/                # CORE BRAIN (SUBSISTEM KOORDINASI)
│   │   ├── kinematics.py           # Clamping clearance & konversi koordinat (cm) -> step
│   │   ├── mqtt_bridge.py          # Asynchronous MQTT Bridge + Synchronization Barrier
│   │   ├── coordinator.py          # Central Coordinator & State Machine (Maks 3x retry)
│   │   ├── safety_monitor.py       # Watchdog heartbeat & penanganan Emergency Stop
│   │   └── system_logger.py        # Logging standar ke konsol dan logs/system.log
│   │
│   └── vision/                     # SUBSISTEM VISION & GEOMETRI
│       ├── __init__.py             # Interface kontrak data Vision (FireTarget)
│       ├── detector.py             # Engine inferensi YOLO best.pt (ekstraksi koordinat piksel)
│       └── homography.py           # Engine transformasi & proyeksi homografi planar
│
├── models/
│   └── best.pt                     # Bobot model YOLO terlatih (22.5 MB, class 0: smoke, 1: fire)
│
├── logs/
│   ├── system.log                  # File log audit aktivitas sistem
│   └── runtime_state.json          # Snapshot posisi koordinat cm & step terkini
│
├── calibrate_homography.py         # GUI interaktif kalibrasi homografi & verifikasi klik
├── detect_fire_pixels.py           # Program visualisasi deteksi api & koordinat piksel
├── inspect_cameras.py              # Visual inspector pemetaan index webcam (Cam 1 & Cam 2)
├── collect_dataset.py              # Pengumpul citra lapangan (interval sampling + undistort)
├── prelabel_dataset.py             # Batch zero-shot pre-labeling (Grounding DINO / Florence-2)
├── annotate_dataset.py             # GUI Human-in-the-Loop verifikasi & export dataset YOLO
├── run_controller_only.py          # Konsol interaktif pengujian kontrol tanpa kamera
├── test_kinematics.py              # Unit testing untuk modul kinematics & clamping
├── test_homography.py              # Unit testing untuk modul homografi & proyeksi
├── test_detector.py                # Unit testing untuk modul deteksi api & asap
├── requirements.txt                # Daftar pustaka Python yang dibutuhkan
├── MANUAL_DATASET.md               # PANDUAN LENGKAP PENGUMPULAN & ANOTASI DATASET
└── README.md                       # Dokumentasi operasional modul kontrol ini
```

> [!TIP]
> **Panduan Pengumpulan & Anotasi Dataset Terperinci**:
> Untuk panduan langkah demi langkah pengumpulan data lapangan dari kamera, silakan baca [**`MANUAL_DATASET.md`**](file:///d:/Tugas/Desain%20Proyek/Deezpro/Software/Development-2/MANUAL_DATASET.md).


---

## 2. Panduan Menyiapkan Environment Python (Langkah demi Langkah)

Ikuti langkah-langkah di bawah ini untuk berpindah ke direktori proyek, membuat *virtual environment*, dan menginstal dependensi:

### Langkah 1: Buka Terminal (PowerShell / Command Prompt) dan Pindah Direktori
```powershell
cd "D:\Tugas\Desain Proyek\Deezpro\Software\Development-2"
```

### Langkah 2: Buat Virtual Environment Terisolasi
Buat lingkungan virtual Python baru bernama `venv`:
```powershell
python -m venv venv
```

### Langkah 3: Aktifkan Virtual Environment
* **Pada PowerShell (Windows):**
  ```powershell
  # Jika muncul pembatasan ExecutionPolicy, izinkan proses lokal:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

  # Aktifkan virtual environment:
  .\venv\Scripts\Activate.ps1
  ```
* **Pada Command Prompt (CMD biasa):**
  ```cmd
  venv\Scripts\activate.bat
  ```

*(Setelah aktif, nama terminal akan diawali tanda `(venv)`)*

### Langkah 4: Perbarui PIP dan Pasang Seluruh Dependensi
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Langkah 5: Verifikasi Pengujian Unit (Unit Test)
Pastikan seluruh pustaka terpasang dengan benar dengan menjalankan pengujian unit otomatis:
```powershell
python -m unittest discover -s . -p "test_*.py"
```
*(Seluruh 11 pengujian unit kinematika dan homografi harus berstatus **OK**)*

---

## 3. Cara Kerja Sistem Kontrol

1. **Translasi Koordinat ke Langkah Motor (Kinematics)**:
   - Frame fisik berukuran $M \times N$ ($200 \times 200$ cm) dengan clearance $m$ dan $n$ (masing-masing 10 cm).
   - Jangkauan efektif: $180 \times 180$ cm.
   - Jika koordinat api jatuh pada area clearance ($X < 10$ atau $Y < 10$), fungsi **Clamping** secara otomatis menyesuaikan nozzle ke titik terdekat.
   - Koordinat diubah ke target langkah motor $(\text{Step}_X, \text{Step}_Y)$ berbasis parameter total langkah $P$ dan $Q$ (default: 8000 langkah).
2. **Pergerakan Simultan & Synchronization Barrier**:
   - Python Coordinator menerbitkan perintah gerak ke ESP 1 dan ESP 2 secara bersamaan melalui MQTT.
   - Coordinator menahan eksekusi (*barrier lock*) sampai **kedua ESP** mengirimkan status `DONE`.
3. **Pemicu APAR & Batas Percobaan Semprot (Max 3 Retries)**:
   - Setelah kedua sumbu selesai bergerak, ESP 3 dipicu untuk menekan tuas APAR selama durasi tertentu (default: 3000 ms).
   - Sistem memverifikasi kondisi api: jika masih menyala, dilakukan penyemprotan ulang maksimal hingga 3 kali.
   - Jika melebihi 3 kali percobaan dan api belum padam, sistem masuk ke status **`FAULT`** untuk menghentikan pemborosan isi tabung APAR dan membunyikan alarm darurat operator.

---

## 4. Cara Menguji Menggunakan `run_controller_only.py`

Skrip ini memungkinkan Anda menguji komunikasi MQTT, pergerakan motor stepper, dan aktuasi servo APAR secara mandiri:

### Menjalankan Program Konsol:
```powershell
python "run_controller_only.py"
```

### Daftar Perintah Interaktif:
* `FIRE <x_cm> <y_cm>`: Menyimulasikan siklus penuh pemadaman api (Clamping $\rightarrow$ Gerak Simultan $\rightarrow$ Barrier $\rightarrow$ Semprot APAR $\rightarrow$ Selesai).
  * *Contoh*: `FIRE 100 120` (titik tengah), `FIRE 5 150` (akan otomatis di-clamp ke $X = 10$).
* `GOTO <x_cm> <y_cm>`: Menggerakkan nozzle ke koordinat fisik tanpa menyemprot APAR.
* `HOME`: Memerintahkan Auto-Homing pada kedua sumbu menuju limit switch.
* `JOG X <steps>` / `JOG Y <steps>`: Menggerakkan motor relatif sejumlah langkah (misal: `JOG X 500`).
* `SPRAY [duration_ms]`: Menguji penekanan tuas servo APAR (misal: `SPRAY 2000`).
* `STATUS`: Menampilkan telemetri real-time: posisi koordinat, step, status koneksi 3x ESP32, dan state sistem.
* `ESTOP`: Penghentian darurat instan seluruh motor dan pelepasan tuas APAR.
* `RESET`: Mereset status sistem dari `FAULT` / `EMERGENCY_STOP` kembali ke `IDLE_MONITORING`.

---

## 5. Pengujian Unit (*Unit Testing*)

Jalankan seluruh pengujian unit logika kinematika dan homografi:
```powershell
python -m unittest discover -s . -p "test_*.py"
```
*(Seluruh 11 skenario pengujian diverifikasi lulus 100%)*

---

## 6. Wiring Hardware ESP32

### ESP32 #1 (Sumbu X) & ESP32 #2 (Sumbu Y):
* `STEP_PIN`: GPIO 14
* `DIR_PIN`: GPIO 12
* `ENABLE_PIN`: GPIO 21 (Active LOW)
* `LIMIT_PIN`: GPIO 27 (Input Pull-up internal, hubungkan ke Ground saat tertekan)

### ESP32 #3 (APAR Servo):
* `SERVO_PIN`: GPIO 18 (Sinyal PWM Servo)
* `VCC`: 5V eksternal (pastikan ground bersama / common ground dengan ESP32)

---

## 7. Diagnostik & Verifikasi Index Kamera (`inspect_cameras.py`)

Jika Anda menggunakan laptop dengan webcam internal dan menghubungkan 2 webcam USB eksternal, gunakan tool diagnostik visual ini untuk memastikan nomor index OpenCV (`0, 1, 2...`) terhubung ke Kamera 1 dan Kamera 2 yang tepat:

```powershell
python "inspect_cameras.py"
```
* Layar akan menampilkan seluruh live stream webcam yang aktif secara berdampingan.
* **Cara Cepat Menemukan Fisik Kamera**: Tutup salah satu lensa webcam USB dengan tangan Anda $\rightarrow$ amati jendela mana di layar yang menjadi gelap.
* **Tekan `[TAB]`**: Berpindah fokus antar kamera.
* **Tekan `[1]`**: Menetapkan webcam terpilih sebagai **Kamera 1** (menggunakan `calib_cam1.npz`).
* **Tekan `[2]`**: Menetapkan webcam terpilih sebagai **Kamera 2** (menggunakan `calib_cam2.npz`).
* **Tekan `[S]`**: Langsung menyimpan konfigurasi ke `config/system_config.yaml`. Seluruh skrip (`calibrate_homography.py`, `detect_fire_pixels.py`, dll.) akan otomatis menggunakan index tersebut!

---

## 8. Cara Melakukan Kalibrasi Homografi Kamera

Program interaktif [**`calibrate_homography.py`**](calibrate_homography.py) digunakan untuk memetakan koordinat piksel kamera $(u, v)$ ke koordinat fisik lantai $(X_w, Y_w)$ dalam satuan centimeter.

### Menjalankan Kalibrasi untuk Kamera 1:
```powershell
python "calibrate_homography.py" --camera 1
```

### Menjalankan Kalibrasi untuk Kamera 2:
```powershell
python "calibrate_homography.py" --camera 2
```

### Langkah di Jendela Kalibrasi:
1. Tekan **`SPASI`** untuk membekukan (*Freeze*) frame saat marker di lantai terlihat jelas.
2. **Klik Mouse Kiri** pada titik-titik sudut atau marker di lantai (pilih preset 1-5 atau ketik koordinat kustom di terminal).
3. Setelah minimal 4 titik terdaftar, tekan **`C`** untuk menghitung matriks Homografi dan melihat pratinjau **Bird's-Eye View (Top-Down)**.
4. Tekan **`S`** untuk menyimpan matriks ke `config/homography_cam1.npz` (atau `cam2.npz`).
5. Tekan **`V`** untuk mengaktifkan **Mode Verifikasi**: klik sembarang titik di lantai untuk melihat koordinat $X$ dan $Y$ (cm) secara langsung!

---

## 8. Cara Menjalankan Deteksi Api & Asap (Koordinat Piksel)

Program [**`detect_fire_pixels.py`**](detect_fire_pixels.py) menjalankan model `models/best.pt` secara *real-time* untuk mendeteksi bahaya api dan asap tanpa melakukan konversi koordinat dunia (murni koordinat piksel).

### Menjalankan Program Deteksi Live Webcam:
```powershell
python "detect_fire_pixels.py" --camera 1
```

### Menjalankan pada Citra Statis (Foto):
```powershell
python "detect_fire_pixels.py" --image "path/to/foto_api.jpg"
```

### Fitur & Kontrol Keyboard:
* **Titik Acuan Kontak Lantai (*Base-Center*) vs Centroid**: Tekan tombol **`P`** untuk beralih antara titik tengah dasar lidah api $(u_b, v_b)$ dan titik centroid $(u_c, v_c)$.
* **Koreksi Lensa (*Undistort*)**: Tekan tombol **`U`** untuk mengaktifkan/menonaktifkan proses undistort `calib_cam1.npz`.
* **Screenshot**: Tekan tombol **`S`** untuk menyimpan tangkapan layar lengkap dengan anotasi bounding box dan koordinat piksel.
* **Banner HUD Status Bahaya**: Menampilkan status peringatan `[!] API TERDETEKSI`, `[!] ASAP TERDETEKSI`, atau `[OK] MONITORING AMAN`.

---

## 9. Pengumpulan & Anotasi Dataset (Sesuai STEPS.md)

Modul ini menyediakan alur kerja lengkap untuk akuisisi citra lapangan dan pelabelan data sesuai standar **`STEPS.md`**:

### A. Pengumpulan Citra Lapangan (`collect_dataset.py`)
Merekam citra dari webcam/kamera statis dengan interval sampling anti-redundansi (1.5 detik per frame) dan koreksi distorsi lensa (*undistort*):
```powershell
python "collect_dataset.py" --source 0 --calib "config/calib_cam1.npz" --out "dataset_raw"
```
* **`[SPACE]`**: Ambil frame manual saat nyala api berfluktuasi atau asap menyebar.
* **`[M]`**: Ganti mode antara `POSITIVE` (Api/Asap) dan `NEGATIVE` (Hard Negative/Background).
* **`[A]`**: Aktifkan/nonaktifkan auto-interval capture.

### B. Zero-Shot Batch Pre-Labeling (`prelabel_dataset.py`)
Gunakan foundation model seperti **Grounding DINO** atau **Microsoft Florence-2** untuk mendeteksi api (*fire/flame*) dan asap (*smoke*) secara semantik tanpa bias overfitting model lama:
```powershell
# Menggunakan Grounding DINO (rekomendasi kecepatan & akurasi)
python "prelabel_dataset.py" --backend grounding_dino --box-thresh 0.30

# Atau menggunakan Microsoft Florence-2
python "prelabel_dataset.py" --backend florence2
```
* Skrip akan memproses seluruh folder `dataset_raw/` dan langsung membuat file label `.txt` (format YOLO) di `dataset_annotated/labels/`.
* Gambar pada subfolder `negative/` otomatis diberi label file 0-byte (background murni).

### C. Verifikasi Human-in-the-Loop & Pembagian Dataset (`annotate_dataset.py`)
Membuka antarmuka grafis interaktif untuk memeriksa draf label, menghapus false positive, dan mengekspor dataset:
```powershell
python "annotate_dataset.py" --raw "dataset_raw" --out "dataset_annotated" --backend grounding_dino
```
* **Drag Mouse Kiri**: Gambar bounding box baru.
* **Klik Kanan pada Box**: Hapus deteksi palsu (*False Positive* / distractor).
* **`[0]` / `[1]`**: Pilih kelas aktif `0: smoke` atau `1: fire`.
* **`[B]`**: Tandai citra sebagai background murni (menghasilkan file `.txt` kosong 0-byte).
* **`[SPACE]`**: Simpan anotasi YOLO dan lanjut ke gambar berikutnya.
* **`[S]`**: Ekspor dan split otomatis ke folder `dataset_yolo/` (80% train / 20% val) lengkap dengan `data.yaml`!


