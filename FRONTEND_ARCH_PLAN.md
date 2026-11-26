# Blueprint Arsitektur Frontend JOSJIS (FRONTEND_ARCH_PLAN.md) — React + Vite

Dokumen ini merinci rencana arsitektur antarmuka pengguna (*Operator Web Dashboard*) untuk sistem pemadam api otomatis JOSJIS (sebelumnya Deezpro). Desain ini merupakan ekspansi dari **Milestone 5 pada [ARCH_PLAN.md](file:///d:/Tugas/Desain%20Proyek/Deezpro/ARCH_PLAN.md)**, menghubungkan sistem komputasi Python (OpenCV, YOLO, Homografi, MQTT Coordinator) dengan antarmuka web modern berbasis **React + Vite**.

---

## 1. Diagram Arsitektur Integrasi Web & Backend

Sistem dirancang dengan pola **Decoupled Client-Server**:
* **Backend (Python FastAPI)**: Mengelola hardware, kamera, AI, kinematika, dan broker MQTT.
* **Frontend (React + Vite)**: Menampilkan visualisasi dual-kamera, telemetri real-time, denah digital twin 2D, dan kontrol interaktif.

```mermaid
flowchart TD
    subgraph Hardware_Layer ["1. Perangkat Keras & Jaringan"]
        Cam1["Webcam 1 (USB)"] --> PyBackend
        Cam2["Webcam 2 (USB)"] --> PyBackend
        ESP["3x ESP32 (X, Y, APAR)"] <-->|MQTT JSON| Broker(("EMQX MQTT Broker"))
    end

    subgraph PyBackend ["2. Python Core & API Server (FastAPI)"]
        Broker <--> MQTT["mqtt_bridge.py"]
        MQTT <--> Coord["coordinator.py (State Machine)"]
        Cam1 & Cam2 --> Vision["vision_pipeline.py (YOLO + Homografi)"]
        Vision --> Coord
        
        Coord & Vision --> FastAPIServer["FastAPI Server (src/server/app.py)"]
        FastAPIServer -->|MJPEG Stream @ 30 FPS| VideoRoute["/api/video/cam1 & /cam2"]
        FastAPIServer <-->|WebSocket @ 10-20 Hz| WSRoute["/ws/telemetry"]
        FastAPIServer -->|REST API (POST / GET)| APIRoute["/api/control/*"]
        FastAPIServer -->|Static File Serving| StaticRoute["frontend/dist (Production Build)"]
    end

    subgraph ReactFrontend ["3. Operator Web Dashboard (React + Vite SPA)"]
        VideoRoute --> CamPanel["Dual Camera Viewfinder (Cam 1 & Cam 2)"]
        WSRoute <--> StateStore["Zustand / React Context Store"]
        APIRoute <--> CtrlPanel["Operator Action Controls (GOTO, SPRAY, ESTOP)"]
        
        StateStore --> GantryCanvas["2D Digital Twin Canvas (200x200 cm)"]
        StateStore --> TelemetryCard["Status & Telemetry Cards"]
        StateStore --> EventLog["Live Event Terminal Log"]
    end
```

---

## 2. Struktur Direktori Lengkap Proyek

Berikut adalah penataan struktur folder yang rapi dan terisolasi di dalam `Software/`:

```text
Software/
├── config/
│   ├── system_config.yaml              # Konfigurasi frame, clearance, step P&Q, MQTT
│   ├── calib_cam1.npz & calib_cam2.npz # Parameter intrinsik Zhang
│   └── homography_cam1.npz & cam2.npz  # Matriks proyeksi lantai
│
├── src/                                # BACKEND PYTHON CORE
│   ├── vision/                         # Subsistem Penglihatan Komputer
│   │   ├── camera_stream.py            # Threaded MJPG capture
│   │   ├── lens_corrector.py           # Koreksi distorsi lensa
│   │   ├── detector.py                 # Inferensi YOLO (Api/Asap)
│   │   ├── homography.py               # HomographyEngine (u, v -> X, Y cm)
│   │   └── fusion.py                   # Late Fusion & Fallback
│   │
│   ├── integration/                    # Subsistem Koordinator & Kontrol
│   │   ├── kinematics.py               # Clamping & translasi koordinat -> steps
│   │   ├── mqtt_bridge.py              # Handler komunikasi MQTT
│   │   ├── coordinator.py              # CentralCoordinator (State Machine)
│   │   ├── safety_monitor.py           # Watchdog & Emergency Stop
│   │   └── system_logger.py            # Pencatat log sistem
│   │
│   ├── server/                         # FASTAPI BACKEND SERVER (Jembatan Web)
│   │   ├── __init__.py
│   │   ├── app.py                      # Instance FastAPI & mount frontend/dist
│   │   ├── routes_video.py             # Endpoint streaming MJPEG Kamera 1 & 2
│   │   ├── routes_control.py           # Endpoint REST API (GOTO, SPRAY, ESTOP, RESET)
│   │   └── websocket_hub.py            # WebSocket manager pemancar telemetri
│   │
│   └── firmware/                       # Source code Arduino/C++ untuk 3x ESP32
│
├── frontend/                           # FRONTEND REACT + VITE (SPA)
│   ├── package.json                    # Dependensi NPM (React, Phosphor/Tabler Icons, Tailwind)
│   ├── vite.config.ts                  # Konfigurasi Vite & proxy backend
│   ├── tailwind.config.js              # Styling tema dark industrial
│   ├── tsconfig.json                   # Konfigurasi TypeScript
│   ├── index.html                      # Entry HTML
│   │
│   └── src/
│       ├── main.tsx                    # Entry script React
│       ├── App.tsx                     # Layout utama Dashboard
│       │
│       ├── components/                 # Komponen Antarmuka Reusable
│       │   ├── Header.tsx              # Navbar, State Badge, & Tombol ESTOP Instan
│       │   ├── DualCameraView.tsx       # Pemutar feed video Kamera 1 & Kamera 2
│       │   ├── GantryDigitalTwin.tsx   # Kanvas simulasi posisi fisik 2D (200x200 cm)
│       │   ├── TelemetryGrid.tsx       # Kartu telemetri langkah motor, cm, status APAR
│       │   ├── ControlConsole.tsx      # Form input koordinat manual, jog, & test spray
│       │   └── SystemLogViewer.tsx     # Terminal log kejadian sistem auto-scrolling
│       │
│       ├── hooks/
│       │   ├── useWebSocket.ts         # Hook koneksi WebSocket dengan auto-reconnect
│       │   └── useGantryControls.ts    # Hook pemanggilan REST API kontrol
│       │
│       └── types/
│           └── telemetry.ts            # Definisi tipe data TypeScript
│ 
├── logs/
│   └── system.log                      # File log terpusat
│
├── run_controller_only.py              # CLI test tool kontrol hardware
├── run_system.py                       # Eksekusi sistem mode headless (tanpa web)
├── run_web_dashboard.py                # SINGLE-COMMAND STARTER: Jalankan backend + buka browser
└── requirements.txt                    # Dependensi Python (FastAPI, uvicorn, paho-mqtt, dll)
```

---

## 3. Spesifikasi Kontrak Antarmuka (API & Data Contracts)

### 3.1 Streaming Video MJPEG
* **Endpoint**:
  * `GET /api/video/cam1` : Mengalirkan stream video Kamera 1 (dengan bounding box YOLO & overlay homografi).
  * `GET /api/video/cam2` : Mengalirkan stream video Kamera 2.
* **Implementasi di React**: Cukup gunakan elemen native HTML:
  ```tsx
  <img src="http://localhost:8000/api/video/cam1" className="w-full rounded-lg" alt="Kamera 1" />
  ```

### 3.2 WebSocket Telemetri (`/ws/telemetry`)
Backend menyiarkan (*broadcast*) paket data status sistem dengan frekuensi 10–20 Hz:
```json
{
  "timestamp": 1774328590.12,
  "system_state": "IDLE_MONITORING",
  "active_target_cm": {"x": 102.5, "y": 98.0},
  "is_fire_detected": true,
  "detection_mode": "DUAL_FUSION",
  "spatial_discrepancy_cm": 4.5,
  "current_position_cm": {"x": 25.5, "y": 40.5},
  "current_position_steps": {"x": 0, "y": 0},
  "nodes": {
    "x": {"state": "IDLE", "pos": 0},
    "y": {"state": "IDLE", "pos": 0},
    "apar": {"state": "READY"}
  },
  "latest_logs": [
    {"time": "11:50:01", "level": "INFO", "msg": "Target locked at (102.5, 98.0) cm"}
  ]
}
```

### 3.3 REST API Perintah Kontrol
* `POST /api/control/goto` : Menggerakkan nozzle ke target (`{"x_cm": 100.0, "y_cm": 100.0}`).
* `POST /api/control/spray` : Menguji semprot APAR (`{"duration_ms": 2000}`).
* `POST /api/control/estop` : Memicu **Emergency Stop instan** (menghentikan seluruh motor).
* `POST /api/control/reset` : Mereset status sistem dari `FAULT` / `EMERGENCY_STOP` ke `IDLE_MONITORING`.

---

## 4. Desain Tampilan UI Dashboard (React Layout)

Tata letak (*wireframe*) dirancang berorientasi pada kemudahan pengawasan operator:

```
+-----------------------------------------------------------------------------------------+
| [JOSJIS FIRE CONTROL]    Status: [TARGET_LOCKED]   MQTT: [ONLINE]   [ ! EMERGENCY STOP ! ]|
+-------------------------------------------------------------+---------------------------+
| [ PANEL DUAL VIDEO FEED (16:9) ]                            | [ 2D GANTRY DIGITAL TWIN ]|
| +----------------------------+----------------------------+ | Lapangan Fisik 200x200 cm |
| | Feed Kamera 1 (Kiri)       | Feed Kamera 2 (Kanan)      | | +-----------------------+ |
| | Target Base: (100, 98) cm  | Target Base: (104, 102) cm | | | Gantry Railing        | |
| +----------------------------+----------------------------+ | | (X, Y) Real-time      | |
|                                                             | |   [Nozzle]            | |
| [ TELEMETRI SISTEM ]                                        | |           * [Titik Api| |
| * Koordinat Terfusi : (102.0, 100.0) cm [KONSISTENSI TINGGI]| +-----------------------+ |
| * Posisi Motor X    : Step 3851 (~102 cm)                   +---------------------------+
| * Posisi Motor Y    : Step 3152 (~100 cm)                   | [ KONSOL KONTROL MANUAL ] |
| * Status APAR       : SIAGA (Tekanan Normal)                | [ AUTO MODE ] [ MANUAL ]  |
|                                                             | [ GOTO Target ]           |
| [ TERMINAL SYSTEM LOG ]                                     | [ Test Semprot 2s ]       |
| > [11:52:10] Koordinat fusi dihitung: (102.0, 100.0) cm     | [ Reset Status ]          |
| > [11:52:11] Mengirim perintah GOTO ke Sumbu X dan Y...     |                           |
+-------------------------------------------------------------+---------------------------+
```

---

## 5. Alur Deployment & Kemudahan Operasional

Salah satu keunggulan arsitektur ini adalah **operator tidak perlu menginstal Node.js saat sistem dijalankan di lapangan**:

1. **Fase Development**:
   * Frontend dikembangkan dengan `npm run dev` pada port 5173 dengan fitur *Hot Module Replacement (HMR)*.
2. **Fase Build**:
   * Jalankan `npm run build` di folder `frontend/`. Vite menghasilkan bundel HTML/JS/CSS statis di folder `frontend/dist`.
3. **Fase Produksi (Single Command)**:
   * FastAPI secara otomatis membaca berkas dari `frontend/dist`.
   * Operator hanya perlu menjalankan satu perintah Python:
     ```powershell
     python run_web_dashboard.py
     ```
   * Skrip akan otomatis menginisialisasi kamera, MQTT, web server, dan **langsung membuka browser** pada alamat `http://localhost:8000`. Dashboard juga bisa diakses melalui smartphone/tablet di jaringan Wi-Fi lokal yang sama.

---

## 6. Rencana Tahapan Eksekusi (Implementation Steps)

1. **Langkah 1 (Backend API Layer)**:
   * Buat dependensi FastAPI: `pip install fastapi uvicorn websockets`.
   * Implementasikan `src/server/app.py`, `routes_video.py` (generator MJPEG dari `camera_stream`), dan `routes_control.py`.
2. **Langkah 2 (Scaffolding React + Vite)**:
   * Buat proyek frontend: `npm create vite@latest frontend -- --template react-ts`.
   * Pasang Tailwind CSS & Icon sets terpilih (`@phosphor-icons/react`, `@tabler/icons-react`, custom SVG).
3. **Langkah 3 (Komponen Utama Dashboard)**:
   * Buat `DualCameraView.tsx` dan `Header.tsx` (termasuk tombol E-STOP).
   * Buat `GantryDigitalTwin.tsx` menggunakan kanvas SVG responsif.
   * Hubungkan WebSocket telemetri via `useWebSocket.ts`.
4. **Langkah 4 (Build & Packaging Terpadu)**:
   * Build frontend dan kaitkan ke `fastapi.staticfiles.StaticFiles`.
   * Buat `run_web_dashboard.py` untuk pengalaman *one-click startup*.

---

## 7. Aturan Desain Frontend JOSJIS (Design Rules & Anti-Pattern Guidelines)

Antarmuka web operator JOSJIS dirancang sebagai **Mission-Critical Industrial Cockpit / Tactical SCADA Dashboard**, bukan landing page produk SaaS konsumer. Desain mengutamakan densitas telemetri tinggi, latensi render instan, keterbacaan kontras tinggi di ruang operasional, serta tone visual profesional bernuansa biru industrial.

### 7.1 Palet Warna: Industrial Deep Blue & Tactical Cyan

Sistem warna dibangun di atas fondasi warna biru fungsional (bukan tema monokrom abu-abu biasa):

| Token Warna | Nilai HEX | Peruntukan & Karakter |
| :--- | :--- | :--- |
| **Canvas Background** | `#080D1A` | Dark Navy pekat sebagai dasar kanvas (memberikan kontras maksimal tanpa silau) |
| **Surface Panel (Cockpit)** | `#0F172A` / `#131E36` | Lapisan panel instrumen dengan tint biru malam pekat |
| **Subtle Border & Grid** | `#1E2E4A` / `#22385E` | Garis pembatas panel dan grid kanvas homografi 2D yang presisi |
| **Primary Accent (Cobalt)** | `#2563EB` / `#3B82F6` | Tombol aksi primer, track status motor aktif, dan sorotan telemetri |
| **Tactical Cyan / Reticle** | `#06B6D4` / `#38BDF8` | Crosshair homografi, vektor pelacakan api, dan indikator frekuensi data |
| **Status Armed / Ready** | `#10B981` | Indikasi hardware ESP32 siap, status MQTT online |
| **Status Warning / Delta** | `#F59E0B` | Peringatan disparitas spasial antar-kamera > 15 cm |
| **Emergency & Fire Alert** | `#EF4444` | Indikator api terkonfirmasi, tombol interupsi darurat (E-STOP) |

### 7.2 Tipografi Khusus Presisi Tinggi

* **Interface & Headings**: Menggunakan **Space Grotesk** atau **Plus Jakarta Sans**. Memberikan karakter industrial modern, geometris, dan ketegasan instrumen teknik.
* **Telemetri & Data Numerik**: Menggunakan **JetBrains Mono** atau **IBM Plex Mono** dengan fitur tabular numbers (`font-mono tabular-nums`). Wajib diterapkan pada:
  * Koordinat target dan posisi motor: `X: 102.5 cm | Y: 98.0 cm`
  * Langkah stepper: `Step X: 3851 | Step Y: 3152`
  * Latensi & Frekuensi: `WebSocket: 18 Hz | MJPEG: 30 FPS`
  * Log konsol dan status kode MQTT

### 7.3 Daftar 10 Anti-Pattern yang DILARANG & Solusi Desain JOSJIS

Berikut adalah aturan ketat untuk menghindari pola klise desain web modern yang tidak sesuai dengan instrumen industri:

| No | Anti-Pattern Generik yang Dihindari | Frekuensi Klise | Aturan & Solusi Khusus JOSJIS Dashboard |
| :---: | :--- | :---: | :--- |
| **1** | **Feature cards dengan ikon di atas dalam grid 3 atau 4** | 63% landing pages | **DILARANG.** Tidak menggunakan deretan kartu generik sejajar. Sebagai gantinya, gunakan **Asymmetrical Cockpit Telemetry Ribbons**, panel densitas tinggi (*density docks*), serta strip status horizontal fungsional yang menyatu langsung dengan viewport kamera dan kanvas gantry. |
| **2** | **Satu icon set seragam tanpa karakter lain (Hanya Lucide)** | 63% all pages | **DILARANG.** Jangan hanya mengimpor satu pustaka generik. Gunakan kurasi ikon gabungan: **Phosphor Icons** untuk kendali aktuator/sakelar, **Tabler Icons** untuk instrumen navigasi/sensor, serta **Custom SVG Technical Glyphs** untuk nozzle APAR, reticle homografi, dan denah rail gantry. |
| **3** | **Warna default komponen (shadcn) dengan hanya tombol utama diubah** | 61% all pages | **DILARANG.** Jangan gunakan palet abu-abu default (zinc/slate-950 standar). Semua komponen mengadopsi token warna bertingkat **Industrial Deep Navy** dengan border pendar halus (`#1E3A5F`) dan kontras fungsional SCADA. |
| **4** | **Tanpa font yang dipilih, atau memakai font default Inter** | 60% all pages | **DILARANG.** Hindari font default generic Inter. Terapkan hierarki ganda yang tegas: **Space Grotesk / Plus Jakarta Sans** untuk elemen navigasi/aksi dan **JetBrains Mono** untuk angka telemetri berakurasi tinggi. |
| **5** | **Satu kata dari headline diberi warna aksen** | 52% by eye | **DILARANG.** Jangan memakai efek gaya landing page di mana satu kata dalam judul diwarnai biru/gradasi. Gunakan tipografi monolitik yang kokoh. Pembeda visual hanya disematkan pada label telemetri operasional aktual (misal: state badge `[AUTO_TARGETING]`). |
| **6** | **Headline di tengah di atas segalanya (The centered headline)** | 51% all pages | **DILARANG.** Tidak ada judul rata tengah yang memboroskan ruang vertikal. Gunakan **Left-Aligned Cockpit HUD Bar** di mana identitas sistem berada di kiri atas, telemetri kesehatan sistem di tengah, dan kontrol darurat (E-STOP) di kanan atas. |
| **7** | **Pill badge atau label kecil mengambang di atas headline** | 51% by eye | **DILARANG.** Hilangkan elemen dekoratif kapsul mengambang tanpa fungsi. Sebagai gantinya, gunakan baris status telemetri yang menyatu (*integrated cockpit ticker*) dengan garis pembatas berkarakter instrumen. |
| **8** | **Setiap section fade-up saat di-scroll** | 51% all pages | **DILARANG.** Tidak boleh ada animasi scroll fade-in yang memperlambat akses visual operator. Seluruh dashboard adalah **Single-Viewport Mission Control (100vh)** tanpa scrolling. Transisi visual hanya terjadi saat data real-time berubah (*reactive state pulse*). |
| **9** | **Ikon "AI" klise: petir, bintang kilau (sparkles), atau otak** | 47% all pages | **DILARANG.** Jangan gunakan simbol petir, sparkles, atau otak. Gunakan glif visi komputer teknis: **Target Reticle (`Crosshair`)**, **Optical Lens (`Aperture`)**, **Planar Matrix Grid (`GridFour`)**, dan **Inference Engine (`Cpu`)**. |
| **10** | **Label huruf kapital kecil dengan letter-spacing renggang berlebih** | 42% all pages | **DILARANG.** Jangan gunakan pseudo-aesthetic tracking huruf kecil renggang yang melelahkan mata. Gunakan tipografi semantik yang bersih, ukuran proporsional (13–14px), dan teks monospace tegas untuk label status teknis. |

### 7.4 Prinsip Ergonomi Antarmuka Operator

1. **Safety First**: Tombol `EMERGENCY STOP` berukuran besar, kontras tinggi, dapat diakses instan dari seluruh tampilan tanpa terhalang dialog modal.
2. **Zero-Scroll Single Cockpit**: Semua informasi krusial (Kamera 1, Kamera 2, Digital Twin 200x200 cm, Telemetri Motor X/Y/APAR, dan Log Operasi) dapat dipantau sekaligus pada satu layar display operator (1080p / 1440p).
3. **High Signal-to-Noise Ratio**: Elemen dekoratif non-fungsional ditiadakan agar operator dapat fokus pada deteksi anomali, verifikasi api, dan kesiapan sistem aktuasi.
