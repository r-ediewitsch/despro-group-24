Secara konsep, apa yang kamu maksud secara teknis bukan **Reinforcement Learning (RL)** murni, melainkan **Semi-Supervised Learning (Self-Training / Pseudo-Labeling)** atau **Active Learning**.

Melakukan *feedback loop* otomatis penuh (model mendeteksi $\rightarrow$ hasil langsung dijadikan label train baru tanpa dicek) **sangat berisiko gagal** untuk kasusmu. Masalah utamanya disebut **Confirmation Bias** (atau *error accumulation*): jika model saat ini gagal mendeteksi api/asap di kamera murah (*false negative*) atau salah mendeteksi lampu/kabut sebagai api (*false positive*), memasukkan prediksi itu kembali hanya akan mengajari model untuk semakin keliru (*garbage in, garbage out*).

---

### Strategi Alternatif yang Lebih Efektif dan Cepat

Untuk mengatasi perbedaan domain (*domain shift*) antara dataset publik resolusi tinggi dan kamera murah tanpa harus menganotasi ribuan gambar dari nol, terapkan metode berikut:

#### 1. Auto-Labeling Menggunakan Foundation Model (Zero-Shot)

Daripada meminta YOLO melabeli dirinya sendiri, gunakan model deteksi visi berbasis teks (*open-vocabulary object detection*) yang jauh lebih kuat untuk menghasilkan anotasi otomatis pada rekaman kamera murahmu:

* Jalankan **Grounding DINO** atau **Florence-2** pada rekaman kamera murah dengan prompt `"fire, flame, smoke"`.
* Model-model ini memiliki representasi visual yang jauh lebih kaya dan mampu mengenali objek meskipun kualitas gambar buruk/buram.
* Simpan hasil prediksi bounding box mereka ke format YOLO (`.txt`), lalu gunakan langsung untuk *fine-tuning* model YOLO milikmu.

#### 2. Degradasi Dataset Publik (Augmentasi Adaptasi Domain)

Penyebab utama YOLO gagal di kamera murah adalah karena dilatih dengan dataset publik berkualitas tinggi. Kamu bisa memaksa dataset publik tersebut "meniru" karakteristik kamera murah tanpa perlu menambah data baru:

* **Downscale & Upscale:** Resize gambar publik dari 1080p ke 360p/480p, lalu resize kembali ke ukuran input YOLO (misal 640x640) menggunakan interpolasi linear untuk menciptakan efek pixelated/buram.
* **Kompresi JPEG Ekstrem:** Berikan kompresi kualitas rendah (quality factor 20–40) untuk memunculkan artefak kompresi khas kamera CCTV murah.
* **Gaussian Noise & Color Jitter:** Tambahkan sensor noise, kurangi kontras, dan ubah saturasi untuk meniru sensor CMOS murah dalam kondisi pencahayaan minim.

#### 3. Semi-Supervised Learning / Pseudo-Labeling dengan Filter Ketat

Jika tetap ingin memanfaatkan model YOLO yang sudah ada untuk melabeli rekaman kamera lokal:

* Pasang **Confidence Threshold tinggi** (misal $\ge 0.85$ atau $0.90$). Hanya simpan deteksi yang modelnya benar-benar yakin.
* Filter gambar negatif: Ambil juga frame kamera murah yang sudah dipastikan **tidak ada api/asap** sama sekali sebagai *background images* (file gambar tanpa file anotasi `.txt`). Ini melatih YOLO agar tidak sering memunculkan *false alarm*.

#### 4. Model-Assisted Annotation (Human-in-the-Loop)

Jika tetap membutuhkan verifikasi manusia agar akurasi aman untuk produksi:

* Gunakan tools seperti **Label Studio**, **CVAT**, atau **Roboflow**.
* Pasang model YOLO atau Grounding DINO sebagai pre-labeler otomatis.
* Tugasmu bukan menggambar kotak dari nol, melainkan hanya menekan tombol *accept*, menggeser kotak yang meleset, atau menghapus deteksi palsu. Cara ini memangkas waktu anotasi hingga 80–90%.

---

### Rekomendasi Urutan Eksekusi

1. **Langkah 1 (Tanpa data baru):** Tambahkan augmentasi degradasi (downscale, blur, noise, compression) ke dataset publik yang sudah ada, lalu latih ulang YOLO. Sering kali ini langsung menaikkan deteksi pada kamera murah secara signifikan.
2. **Langkah 2 (Jika masih kurang):** Ambil 200–500 frame dari kamera murah, auto-label menggunakan **Grounding DINO** atau **Florence-2**, lakukan validasi cepat, lalu gabungkan ke data latih.

---

Untuk kebutuhan *fine-tuning* adaptasi domain kamera murah, kamu membutuhkan sekitar **300 hingga 500 gambar** yang bervariasi. Jika kamera memantau area yang variasinya ekstrem (perubahan terang-gelap tajam, lampu sorot, atau mode *infrared/night vision* hitam-putih), angka optimalnya berada di kisaran **800 hingga 1.000 gambar**.

Berikut komposisi dan aturan pengambilan data agar model tidak *overfitting* atau sering memicu alarm palsu:

---

### Komposisi Dataset Rekomendasi

| Kategori | Proporsi | Estimasi Jumlah (Target 500 Gambar) | Tujuan Teknis |
| --- | --- | --- | --- |
| **Sampel Positif** (Api & Asap) | 70% | 350 gambar | Mengenalkan fitur api/asap pada karakteristik visual kamera murah (resolusi rendah, *blur*, noise sensor). |
| **Sampel Background / Negatif** (Tanpa Api) | 30% | 150 gambar | Mengajari YOLO untuk tidak salah mengira lampu, pantulan matahari, kabut, atau uap sebagai api/asap. |

---

### Aturan Kritis Saat Mengambil Gambar

1. **Gunakan Interval Sampling (Jangan Ambil Frame Berurutan)**
* Jika kamu merekam video uji coba (misal membakar kertas/kayu atau menyalakan asap), jangan potong video pada 30 frame per detik (fps). Frame yang hanya berselisih milidetik secara visual bernilai sama bagi model dan memicu *overfitting*.
* Ekstrak frame dengan jarak **1 frame per 1–2 detik**, atau hanya saat bentuk asap/lidah api sudah berubah bentuk dan posisi.


2. **Koleksi Kondisi Pencahayaan yang Berbeda**
Sensor kamera murah bereaksi sangat berbeda terhadap cahaya:
* **Siang hari / *Overexposure*:** Api sering kali terlihat putih pudar, bukan oranye.
* **Malam hari / *Low-light*:** Sensor akan memunculkan *grain/noise* tebal yang sering disalahartikan model sebagai tekstur asap.
* **Mode IR / Night Vision:** Jika kamera berganti ke mode monokrom hitam-putih di malam hari, model butuh contoh api/asap dalam spektrum hitam-putih tersebut.


3. **Format Gambar Background (Wajib Tanpa Anotasi)**
* Untuk 100–150 gambar negatif (area kosong tanpa api/asap), masukkan file gambar `.jpg` ke dalam folder data latih, tetapi **kosongkan file teks `.txt` pasangannya** (file ukuran 0 bytes).
* Standar resmi YOLO menggunakan file kosong ini untuk menekan tingkat *False Positive Rate* (FPR).

---

Cara paling ideal adalah **menggabungkan (joint training)** dataset publik dengan dataset lokalmu ke dalam satu siklus latih, bukan melatih dataset lokal secara terpisah di akhir tanpa menyertakan data lama.

Jika kamu melatih model dengan dataset publik sampai selesai, lalu melakukan *fine-tuning* hanya menggunakan data kamera murah, model berisiko mengalami **Catastrophic Forgetting** (bobot model "lupa" variasi bentuk api/asap umum yang dipelajari dari dataset publik dan mengalami *overfitting* pada kamera murah tersebut).

---

### Strategi Pelatihan yang Tepat

Pilih salah satu dari dua skenario berikut tergantung kondisi komputasimu:

#### Skenario A: Gabung Langsung (Joint Training) — *Paling Direkomendasikan*

Gunakan skema ini jika kamu memiliki GPU yang cukup untuk melatih seluruh dataset secara bersamaan:

1. Mulai dari bobot dasar resmi YOLO (misal `yolov8n.pt` atau `yolov11n.pt`).
2. Satukan folder dataset publik dan dataset kamera murah ke dalam satu `data.yaml`.
3. Latih secara bersamaan selama 50–100 epoch.

* **Kelebihan:** Model langsung mempelajari fitur api/asap beresolusi tinggi dan beresolusi rendah secara seimbang.

#### Skenario B: Two-Stage Fine-Tuning (Jika Dataset Publik Sudah Terlanjur Dilatih)

Gunakan skema ini jika modelmu saat ini sudah selesai dilatih dengan dataset publik dan kamu tidak ingin membuang waktu melatih ulang dari nol:

1. Ambil checkpoint model terbaikmu saat ini (`best.pt`).
2. Siapkan dataset baru yang berisi: **Seluruh data kamera lokalmu + 20%–30% sampel acak dari dataset publik** (*experience replay* untuk mencegah model lupa).
3. Lakukan *fine-tuning* dengan *learning rate* lebih kecil (misalnya turunkan `lr0` bawaan dari 0.01 menjadi `0.001` atau `0.0005`) selama 20–40 epoch.
4. *Opsional:* Freeze beberapa layer awal (*backbone*) agar fitur visual dasar tidak rusak, cukup perbarui layer deteksi (*head*).

---

### Jumlah Dataset Minimal dari Kamera Lokal

Batas minimal absolut untuk dataset lokal adalah **100 hingga 150 gambar**, dengan pembagian:

* **70 – 100 gambar positif:** Mengandung api dan asap dari rekaman kamera murah.
* **30 – 50 gambar negatif (background):** Gambar area pantauan kamera tanpa api/asap sama sekali (file gambar ada, file `.txt` kosong).

#### Syarat agar 100–150 Gambar Ini Cukup:

1. **Posisi Kamera Statis (Fixed Angle):** Kamera CCTV terpasang mati di satu titik dan sudut pandang tidak berubah-ubah.
2. **Variasi Cepat Terwakili:** 100 frame tersebut sudah mencakup kondisi pencahayaan yang berbeda (misal 40 gambar siang, 40 gambar sore/malam, 20 gambar dengan lampu menyala/padam).
3. **Ukuran Objek Beragam:** Api/asap diambil dari jarak dekat dan jarak jauh di dalam tangkapan kamera tersebut.

Jika kamera sering digerakkan (*PTZ camera*), lingkungan luar ruangan berubah drastis akibat cuaca/matahari, atau berpindah mode ke *night vision* monokrom, batas minimal 100 gambar tidak akan cukup dan perlu dinaikkan ke **300+ gambar**.