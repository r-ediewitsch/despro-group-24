# Versi Saat Ini

- Sistem terdiri dari 2 ESP32 yang masing-masing mengendalikan 1 motor. Motor sumbu-X dan motor sumbu-Y.
  - Motor sumbu-X dikendalikan dengan kode `stepper-test.ino`.
  - Motor sumbu-Y dikendalikan dengan kode `stepper-test-2.ino`.
- Kontrol dilakukan menggunakan protokol MQTT. Kode transmitter terdapat pada `../Python Transmitter/main.py` menggunakan 2 topik:
  - Sumbu-X dikendalikan dengan topic `my_stepper_esp32/1/{cmd|status}`
  - Sumbu-Y dikendalikan dengan topic `my_stepper_esp32/2/{cmd|status}`
- Sistem saat ini masih dikendalikan secara manual dengan terminal, dimana user menginputkan **jumlah step** beserta arahnya (dalam bentuk +/-).

---

# Versi To-Be

- Sistem akan terdiri dari 3 ESP32 yang terhubung dengan protokol MQTT:
  - ESP 1 mengendalikan motor sumbu-X
  - ESP 2 mengendalikan motor sumbu-Y
  - ESP 3 mengendalikan servo untuk menyemprotkan APAR.
- Kontrol dilakukan dengan 6 topic MQTT, mirip dengan versi saat ini.
- Berbeda dengan versi saat ini yang mengontrol pergerakan APAR menggunakan *step motor* secara manual, versi to-be mengontrolnya menggunakan **koordinat**
  - Sistem terpasang pada sebuah *frame* M x N, pada pengukuran awal, M dan N memiliki panjang 2 meter.
  - Terdapat selisih antara panjang *frame* dengan pergerakan maksimal dari sistem railing dikarenakan *clearance* pemasangan rail.
    - Anggap di kedua sisi pada masing-masing sumbu, terdapat selisih m cm untuk sumbu-X dan n cm untuk sumbu-Y
    - Maka pergerakan maksimal pada sumbu-X adalah M-2m dan  pergerakan maksimal pada sumbu-Y adalah N-2n.
    - Jika koordinat api terletak pada titik dimana selisih tersebut berada, maka program harus bisa menghitung **titik terdekat** dengan titik api, meskipun tidak tepat di atasnya.
  - Jumlah step pada motor sumbu-X dari ujung-ke-ujung didefinisikan sebagai P. Jumlah step pada motor sumbu-Y dari ujung-ke-ujung didefinisikan sebagai Q. Jumlah step ini dihitung setelah dikurangi dengan *clearance* rail tadi.
- Program harus bisa menghitung jumlah step yang dibutuhkan untuk masing-masing sumbu berdasarkan koordinat yang dikirimkan oleh sistem deteksi api.
- Untuk flownya secara umum: Kamera Mendeteksi Api -> Koordinat Api Didapatkan dari Proses Homografi -> Program Men-Translate Koordinat ke Step -> Kedua Sumbu Railing Bergerak Berdasarkan Jumlah Step Bersamaan -> Setelah Sumbu Selesai Bergerak, Semprot APAR.

---

# Yang Perlu Dipertimbangkan

- Dimana seharusnya perhitungan step dilakukan? Apakah pada kode Python yang men-transmit command, atau pada edge di ESP32?
- Sistem harus menyimpan koordinat terakhir beserta jumlah step relatifnya secara global. Bagaimana arsitektur sistem seharusnya? Apakah menggunakan database atau bagaimana?
- Bagaimana sistem secara umum? Apa yang perlu ditampilkan di frontend, apa yang harus dihandle di backend? Apakah view kamera perlu ditampilkan, apakah perlu log? Berikan pendapat kamu!
- Berikan juga string yang digunakan untuk komunikasi menggunakan MQTT agar sistem terstandarisasi.
