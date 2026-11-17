"""
Script deteksi real-time YOLO untuk Kamera 2.
Menggunakan Indeks Kamera 2 dan calib_cam2.npz dengan akselerasi hardware MJPG.
"""
import sys
import os

# Import main dari detect_cam1
from detect_cam1 import main

if __name__ == "__main__":
    # Atur default camera index ke 2 jika tidak ditentukan
    if "--camera" not in sys.argv:
        sys.argv.extend(["--camera", "2"])
    # Atur default file kalibrasi ke calib_cam2.npz jika tidak ditentukan
    if "--calib" not in sys.argv:
        sys.argv.extend(["--calib", "calib_cam2.npz"])
    main()
