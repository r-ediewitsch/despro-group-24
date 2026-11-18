"""
=====================================================================
DEEZPRO VISION - FIRE & SMOKE DETECTOR (YOLO best.pt)
=====================================================================
File: detector.py
Deskripsi: Modul pendeteksi objek api (fire) dan asap (smoke) murni
            menggunakan model YOLO (best.pt). Menghasilkan informasi
            deteksi spasial berbasis koordinat piksel citra: bounding box
            (x1, y1, x2, y2), centroid geometris (u_c, v_c), dan
            titik dasar kontak lantai (u_b, v_b).
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

# Redam error izin Ultralytics settings di lingkungan terbatas
os.environ["YOLO_CONFIG_DIR"] = os.path.join(os.path.expanduser("~"), ".ultralytics_temp")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class PixelDetection:
    """
    Struktur data hasil deteksi objek api/asap pada sistem koordinat piksel citra.
    """
    cls_id: int                           # 0: smoke, 1: fire (sesuai best.pt)
    cls_name: str                         # "smoke" atau "fire"
    confidence: float                     # Nilai keyakinan (0.0 s.d. 1.0)
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2) piksel
    centroid: Tuple[int, int]             # (u_c, v_c) titik tengah bounding box
    base_center: Tuple[int, int]          # (u_b, v_b) titik dasar api yang menyentuh lantai
    is_fire: bool
    is_smoke: bool

    @property
    def width_px(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height_px(self) -> int:
        return self.bbox[3] - self.bbox[1]


class FireSmokeDetector:
    """
    Detector Engine berbasis Ultralytics YOLO untuk klasifikasi dan
    lokalisasi piksel bahaya api dan asap.
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf_threshold: float = 0.40,
        imgsz: int = 640,
    ):
        if YOLO is None:
            raise ImportError(
                "Library 'ultralytics' belum terpasang! Pasang dengan perintah: pip install ultralytics"
            )

        self.model_path = self._resolve_model_path(model_path)
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)

        # Muat bobot model YOLO
        self.model = YOLO(self.model_path)
        self.class_names = dict(self.model.names) if hasattr(self.model, "names") else {0: "smoke", 1: "fire"}

        # Identifikasi kelas ID api dan asap
        self.fire_class_ids = []
        self.smoke_class_ids = []

        fire_keywords = {"fire", "api", "flame"}
        smoke_keywords = {"smoke", "asap"}

        for cid, name in self.class_names.items():
            name_lower = str(name).lower()
            if any(k in name_lower for k in fire_keywords):
                self.fire_class_ids.append(cid)
            elif any(k in name_lower for k in smoke_keywords):
                self.smoke_class_ids.append(cid)

        # Fallback jika model memiliki kelas 0: smoke, 1: fire standar
        if not self.fire_class_ids and 1 in self.class_names:
            self.fire_class_ids = [1]
        if not self.smoke_class_ids and 0 in self.class_names:
            self.smoke_class_ids = [0]

    def _resolve_model_path(self, path_str: str) -> str:
        """Mencari lokasi berkas model di path relatif atau folder alternatif."""
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str

        # 1. Cek langsung
        if os.path.exists(path_str):
            return os.path.abspath(path_str)

        # 2. Cek relatif terhadap direktori skrip ini
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(script_dir, path_str)
        if os.path.exists(candidate):
            return candidate

        # 3. Cek di models/best.pt atau callibration/best.pt_packed.pt
        candidates = [
            os.path.join(script_dir, "models", "best.pt"),
            os.path.join(script_dir, "models", "best.pt_packed.pt"),
            os.path.join(os.path.dirname(script_dir), "Development", "callibration", "best.pt_packed.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        raise FileNotFoundError(f"Berkas model YOLO tidak ditemukan pada: {path_str}")

    def detect(self, frame: np.ndarray) -> List[PixelDetection]:
        """
        Mengeksekusi inferensi YOLO pada frame citra dan mengekstrak koordinat piksel.

        Returns:
            List[PixelDetection] berisi objek-objek api/asap yang terdeteksi.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: List[PixelDetection] = []

        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            is_fire = cls_id in self.fire_class_ids
            is_smoke = cls_id in self.smoke_class_ids

            # Hitung Titik Representasi Piksel:
            # 1. Centroid Geometris: Pusat tengah bounding box
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))

            # 2. Base Center: Titik tengah bagian bawah (titik kontak lantai)
            bx = int(round((x1 + x2) / 2.0))
            by = int(round(y2))

            det = PixelDetection(
                cls_id=cls_id,
                cls_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                centroid=(cx, cy),
                base_center=(bx, by),
                is_fire=is_fire,
                is_smoke=is_smoke,
            )
            detections.append(det)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[PixelDetection],
        target_mode: str = "base",  # "base" (u_b, v_b) atau "centroid" (u_c, v_c)
    ) -> np.ndarray:
        """
        Menggambar bounding box, titik acuan piksel, dan label koordinat pada frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.centroid
            bx, by = det.base_center

            # Warna: Merah untuk api, Oranye/Amber untuk asap, Hijau neon untuk lainnya
            if det.is_fire:
                box_color = (0, 0, 255)       # Merah (BGR)
                pt_color = (0, 255, 255)      # Kuning terang
                thickness = 3
            elif det.is_smoke:
                box_color = (0, 165, 255)     # Amber / Oranye
                pt_color = (0, 200, 255)
                thickness = 2
            else:
                box_color = (0, 255, 128)
                pt_color = (0, 255, 255)
                thickness = 2

            # 1. Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # 2. Gambar Titik Centroid (Pusat)
            cv2.circle(annotated, (cx, cy), 4, (255, 0, 255), -1)  # Titik ungu

            # 3. Gambar Titik Base-Center (Kontak Lantai)
            cv2.circle(annotated, (bx, by), 6, pt_color, -1)       # Titik kuning
            # Crosshair pada target utama
            target_pt = (bx, by) if target_mode == "base" else (cx, cy)
            tx, ty = target_pt
            cv2.drawMarker(annotated, (tx, ty), pt_color, cv2.MARKER_CROSS, 20, 2)

            # 4. Garis vertikal penghubung centroid ke dasar
            cv2.line(annotated, (cx, cy), (bx, by), (200, 200, 200), 1, cv2.LINE_AA)

            # 5. Label Informasi Piksel
            label = (
                f"{det.cls_name.upper()} {det.confidence:.2f} | "
                f"Base: ({bx}, {by}) px | Centroid: ({cx}, {cy}) px"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

            # Background label box
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - 22)),
                (x1 + tw + 8, max(22, y1)),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
