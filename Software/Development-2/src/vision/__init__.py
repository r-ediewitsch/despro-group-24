"""
Deezpro Vision Package (Placeholder interface for vision subsystem)
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FireTarget:
    """Kontrak data output deteksi api dari Vision Pipeline."""
    x_cm: float                           # Koordinat X fisik hasil estimasi
    y_cm: float                           # Koordinat Y fisik hasil estimasi
    confidence: float                     # Tingkat keyakinan deteksi (0.0 - 1.0)
    has_smoke: bool = False               # Indikator deteksi asap
    is_single_cam: bool = False           # True jika estimasi berasal dari 1 kamera (fallback)
    cam1_pos: Optional[Tuple[float, float]] = None # Posisi mentah Kamera 1
    cam2_pos: Optional[Tuple[float, float]] = None # Posisi mentah Kamera 2
    timestamp: float = 0.0                # Unix timestamp saat deteksi
    is_stable: bool = True                # True jika terverifikasi stabil >= 5 frame
