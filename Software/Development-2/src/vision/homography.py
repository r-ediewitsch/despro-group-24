"""
=====================================================================
DEEZPRO VISION - HOMOGRAPHY ENGINE & CALIBRATOR
=====================================================================
File: homography.py
Deskripsi: Modul estimasi dan transformasi homografi planar antara bidang
            citra kamera (u, v piksel) dengan bidang fisik lantai
            (X_w, Y_w centimeter) pada sistem frame 2x2 meter.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


class HomographyEngine:
    """
    Engine untuk mengestimasi, menerapkan, dan memvalidasi
    transformasi homografi planar kamera ke lantai (ground plane).
    """

    def __init__(self, H_matrix: Optional[np.ndarray] = None):
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None

        if H_matrix is not None:
            self.set_matrix(H_matrix)

        self.image_points: List[Tuple[float, float]] = []
        self.world_points: List[Tuple[float, float]] = []
        self.reprojection_error_cm: float = 0.0

    def set_matrix(self, H: np.ndarray):
        """Menetapkan matriks homografi 3x3 dan menghitung inversnya."""
        if H.shape != (3, 3):
            raise ValueError(f"Matriks homografi harus berukuran 3x3, diterima: {H.shape}")
        self.H = H.astype(np.float64)
        # Normalisasi elemen h33 agar bernilai 1.0 jika tidak nol
        if abs(self.H[2, 2]) > 1e-9:
            self.H /= self.H[2, 2]
        self.H_inv = np.linalg.inv(self.H)

    def add_point_pair(self, img_pt: Tuple[float, float], world_pt_cm: Tuple[float, float]):
        """Menambahkan satu pasang titik korespondensi citra (u, v) dan dunia nyata (X, Y cm)."""
        self.image_points.append((float(img_pt[0]), float(img_pt[1])))
        self.world_points.append((float(world_pt_cm[0]), float(world_pt_cm[1])))

    def clear_points(self):
        """Menghapus seluruh daftar titik korespondensi."""
        self.image_points.clear()
        self.world_points.clear()
        self.reprojection_error_cm = 0.0

    def compute(self, method=cv2.RANSAC, ransac_thresh: float = 3.0) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Menghitung matriks homografi 3x3 dari titik-titik korespondensi yang telah didaftarkan.
        Minimal dibutuhkan 4 pasang titik (disarankan 6 - 12 pasang titik).

        Returns:
            Tuple[H_matrix, mean_reprojection_error_cm, inlier_mask]
        """
        n_pts = len(self.image_points)
        if n_pts < 4:
            raise ValueError(f"Dibutuhkan minimal 4 pasang titik korespondensi, saat ini baru ada {n_pts} titik!")

        src_pts = np.array(self.image_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(self.world_points, dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, method, ransacReprojThreshold=ransac_thresh)
        if H is None:
            raise RuntimeError("Gagal menghitung matriks homografi! Pastikan titik tidak berada pada satu garis lurus (collinear).")

        self.set_matrix(H)
        self.reprojection_error_cm = self.evaluate_reprojection_error()

        return self.H, self.reprojection_error_cm, mask

    def project_image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Mentransformasikan koordinat piksel citra (u, v) menjadi koordinat dunia fisik lantai (X_w, Y_w dalam cm).
        Rumus Matematis (Proposal Bagian 2.2.4 & 4.2.2):
            X_w = (h11*u + h12*v + h13) / (h31*u + h32*v + h33)
            Y_w = (h21*u + h22*v + h23) / (h31*u + h32*v + h33)
        """
        if self.H is None:
            raise RuntimeError("Matriks homografi belum diinisialisasi atau dihitung!")

        # Vektor koordinat homogen [u, v, 1]^T
        p_img = np.array([u, v, 1.0], dtype=np.float64)
        p_world = np.dot(self.H, p_img)

        scale = p_world[2]
        if abs(scale) < 1e-9:
            scale = 1e-9

        x_w = p_world[0] / scale
        y_w = p_world[1] / scale

        return float(x_w), float(y_w)

    def project_world_to_image(self, x_w_cm: float, y_w_cm: float) -> Tuple[float, float]:
        """
        Mentransformasikan koordinat dunia fisik (X_w, Y_w cm) kembali ke koordinat piksel citra (u, v).
        """
        if self.H_inv is None:
            raise RuntimeError("Matriks invers homografi belum tersedia!")

        p_world = np.array([x_w_cm, y_w_cm, 1.0], dtype=np.float64)
        p_img = np.dot(self.H_inv, p_world)

        scale = p_img[2]
        if abs(scale) < 1e-9:
            scale = 1e-9

        u = p_img[0] / scale
        v = p_img[1] / scale

        return float(u), float(v)

    def evaluate_reprojection_error(self) -> float:
        """
        Menghitung Mean Reprojection Error (dalam cm) antara titik dunia riil
        dengan hasil proyeksi ulang dari koordinat citra.
        """
        if self.H is None or len(self.image_points) == 0:
            return 0.0

        errors = []
        for (u, v), (real_x, real_y) in zip(self.image_points, self.world_points):
            pred_x, pred_y = self.project_image_to_world(u, v)
            dist = np.sqrt((pred_x - real_x) ** 2 + (pred_y - real_y) ** 2)
            errors.append(dist)

        return float(np.mean(errors))

    def generate_top_down_view(
        self,
        image: np.ndarray,
        view_size_px: Tuple[int, int] = (600, 600),
        max_world_cm: Tuple[float, float] = (200.0, 200.0),
    ) -> np.ndarray:
        """
        Menghasilkan Bird's-Eye View (Warped Top-Down View) dari lantai 2x2 meter.
        Berguna untuk memverifikasi secara visual bahwa lantai terlihat ortogonal dari atas.
        """
        if self.H is None:
            raise RuntimeError("Matriks homografi belum dihitung!")

        vw, vh = view_size_px
        mw_x, mw_y = max_world_cm

        # Matriks skala konversi dari cm ke piksel tampilan Bird's Eye
        # (0, 0) cm -> (0, 0) px, (mw_x, mw_y) cm -> (vw, vh) px
        S = np.array([
            [vw / mw_x, 0.0, 0.0],
            [0.0, vh / mw_y, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Matriks warp total: H_warp = S * H
        H_warp = np.dot(S, self.H)

        top_down = cv2.warpPerspective(image, H_warp, (vw, vh), flags=cv2.INTER_LINEAR)
        return top_down

    def save(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Menyimpan matriks homografi, titik kalibrasi, dan galat ke file .npz."""
        if self.H is None:
            raise RuntimeError("Tidak ada matriks homografi yang dapat disimpan!")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data_to_save = {
            "homography_matrix": self.H,
            "image_points": np.array(self.image_points, dtype=np.float32),
            "world_points": np.array(self.world_points, dtype=np.float32),
            "reprojection_error_cm": np.float32(self.reprojection_error_cm),
        }
        if extra_meta:
            for k, v in extra_meta.items():
                data_to_save[k] = v

        np.savez(filepath, **data_to_save)

    @classmethod
    def load(cls, filepath: str) -> "HomographyEngine":
        """Memuat matriks homografi dan metadata dari file .npz."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File homografi tidak ditemukan: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        H = data["homography_matrix"]

        engine = cls(H)
        if "image_points" in data:
            engine.image_points = [tuple(pt) for pt in data["image_points"]]
        if "world_points" in data:
            engine.world_points = [tuple(pt) for pt in data["world_points"]]
        if "reprojection_error_cm" in data:
            engine.reprojection_error_cm = float(data["reprojection_error_cm"])

        return engine
