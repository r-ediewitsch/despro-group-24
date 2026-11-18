"""
=====================================================================
DEEZPRO INTEGRATION - KINEMATICS ENGINE
=====================================================================
File: kinematics.py
Deskripsi: Modul perhitungan translasi koordinat fisik (cm) ke langkah
            motor stepper (steps), termasuk fungsi penjepitan batas
            (clamping) terhadap area clearance railing M x N.
"""

from typing import Tuple, Dict, Any


class KinematicsEngine:
    """
    Kinematics Engine untuk sistem railing Cartesian XY Deezpro.
    
    Menangani:
    1. Validasi boundary dan clamping ke titik terdekat jika koordinat di area clearance.
    2. Konversi koordinat fisik dunia (cm) ke langkah motor stepper (steps).
    3. Konversi balik dari posisi langkah (steps) ke koordinat fisik (cm).
    """

    def __init__(
        self,
        width_m_cm: float = 200.0,
        height_n_cm: float = 200.0,
        clearance_x_cm: float = 10.0,
        clearance_y_cm: float = 10.0,
        total_steps_p: int = 8000,
        total_steps_q: int = 8000,
    ):
        self.width_m = float(width_m_cm)
        self.height_n = float(height_n_cm)
        self.clearance_x = float(clearance_x_cm)
        self.clearance_y = float(clearance_y_cm)
        self.total_steps_p = int(total_steps_p)
        self.total_steps_q = int(total_steps_q)

        # Rentang gerak efektif railing (M - 2m dan N - 2n)
        self.travel_x_cm = self.width_m - (2.0 * self.clearance_x)
        self.travel_y_cm = self.height_n - (2.0 * self.clearance_y)

        if self.travel_x_cm <= 0 or self.travel_y_cm <= 0:
            raise ValueError(
                f"Dimensi frame ({self.width_m}x{self.height_n}) harus lebih besar "
                f"dari dua kali clearance ({self.clearance_x*2}x{self.clearance_y*2})!"
            )

        # Faktor konversi (steps per cm)
        self.steps_per_cm_x = float(self.total_steps_p) / self.travel_x_cm
        self.steps_per_cm_y = float(self.total_steps_q) / self.travel_y_cm

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KinematicsEngine":
        """Inisialisasi KinematicsEngine dari dictionary konfigurasi YAML."""
        frame_cfg = config.get("frame", {})
        clearance_cfg = config.get("clearance", {})
        stepper_cfg = config.get("stepper", {})

        return cls(
            width_m_cm=frame_cfg.get("width_m_cm", 200.0),
            height_n_cm=frame_cfg.get("height_n_cm", 200.0),
            clearance_x_cm=clearance_cfg.get("clearance_x_cm", 10.0),
            clearance_y_cm=clearance_cfg.get("clearance_y_cm", 10.0),
            total_steps_p=stepper_cfg.get("total_steps_x_p", 8000),
            total_steps_q=stepper_cfg.get("total_steps_y_q", 8000),
        )

    def clamp_coordinates(self, x_cm: float, y_cm: float) -> Tuple[float, float, bool]:
        """
        Menjepit koordinat target ke dalam batas jangkauan efektif railing.
        Jika koordinat api berada di zona clearance, mencari titik terdekat.

        Returns:
            Tuple[clamped_x_cm, clamped_y_cm, is_clamped_flag]
        """
        min_x = self.clearance_x
        max_x = self.width_m - self.clearance_x

        min_y = self.clearance_y
        max_y = self.height_n - self.clearance_y

        clamped_x = max(min_x, min(x_cm, max_x))
        clamped_y = max(min_y, min(y_cm, max_y))

        is_clamped = (clamped_x != x_cm) or (clamped_y != y_cm)
        return clamped_x, clamped_y, is_clamped

    def coordinates_to_steps(self, x_cm: float, y_cm: float) -> Tuple[int, int, Tuple[float, float], bool]:
        """
        Menerjemahkan koordinat fisik api (cm) menjadi target langkah motor (steps).

        Returns:
            Tuple[target_step_x, target_step_y, (clamped_x_cm, clamped_y_cm), is_clamped]
        """
        clamped_x, clamped_y, is_clamped = self.clamp_coordinates(x_cm, y_cm)

        # Hitung langkah relatif dari titik asal efektif (titik (m, n))
        delta_x = clamped_x - self.clearance_x
        delta_y = clamped_y - self.clearance_y

        step_x = int(round(delta_x * self.steps_per_cm_x))
        step_y = int(round(delta_y * self.steps_per_cm_y))

        # Batasan software endstop demi keamanan mekanis
        step_x = max(0, min(step_x, self.total_steps_p))
        step_y = max(0, min(step_y, self.total_steps_q))

        return step_x, step_y, (clamped_x, clamped_y), is_clamped

    def steps_to_coordinates(self, step_x: int, step_y: int) -> Tuple[float, float]:
        """
        Menerjemahkan posisi langkah motor stepper (steps) kembali ke koordinat fisik (cm).
        """
        safe_step_x = max(0, min(int(step_x), self.total_steps_p))
        safe_step_y = max(0, min(int(step_y), self.total_steps_q))

        phys_x = self.clearance_x + (safe_step_x / self.steps_per_cm_x)
        phys_y = self.clearance_y + (safe_step_y / self.steps_per_cm_y)

        return round(phys_x, 2), round(phys_y, 2)

    def get_kinematic_summary(self) -> Dict[str, Any]:
        """Mengambil ringkasan parameter konfigurasi mekanis."""
        return {
            "frame_dimension_cm": (self.width_m, self.height_n),
            "clearance_cm": (self.clearance_x, self.clearance_y),
            "effective_travel_cm": (self.travel_x_cm, self.travel_y_cm),
            "total_steps": (self.total_steps_p, self.total_steps_q),
            "resolution_steps_per_cm": (round(self.steps_per_cm_x, 2), round(self.steps_per_cm_y, 2)),
            "resolution_mm_per_step": (
                round(10.0 / self.steps_per_cm_x, 4),
                round(10.0 / self.steps_per_cm_y, 4),
            ),
        }
