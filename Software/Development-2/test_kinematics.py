"""
Unit test untuk Kinematics Engine Deezpro.
"""

import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.integration.kinematics import KinematicsEngine


class TestKinematicsEngine(unittest.TestCase):
    def setUp(self):
        # Frame 200x200 cm, clearance 10 cm, P=8000, Q=8000
        # Effective travel = 180 cm, steps/cm = 8000 / 180 = 44.444...
        self.kin = KinematicsEngine(
            width_m_cm=200.0,
            height_n_cm=200.0,
            clearance_x_cm=10.0,
            clearance_y_cm=10.0,
            total_steps_p=8000,
            total_steps_q=8000,
        )

    def test_effective_travel(self):
        self.assertEqual(self.kin.travel_x_cm, 180.0)
        self.assertEqual(self.kin.travel_y_cm, 180.0)

    def test_origin_point(self):
        # Titik minimum efektif (10, 10) harus menghasilkan step (0, 0)
        sx, sy, clamped, is_clamped = self.kin.coordinates_to_steps(10.0, 10.0)
        self.assertEqual(sx, 0)
        self.assertEqual(sy, 0)
        self.assertEqual(clamped, (10.0, 10.0))
        self.assertFalse(is_clamped)

    def test_max_point(self):
        # Titik maksimum efektif (190, 190) harus menghasilkan step (8000, 8000)
        sx, sy, clamped, is_clamped = self.kin.coordinates_to_steps(190.0, 190.0)
        self.assertEqual(sx, 8000)
        self.assertEqual(sy, 8000)
        self.assertEqual(clamped, (190.0, 190.0))
        self.assertFalse(is_clamped)

    def test_center_point(self):
        # Titik tengah frame (100, 100) harus menghasilkan step setengahnya (4000, 4000)
        sx, sy, clamped, is_clamped = self.kin.coordinates_to_steps(100.0, 100.0)
        self.assertEqual(sx, 4000)
        self.assertEqual(sy, 4000)
        self.assertEqual(clamped, (100.0, 100.0))
        self.assertFalse(is_clamped)

    def test_clamping_in_clearance(self):
        # Kasus api di zona clearance (X = 2 cm, Y = 5 cm), harus di-clamp ke titik terdekat (10, 10)
        sx, sy, clamped, is_clamped = self.kin.coordinates_to_steps(2.0, 5.0)
        self.assertTrue(is_clamped)
        self.assertEqual(clamped, (10.0, 10.0))
        self.assertEqual(sx, 0)
        self.assertEqual(sy, 0)

    def test_clamping_beyond_frame(self):
        # Kasus api di luar frame (X = 250 cm, Y = 195 cm)
        sx, sy, clamped, is_clamped = self.kin.coordinates_to_steps(250.0, 195.0)
        self.assertTrue(is_clamped)
        self.assertEqual(clamped, (190.0, 190.0))
        self.assertEqual(sx, 8000)
        self.assertEqual(sy, 8000)

    def test_round_trip_conversion(self):
        # Uji konversi bolak-balik
        original_x, original_y = 75.0, 140.0
        sx, sy, _, _ = self.kin.coordinates_to_steps(original_x, original_y)
        calc_x, calc_y = self.kin.steps_to_coordinates(sx, sy)
        self.assertAlmostEqual(original_x, calc_x, delta=0.1)
        self.assertAlmostEqual(original_y, calc_y, delta=0.1)


if __name__ == "__main__":
    unittest.main()
