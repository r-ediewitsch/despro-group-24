"""
Unit test untuk Homography Engine Deezpro.
"""

import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from src.vision.homography import HomographyEngine


class TestHomographyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HomographyEngine()

        # Titik simulasi perspektif kamera miring memandang lantai 200x200 cm
        # Titik dunia nyata di lantai (cm):
        self.world_pts = [
            (10.0, 10.0),    # Kiri-bawah
            (190.0, 10.0),   # Kanan-bawah
            (190.0, 190.0),  # Kanan-atas
            (10.0, 190.0),   # Kiri-atas
            (100.0, 100.0),  # Pusat tengah
        ]

        # Koordinat piksel terproyeksi (trapesium perspektif khas kamera miring):
        self.img_pts = [
            (220.0, 850.0),  # Kiri-bawah
            (1700.0, 850.0), # Kanan-bawah
            (1350.0, 320.0), # Kanan-atas
            (570.0, 320.0),  # Kiri-atas
            (960.0, 520.0),  # Tengah
        ]

        for ipt, wpt in zip(self.img_pts, self.world_pts):
            self.engine.add_point_pair(ipt, wpt)

    def test_compute_and_reprojection(self):
        H, err, mask = self.engine.compute()
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (3, 3))
        # Rata-rata error reprojeksi pada titik sintetis harus sangat kecil (< 3 cm)
        self.assertLess(err, 3.0)

    def test_forward_projection(self):
        self.engine.compute()
        # Uji titik kiri-bawah (220, 850) -> harus mendekati (10, 10) cm
        xw, yw = self.engine.project_image_to_world(220.0, 850.0)
        self.assertAlmostEqual(xw, 10.0, delta=2.0)
        self.assertAlmostEqual(yw, 10.0, delta=2.0)

        # Uji titik kanan-bawah (1700, 850) -> harus mendekati (190, 10) cm
        xw, yw = self.engine.project_image_to_world(1700.0, 850.0)
        self.assertAlmostEqual(xw, 190.0, delta=2.0)
        self.assertAlmostEqual(yw, 10.0, delta=2.0)

    def test_round_trip_inverse(self):
        self.engine.compute()
        # Uji transformasi bolak-balik: World -> Image -> World
        orig_x, orig_y = 100.0, 100.0
        u, v = self.engine.project_world_to_image(orig_x, orig_y)
        calc_x, calc_y = self.engine.project_image_to_world(u, v)
        self.assertAlmostEqual(orig_x, calc_x, delta=0.5)
        self.assertAlmostEqual(orig_y, calc_y, delta=0.5)

    def test_save_and_load(self):
        self.engine.compute()
        test_file = "test_homography_temp.npz"
        try:
            self.engine.save(test_file, extra_meta={"camera_id": 1})
            loaded = HomographyEngine.load(test_file)
            self.assertIsNotNone(loaded.H)
            self.assertEqual(len(loaded.image_points), 5)
            self.assertEqual(len(loaded.world_points), 5)

            # Proyeksi harus identik
            w1 = self.engine.project_image_to_world(570.0, 320.0)
            w2 = loaded.project_image_to_world(570.0, 320.0)
            self.assertAlmostEqual(w1[0], w2[0], places=4)
            self.assertAlmostEqual(w1[1], w2[1], places=4)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
