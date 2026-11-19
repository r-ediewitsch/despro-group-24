"""
Unit test untuk Fire & Smoke Detector Engine Deezpro.
"""

import os
import sys
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.vision.detector import FireSmokeDetector, PixelDetection


class TestFireSmokeDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ultralytics
        except ImportError:
            raise unittest.SkipTest("Library 'ultralytics' belum terpasang di Python environment ini.")
        model_path = os.path.join(current_dir, "models", "best.pt")
        cls.detector = FireSmokeDetector(model_path=model_path, conf_threshold=0.30)

    def test_model_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIn(0, self.detector.class_names)
        self.assertIn(1, self.detector.class_names)
        self.assertEqual(self.detector.class_names[0], "smoke")
        self.assertEqual(self.detector.class_names[1], "fire")

    def test_detection_structure(self):
        # Buat objek PixelDetection simulasi
        det = PixelDetection(
            cls_id=1,
            cls_name="fire",
            confidence=0.88,
            bbox=(100, 150, 300, 450),
            centroid=(200, 300),
            base_center=(200, 450),
            is_fire=True,
            is_smoke=False,
        )

        self.assertEqual(det.width_px, 200)
        self.assertEqual(det.height_px, 300)
        self.assertEqual(det.centroid, (200, 300))
        self.assertEqual(det.base_center, (200, 450))
        self.assertTrue(det.is_fire)
        self.assertFalse(det.is_smoke)

    def test_inference_on_blank_image(self):
        # Uji inferensi pada citra hitam (tidak boleh crash, harus menghasilkan list)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(blank)
        self.assertIsInstance(results, list)

    def test_draw_detections(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        det = PixelDetection(
            cls_id=1,
            cls_name="fire",
            confidence=0.92,
            bbox=(50, 50, 150, 200),
            centroid=(100, 125),
            base_center=(100, 200),
            is_fire=True,
            is_smoke=False,
        )
        drawn = self.detector.draw_detections(blank, [det], target_mode="base")
        self.assertEqual(drawn.shape, blank.shape)


if __name__ == "__main__":
    unittest.main()
