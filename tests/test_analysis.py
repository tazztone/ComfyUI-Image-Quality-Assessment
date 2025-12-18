import sys
import os
import unittest
import numpy as np
import torch
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_nodes import (
    Analysis_BlurDetection,
    Analysis_Defocus,
    Analysis_SharpnessFocusScore,
    Analysis_Entropy,
    Analysis_EdgeDensity
)

class TestAnalysisStaticMethods(unittest.TestCase):
    """Verify helper methods are static and callable directly from the class."""

    def test_defocus_interpret(self):
        # Low score = sharp
        result = Analysis_Defocus.interpret(0.1)
        self.assertIn("Very sharp", result)

        # High score = blurry
        result = Analysis_Defocus.interpret(0.9)
        self.assertIn("Severe defocus", result)

    def test_blur_interpret(self):
        result = Analysis_BlurDetection.interpret_blur(25)
        self.assertIn("Very blurry", result)

        result = Analysis_BlurDetection.interpret_blur(400)
        self.assertIn("Very sharp", result)

    def test_sharpness_interpret(self):
        result = Analysis_SharpnessFocusScore.interpret_score(50, "Laplacian")
        self.assertIn("Very blurry", result)

        result = Analysis_SharpnessFocusScore.interpret_score(0.8, "Hybrid")
        self.assertIn("Very sharp", result)

    def test_entropy_interpret(self):
        result = Analysis_Entropy.interpret_entropy(1.0)
        self.assertIn("Very low entropy", result)

        result = Analysis_Entropy.interpret_entropy(7.0)
        self.assertIn("High entropy", result)


class TestAnalysisLogic(unittest.TestCase):
    """Test pure image processing logic."""

    def test_fft_analysis_returns_valid_outputs(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[20:40, 20:40] = 255  # Add a bright square

        score, heatmap, mask = Analysis_Defocus.fft_analysis(gray, "FFT Ratio (Sum)")

        self.assertIsInstance(score, (float, np.floating))
        # Heatmap from applyColorMap is BGR, so 3 channels
        self.assertEqual(heatmap.shape[2], 3)
        self.assertEqual(mask.shape[2], 3)

    def test_edge_width_analysis(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[30:35, :] = 255  # Horizontal line

        score, edge_vis, mask_vis = Analysis_Defocus.edge_width_analysis(gray, "Sobel")

        self.assertIsInstance(score, (float, np.floating))


class TestEntropyComputation(unittest.TestCase):
    """Test entropy calculation."""

    def test_compute_entropy_uniform(self):
        # Uniform block = low entropy
        block = np.ones((32, 32), dtype=np.uint8) * 128
        entropy = Analysis_Entropy.compute_entropy(block)
        self.assertLess(entropy, 1.0)

    def test_compute_entropy_random(self):
        # Random block = high entropy
        np.random.seed(42)
        block = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        entropy = Analysis_Entropy.compute_entropy(block)
        self.assertGreater(entropy, 5.0)


class TestAnalysisExecution(unittest.TestCase):
    """Test that execute runs with batch tensors."""

    def setUp(self):
        # Create a dummy batch tensor (B, H, W, C)
        # B=2 to test batch loop
        self.batch_image = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
        self.batch_image[0, 20:40, 20:40, :] = 1.0  # Image 1: Bright square
        self.batch_image[1, :, :, :] = 0.5          # Image 2: Uniform gray

    def test_blur_detection_execute(self):
        result = Analysis_BlurDetection.execute(self.batch_image, 16, False, "mean")
        self.assertIsNotNone(result)
        # Check result structure
        self.assertIn("ui", result)
        self.assertIn("result", result)
        # Check output tuple size
        self.assertEqual(len(result["result"]), 4) # score, map, text, raw
        # Check batch map size
        self.assertEqual(result["result"][1].shape[0], 2)

    def test_defocus_analysis_execute(self):
        result = Analysis_Defocus.execute(
            self.batch_image, "FFT Ratio (Sum)", True, "Sobel", "mean"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["result"][2].shape[0], 2) # heatmaps

    def test_edge_density_execute(self):
         result = Analysis_EdgeDensity.execute(
             self.batch_image, "Canny", 32, True, "mean"
         )
         self.assertIsNotNone(result)
         self.assertEqual(result["result"][1].shape[0], 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
