import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iqa_core import tensor_to_numpy

@pytest.mark.unit
class TestMetrics:
    """Tests for computational metrics and tensor conversions."""

    def test_tensor_to_numpy_conversion(self, sample_image_tensor):
        """Test conversion of ComfyUI tensor to numpy using fixture"""
        result = tensor_to_numpy(sample_image_tensor)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)
        assert result[0].dtype == np.uint8

    def test_colorfulness_logic(self):
        """Standalone test for colorfulness logic (Hasler & Suesstrunk)"""
        # Grayscale image
        R = np.full((100, 100), 128, dtype=float)
        G = np.full((100, 100), 128, dtype=float)
        B = np.full((100, 100), 128, dtype=float)

        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = stdRoot + (0.3 * meanRoot)
        assert colorfulness == 0.0

        # Varied color image
        np.random.seed(42)
        R = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(float)
        G = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(float)
        B = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(float)

        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = stdRoot + (0.3 * meanRoot)
        assert colorfulness > 50
