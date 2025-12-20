"""
Unit tests for metrics and tensor conversions.
Tests utils/iqa_core.py without requiring ComfyUI server.
"""

import pytest
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Load the iqa_core module directly using importlib to bypass package __init__.py
iqa_core_path = custom_node_root / "utils" / "iqa_core.py"
spec = importlib.util.spec_from_file_location("iqa_core_module", iqa_core_path)
iqa_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iqa_core)

# Import what we need from the loaded module
tensor_to_numpy = iqa_core.tensor_to_numpy


@pytest.mark.unit
class TestMetrics:
    """Tests for computational metrics and tensor conversions."""

    def test_tensor_to_numpy_conversion(self):
        """Test conversion of ComfyUI tensor to numpy."""
        import torch
        sample_tensor = torch.rand(1, 64, 64, 3)
        result = tensor_to_numpy(sample_tensor)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)
        assert result[0].dtype == np.uint8

    def test_colorfulness_logic(self):
        """Standalone test for colorfulness logic (Hasler & Suesstrunk)."""
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
