"""
Unit tests for analysis nodes.
Verifies logic for Blur Detection, Color Temperature, etc.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# ===================================================================================
# IMPORT HELPER
# ===================================================================================
TEST_PKG_NAME = "iqa_test_pkg_analysis"
CUSTOM_NODE_ROOT = Path(__file__).parent.parent.parent


def load_module_into_package(filename, submodule_name):
    """Load a file as a submodule of our dummy package."""
    full_pkg_name = f"{TEST_PKG_NAME}.{submodule_name}"
    filepath = CUSTOM_NODE_ROOT / filename

    spec = importlib.util.spec_from_file_location(full_pkg_name, filepath)
    if spec is None:
        raise ImportError(f"Could not load {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_pkg_name] = module
    spec.loader.exec_module(module)
    return module


# ===================================================================================
# SETUP
# ===================================================================================

# 1. Create dummy package
if TEST_PKG_NAME not in sys.modules:
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(TEST_PKG_NAME, None, is_package=True)
    )
    sys.modules[TEST_PKG_NAME] = pkg

# 2. Load dependencies
comfy_compat = load_module_into_package("comfy_compat.py", "comfy_compat")
iqa_core = load_module_into_package("iqa_core.py", "iqa_core")

# 3. Load analysis_nodes
# analysis_nodes imports cv2, plt, etc.
# We assume these are present in the env.
try:
    analysis_nodes = load_module_into_package("analysis_nodes.py", "analysis_nodes")
except ImportError as e:
    pytest.fail(f"Could not import analysis_nodes dependencies: {e}")

Analysis_BlurDetection = analysis_nodes.Analysis_BlurDetection
Analysis_ColorTemperature = analysis_nodes.Analysis_ColorTemperature


@pytest.mark.unit
class TestBlurDetection:
    def test_interpret_blur(self):
        """Test interpretation logic."""
        assert "Very blurry" in Analysis_BlurDetection.interpret_blur(10.0)
        assert "Slightly blurry" in Analysis_BlurDetection.interpret_blur(100.0)
        assert "Very sharp" in Analysis_BlurDetection.interpret_blur(500.0)

    def test_execute_solid_color(self):
        """Solid color should be 0 blur (variance 0) -> Very blurry."""
        # Create a solid gray image (64x64)
        # Setup input tensor: (1, 64, 64, 3) range 0-1
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5

        # Don't visualize to save time/dependencies
        with patch.object(analysis_nodes.plt, "subplots", autospec=True):
            # We must mock subplots because code calls it if visualize=True
            # But set visualize_blur_map = False
            res = Analysis_BlurDetection.execute(
                image=img, block_size=32, visualize_blur_map=False, aggregation="mean"
            )

            # Score should be 0.0 because variance of flat area is 0
            final_score = res["result"][0]
            assert final_score == pytest.approx(0.0)
            assert "Very blurry" in res["result"][2]

    def test_execute_noise(self):
        """Noise should have high variance -> sharper."""
        np.random.seed(42)
        noise = np.random.rand(64, 64, 3).astype(np.float32)
        img_tensor = torch.from_numpy(noise)[None,]  # shape (1, 64, 64, 3)

        res = Analysis_BlurDetection.execute(
            image=img_tensor,
            block_size=16,  # smaller block to get local variance
            visualize_blur_map=False,
            aggregation="mean",
        )

        score = res["result"][0]
        # Random noise has high laplacian variance
        assert score > 10.0


@pytest.mark.unit
class TestColorTemperature:
    def test_estimate_warm(self):
        """Red image should be warm."""
        # Pure Red: 255, 0, 0
        img_uint8 = np.zeros((10, 10, 3), dtype=np.uint8)
        img_uint8[:, :, 0] = 255  # R

        kelvin, label, _ = Analysis_ColorTemperature._estimate_color_temperature(
            img_uint8
        )

        # Red is low Kelvin (approx 1000-2000K usually?)
        # Let's check assert
        assert kelvin < 3000
        assert label == "Warm"

    def test_estimate_cool(self):
        """Blue-ish white image should be cool."""
        # Cool White / Daylight: slightly more blue
        # R=200, G=220, B=255
        img_uint8 = np.zeros((10, 10, 3), dtype=np.uint8)
        img_uint8[:, :, 0] = 200  # R
        img_uint8[:, :, 1] = 220  # G
        img_uint8[:, :, 2] = 255  # B

        kelvin, label, _ = Analysis_ColorTemperature._estimate_color_temperature(
            img_uint8
        )

        # Should be relatively cool (> 5000K or labeled Cool)
        # Note: Simple CCT formulas vary, but this should definitely not be 'Warm' (<3000)
        assert kelvin > 5000

    def test_execute(self):
        """Test full execution flow."""
        # White image (Neutral-ish but high intensity)
        # White (255,255,255) is usually ~5500-6500K depending on reference
        img = torch.ones((1, 10, 10, 3), dtype=torch.float32)

        # Mock plt to avoid plotting overhead
        with patch.object(analysis_nodes.plt, "savefig"):
            with patch.object(
                analysis_nodes.plt, "subplots", return_value=(MagicMock(), MagicMock())
            ):
                # Mock cv2.imdecode to return a dummy image so we don't need real bytes from plt
                with patch("cv2.imdecode") as mock_decode:
                    mock_decode.return_value = np.zeros(
                        (10, 20, 3), dtype=np.uint8
                    )  # BGR

                    res = Analysis_ColorTemperature.execute(img, aggregation="mean")

                    kelvin = res["result"][0]
                    # Label is res["result"][1]

                    # 5503K is roughly D55/D65 white point logic in simple estimators
                    assert kelvin > 5000
