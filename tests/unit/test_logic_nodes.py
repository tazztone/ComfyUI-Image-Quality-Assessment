"""
Unit tests for logic nodes: Score Normalizer, Ensemble, Ranker, Filter.
Uses custom import logic to verify nodes with relative imports without loading the full package.
"""

import pytest
import sys
import importlib.util
from pathlib import Path

# ===================================================================================
# IMPORT HELPER: Load nodes that use relative imports
# ===================================================================================
# We create a dummy package 'iqa_test_pkg' and load the node files into it.
# This allows 'from .comfy_compat' and 'from .iqa_core' to resolve correctly.

TEST_PKG_NAME = "iqa_test_pkg"
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


# 1. Create the dummy package
if TEST_PKG_NAME not in sys.modules:
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(TEST_PKG_NAME, None, is_package=True)
    )
    sys.modules[TEST_PKG_NAME] = pkg

# 2. Load dependencies
# score_normalizer needs 'iqa_core' and 'comfy_compat'
comfy_compat = load_module_into_package("comfy_compat.py", "comfy_compat")
iqa_core = load_module_into_package("iqa_core.py", "iqa_core")

# 3. Load the node modules
# Note: logic_nodes imports torch, so torch must be installed.
logic_nodes = load_module_into_package("logic_nodes.py", "logic_nodes")
score_normalizer = load_module_into_package("score_normalizer.py", "score_normalizer")

# 4. Extract Classes
IQA_ScoreNormalizer = score_normalizer.IQA_ScoreNormalizer
PyIQA_EnsembleNode = logic_nodes.PyIQA_EnsembleNode
IQA_ThresholdFilter = logic_nodes.IQA_ThresholdFilter
IQA_BatchRanker = logic_nodes.IQA_BatchRanker

# ===================================================================================
# TESTS
# ===================================================================================


@pytest.mark.unit
class TestScoreNormalizer:
    def test_normalize_basic(self):
        """Test simple rescaling 0-1 to 0-100."""
        res = IQA_ScoreNormalizer.execute(
            score=0.5,
            input_min=0.0,
            input_max=1.0,
            output_min=0.0,
            output_max=100.0,
            invert=False,
            clamp=True,
        )["result"]
        # Result is a tuple: (value, text, list)
        val = res[0]
        assert val == pytest.approx(50.0)

    def test_normalize_list(self):
        """Test batch normalization."""
        scores = [0.0, 0.5, 1.0]
        res = IQA_ScoreNormalizer.execute(
            score=scores,
            input_min=0.0,
            input_max=1.0,
            output_min=0.0,
            output_max=10.0,
            invert=False,
            clamp=True,
        )["result"]

        # If input is list, output[0] is list
        val = res[0]
        assert val == pytest.approx([0.0, 5.0, 10.0])

    def test_invert(self):
        """Test inversion (lower is better -> higher is better)."""
        # input 0.2 (good quality in LPIPS) -> should become 0.8 -> 80
        res = IQA_ScoreNormalizer.execute(
            score=0.2,  # Good score
            input_min=0.0,
            input_max=1.0,
            output_min=0.0,
            output_max=100.0,
            invert=True,
            clamp=True,
        )["result"]

        val = res[0]
        assert pytest.approx(val) == 80.0

    def test_clamp(self):
        """Test clamping values outside range."""
        res = IQA_ScoreNormalizer.execute(
            score=1.5,  # Above max
            input_min=0.0,
            input_max=1.0,
            output_min=0.0,
            output_max=100.0,
            invert=False,
            clamp=True,
        )["result"]

        val = res[0]
        assert val == 100.0


@pytest.mark.unit
class TestEnsembleNode:
    def test_weighted_average(self):
        """Test basic weighted average."""
        # Score 1: 100 (wt 1)
        # Score 2: 50 (wt 1)
        # Avg: 75
        res = PyIQA_EnsembleNode.execute(
            score_1=100.0,
            weight_1=1.0,
            score_2=50.0,
            weight_2=1.0,
            score_3=0.0,
            weight_3=0.0,
            score_4=0.0,
            weight_4=0.0,
        )
        val = res[0]  # output is (value, text)
        assert val == pytest.approx(75.0)

    def test_uneven_weights(self):
        """Test unequal weights."""
        # S1: 100 (wt 3) = 300
        # S2: 0 (wt 1) = 0
        # Total: 300 / 4 = 75
        res = PyIQA_EnsembleNode.execute(
            score_1=100.0,
            weight_1=3.0,
            score_2=0.0,
            weight_2=1.0,
            score_3=0.0,
            weight_3=0.0,
            score_4=0.0,
            weight_4=0.0,
        )
        assert res[0] == pytest.approx(75.0)

    def test_batch_ensemble(self):
        """Test batch processing for ensemble."""
        # Batch of 2
        s1 = [10.0, 20.0]
        s2 = [30.0, 40.0]

        res = PyIQA_EnsembleNode.execute(
            score_1=s1,
            weight_1=1.0,
            score_2=s2,
            weight_2=1.0,
            score_3=0.0,
            weight_3=0.0,
            score_4=0.0,
            weight_4=0.0,
        )

        val = res[0]
        assert val == [20.0, 30.0]


@pytest.mark.unit
class TestBatchRanker:
    def test_sort_descending(self):
        """Test sorting highest to lowest."""
        import torch

        # 3 images, scores: 10, 30, 20
        # Expected order: img2, img3, img1
        images = torch.zeros((3, 64, 64, 3))
        # Mark images so we can identify them
        images[0, 0, 0, 0] = 1.0  # ID 1
        images[1, 0, 0, 0] = 2.0  # ID 2
        images[2, 0, 0, 0] = 3.0  # ID 3

        scores = [10.0, 30.0, 20.0]

        res = IQA_BatchRanker.execute(
            images=images, scores=scores, order="descending", take_top_n=0
        )

        out_imgs = res[0]
        out_scores = res[1]

        # Check scores
        assert out_scores == [30.0, 20.0, 10.0]

        # Check images (by identifying marker)
        assert out_imgs[0, 0, 0, 0] == 2.0  # Original index 1 (score 30)
        assert out_imgs[1, 0, 0, 0] == 3.0  # Original index 2 (score 20)
        assert out_imgs[2, 0, 0, 0] == 1.0  # Original index 0 (score 10)

    def test_take_top_n(self):
        """Test keeping only top N images."""
        import torch

        images = torch.zeros((3, 10, 10, 3))
        scores = [10.0, 30.0, 20.0]

        res = IQA_BatchRanker.execute(
            images=images, scores=scores, order="descending", take_top_n=1
        )

        out_imgs = res[0]
        assert len(out_imgs) == 1
        assert res[1] == [30.0]


@pytest.mark.unit
class TestThresholdFilter:
    def test_filter_greater(self):
        """Test passscore > threshold."""
        import torch

        # 2 images, scores: 0.8, 0.4
        # Threshold 0.5, greater
        # Pass: img1
        # Fail: img2
        images = torch.zeros((2, 10, 10, 3))
        images[0, 0, 0, 0] = 1.0  # Pass
        images[1, 0, 0, 0] = 0.0  # Fail

        scores = [0.8, 0.4]

        res = IQA_ThresholdFilter.execute(
            images=images, scores=scores, threshold=0.5, operation="greater"
        )

        passed = res[0]
        failed = res[2]

        assert len(passed) == 1
        assert len(failed) == 1
        assert passed[0, 0, 0, 0] == 1.0

    def test_filter_less(self):
        """Test pass score < threshold."""
        import torch

        # Scores: 0.2, 0.8
        # Threshold 0.5, less
        # Pass: 0.2
        images = torch.zeros((2, 10, 10, 3))
        scores = [0.2, 0.8]

        res = IQA_ThresholdFilter.execute(
            images=images, scores=scores, threshold=0.5, operation="less"
        )

        passed_scores = res[1]
        assert passed_scores == [0.2]
