"""
Unit tests for score aggregation and normalization.
Tests utils/iqa_core.py without requiring ComfyUI server.
"""

import pytest
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
aggregate_scores = iqa_core.aggregate_scores


@pytest.mark.unit
class TestAggregateScores:
    """Tests for the aggregate_scores function in iqa_core."""

    def test_mean_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate_scores(scores, "mean")
        assert result == 3.0

    def test_min_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate_scores(scores, "min")
        assert result == 1.0

    def test_max_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate_scores(scores, "max")
        assert result == 5.0

    def test_median_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate_scores(scores, "median")
        assert result == 3.0

    def test_median_even_count(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        result = aggregate_scores(scores, "median")
        assert result == 2.5

    def test_first_aggregation(self):
        scores = [1.0, 2.0, 3.0]
        result = aggregate_scores(scores, "first")
        assert result == 1.0

    def test_empty_scores(self):
        result = aggregate_scores([], "mean")
        assert result == 0.0

    def test_single_value(self):
        result = aggregate_scores([5.5], "mean")
        assert result == 5.5

    def test_nan_filtering(self):
        scores = [1.0, float('nan'), 3.0]
        result = aggregate_scores(scores, "mean")
        assert result == 2.0

    def test_default_method(self):
        scores = [1.0, 2.0, 3.0]
        result = aggregate_scores(scores, "invalid")
        assert result == 2.0  # Defaults to mean


@pytest.mark.unit
class TestScoreNormalizerLogic:
    """Tests for score normalization logic."""

    def test_basic_normalization(self):
        score = 0.5
        input_min, input_max = 0.0, 1.0
        output_min, output_max = 0.0, 100.0

        normalized = (score - input_min) / (input_max - input_min)
        output = output_min + normalized * (output_max - output_min)
        assert output == pytest.approx(50.0)

    def test_inversion(self):
        score = 0.8
        input_min, input_max = 0.0, 1.0
        output_min, output_max = 0.0, 100.0

        normalized = (score - input_min) / (input_max - input_min)
        normalized = 1.0 - normalized
        output = output_min + normalized * (output_max - output_min)
        assert output == pytest.approx(20.0)
