"""
Standalone unit tests for ComfyUI-IQA-Node.
These tests do not require the ComfyUI environment or venv.
They test the core logic and utility functions in isolation.

Run with: python -m pytest tests/ -v
Or simply: python tests/test_unit.py
"""
import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAggregateScores(unittest.TestCase):
    """Tests for the aggregate_scores function in iqa_core."""

    def setUp(self):
        # Import here to avoid import errors if dependencies missing
        from iqa_core import aggregate_scores
        self.aggregate = aggregate_scores

    def test_mean_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.aggregate(scores, "mean")
        self.assertAlmostEqual(result, 3.0)

    def test_min_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.aggregate(scores, "min")
        self.assertAlmostEqual(result, 1.0)

    def test_max_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.aggregate(scores, "max")
        self.assertAlmostEqual(result, 5.0)

    def test_median_aggregation(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.aggregate(scores, "median")
        self.assertAlmostEqual(result, 3.0)

    def test_median_even_count(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        result = self.aggregate(scores, "median")
        self.assertAlmostEqual(result, 2.5)

    def test_first_aggregation(self):
        scores = [1.0, 2.0, 3.0]
        result = self.aggregate(scores, "first")
        self.assertAlmostEqual(result, 1.0)

    def test_empty_scores(self):
        result = self.aggregate([], "mean")
        self.assertEqual(result, 0.0)

    def test_single_value(self):
        result = self.aggregate([5.5], "mean")
        self.assertAlmostEqual(result, 5.5)

    def test_nan_filtering(self):
        scores = [1.0, float('nan'), 3.0]
        result = self.aggregate(scores, "mean")
        self.assertAlmostEqual(result, 2.0)

    def test_default_method(self):
        scores = [1.0, 2.0, 3.0]
        result = self.aggregate(scores, "invalid")
        self.assertAlmostEqual(result, 2.0)  # Defaults to mean


class TestTensorToNumpy(unittest.TestCase):
    """Tests for the tensor_to_numpy function in iqa_core."""

    def setUp(self):
        try:
            import torch
            from iqa_core import tensor_to_numpy
            self.torch = torch
            self.tensor_to_numpy = tensor_to_numpy
            self.skip = False
        except ImportError:
            self.skip = True

    def test_basic_conversion(self):
        if self.skip:
            self.skipTest("torch not available")

        # Create a simple tensor [B, H, W, C]
        tensor = self.torch.rand(2, 64, 64, 3)
        result = self.tensor_to_numpy(tensor)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (64, 64, 3))
        self.assertEqual(result[0].dtype, np.uint8)

    def test_value_range(self):
        if self.skip:
            self.skipTest("torch not available")

        # Create a tensor with known values
        tensor = self.torch.ones(1, 10, 10, 3) * 0.5
        result = self.tensor_to_numpy(tensor)

        # 0.5 * 255 = 127.5 -> 127 or 128
        self.assertTrue(np.all(result[0] >= 127))
        self.assertTrue(np.all(result[0] <= 128))


class TestGetHash(unittest.TestCase):
    """Tests for the get_hash function in iqa_core."""

    def setUp(self):
        from iqa_core import get_hash
        self.get_hash = get_hash

    def test_string_hash(self):
        result = self.get_hash("test_string")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 64)  # SHA256 hex digest

    def test_int_hash(self):
        result = self.get_hash(42)
        self.assertIsNotNone(result)

    def test_dict_hash(self):
        result = self.get_hash({"key": "value", "num": 123})
        self.assertIsNotNone(result)

    def test_deterministic(self):
        hash1 = self.get_hash({"a": 1, "b": 2})
        hash2 = self.get_hash({"b": 2, "a": 1})  # Same dict, different order
        self.assertEqual(hash1, hash2)


class TestModelCache(unittest.TestCase):
    """Tests for the ModelCache class in iqa_core."""

    def setUp(self):
        from iqa_core import ModelCache
        self.ModelCache = ModelCache

    def test_basic_cache(self):
        cache = self.ModelCache(max_size=3)

        cache.put("key1", "model1")
        cache.put("key2", "model2")

        self.assertEqual(cache.get("key1"), "model1")
        self.assertEqual(cache.get("key2"), "model2")

    def test_cache_miss(self):
        cache = self.ModelCache(max_size=3)
        self.assertIsNone(cache.get("nonexistent"))

    def test_lru_eviction(self):
        cache = self.ModelCache(max_size=2)

        cache.put("key1", "model1")
        cache.put("key2", "model2")
        cache.put("key3", "model3")  # This should evict key1

        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "model2")
        self.assertEqual(cache.get("key3"), "model3")

    def test_access_updates_lru(self):
        cache = self.ModelCache(max_size=2)

        cache.put("key1", "model1")
        cache.put("key2", "model2")
        cache.get("key1")  # Access key1, making key2 LRU
        cache.put("key3", "model3")  # Should evict key2

        self.assertEqual(cache.get("key1"), "model1")
        self.assertIsNone(cache.get("key2"))
        self.assertEqual(cache.get("key3"), "model3")


class TestScoreNormalizerLogic(unittest.TestCase):
    """Tests for score normalization logic."""

    def test_basic_normalization(self):
        # Simulate the normalization logic from score_normalizer.py
        score = 0.5
        input_min, input_max = 0.0, 1.0
        output_min, output_max = 0.0, 100.0

        normalized = (score - input_min) / (input_max - input_min)
        output = output_min + normalized * (output_max - output_min)

        self.assertAlmostEqual(output, 50.0)

    def test_inversion(self):
        score = 0.8
        input_min, input_max = 0.0, 1.0
        output_min, output_max = 0.0, 100.0

        normalized = (score - input_min) / (input_max - input_min)
        normalized = 1.0 - normalized  # Invert
        output = output_min + normalized * (output_max - output_min)

        self.assertAlmostEqual(output, 20.0)

    def test_clamping(self):
        score = 1.5  # Outside input range
        input_min, input_max = 0.0, 1.0
        output_min, output_max = 0.0, 100.0

        normalized = (score - input_min) / (input_max - input_min)
        output = output_min + normalized * (output_max - output_min)
        output = max(output_min, min(output_max, output))  # Clamp

        self.assertAlmostEqual(output, 100.0)


class TestColorfulnessMetric(unittest.TestCase):
    """Tests for colorfulness calculation (Hasler & Suesstrunk method)."""

    def test_grayscale_image(self):
        # A grayscale image should have low colorfulness
        # Simulating a grayscale image where R=G=B
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

        self.assertAlmostEqual(colorfulness, 0.0)

    def test_colorful_image(self):
        # An image with varied colors should have higher colorfulness
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

        self.assertGreater(colorfulness, 50)  # Should be significantly colorful


if __name__ == "__main__":
    unittest.main()
