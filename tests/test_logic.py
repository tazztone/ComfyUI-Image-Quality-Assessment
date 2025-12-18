import unittest
import sys
import os
import torch
import numpy as np

# Setup path to import from package root
# We need to treat the current repo as a package to support relative imports inside it.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
parent_of_repo = os.path.dirname(repo_root)
repo_name = os.path.basename(repo_root)

# Insert parent directory so we can import 'repo_name'
sys.path.insert(0, parent_of_repo)

# Dynamic imports
try:
    IQA_ScoreNormalizer = __import__(f"{repo_name}.score_normalizer", fromlist=["IQA_ScoreNormalizer"]).IQA_ScoreNormalizer
    logic_nodes = __import__(f"{repo_name}.logic_nodes", fromlist=["PyIQA_EnsembleNode", "IQA_ThresholdFilter", "IQA_BatchRanker"])
    PyIQA_EnsembleNode = logic_nodes.PyIQA_EnsembleNode
    IQA_ThresholdFilter = logic_nodes.IQA_ThresholdFilter
    IQA_BatchRanker = logic_nodes.IQA_BatchRanker
    IQA_HeatmapVisualizer = __import__(f"{repo_name}.visualization_nodes", fromlist=["IQA_HeatmapVisualizer"]).IQA_HeatmapVisualizer
except ImportError:
    # Fallback if folder name is different or simple path insertion works better
    sys.path.insert(0, repo_root)
    from score_normalizer import IQA_ScoreNormalizer
    from logic_nodes import PyIQA_EnsembleNode, IQA_ThresholdFilter, IQA_BatchRanker
    from visualization_nodes import IQA_HeatmapVisualizer


class TestIQAScoreNormalizer(unittest.TestCase):
    def test_normalize_basic(self):
        # 0.5 in 0-1 range -> 50 in 0-100 range
        res = IQA_ScoreNormalizer.execute(0.5, 0.0, 1.0, 0.0, 100.0, False, True)
        # res is dict: {'ui': ..., 'result': NodeOutput(...)}
        # NodeOutput is tuple: (normalized_val, text, raw_list)
        self.assertAlmostEqual(res['result'][0], 50.0)

    def test_normalize_list(self):
        res = IQA_ScoreNormalizer.execute([0.0, 0.5, 1.0], 0.0, 1.0, 0.0, 100.0, False, True)
        # raw_scores is at index 2
        self.assertEqual(len(res['result'][2]), 3)
        self.assertAlmostEqual(res['result'][2][0], 0.0)
        self.assertAlmostEqual(res['result'][2][1], 50.0)
        self.assertAlmostEqual(res['result'][2][2], 100.0)

    def test_invert(self):
        # 0.2 -> normalized 0.2 -> invert 0.8 -> 80
        res = IQA_ScoreNormalizer.execute(0.2, 0.0, 1.0, 0.0, 100.0, True, True)
        self.assertAlmostEqual(res['result'][0], 80.0)

    def test_clamping(self):
        # 1.5 -> normalized 1.5 -> clamp to 1.0 -> 100
        res = IQA_ScoreNormalizer.execute(1.5, 0.0, 1.0, 0.0, 100.0, False, True)
        self.assertAlmostEqual(res['result'][0], 100.0)

        # Without clamping: 1.5 -> 150
        res = IQA_ScoreNormalizer.execute(1.5, 0.0, 1.0, 0.0, 100.0, False, False)
        self.assertAlmostEqual(res['result'][0], 150.0)

class TestPyIQAEnsembleNode(unittest.TestCase):
    def test_ensemble_basic(self):
        # (10*1 + 20*1) / 2 = 15
        # 8 args: s1, w1, s2, w2, s3, w3, s4, w4
        res = PyIQA_EnsembleNode.execute(10.0, 1.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        # returns NodeOutput tuple: (val, text)
        self.assertAlmostEqual(res[0], 15.0)

    def test_ensemble_weights(self):
        # (10*2 + 20*1) / 3 = 40/3 = 13.333
        res = PyIQA_EnsembleNode.execute(10.0, 2.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(res[0], 13.3333333)

    def test_ensemble_lists(self):
        # s1 = [10, 10], s2 = [20, 40]
        # w1=1, w2=1
        # i0: (10+20)/2 = 15
        # i1: (10+40)/2 = 25
        res = PyIQA_EnsembleNode.execute([10.0, 10.0], 1.0, [20.0, 40.0], 1.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(len(res[0]), 2)
        self.assertAlmostEqual(res[0][0], 15.0)
        self.assertAlmostEqual(res[0][1], 25.0)

class TestIQAThresholdFilter(unittest.TestCase):
    def test_filter_greater(self):
        # 4 images, 10x10 RGB
        images = torch.rand(4, 10, 10, 3)
        scores = [0.1, 0.4, 0.6, 0.9]
        threshold = 0.5

        # Expect indices 2 and 3 to pass (0.6, 0.9 > 0.5)
        # args: images, scores, threshold, operation
        res = IQA_ThresholdFilter.execute(images, scores, threshold, "greater")
        # returns NodeOutput(p_imgs, p_sc, f_imgs, f_sc)
        passed_imgs = res[0]
        passed_scores = res[1]
        failed_imgs = res[2]

        self.assertEqual(len(passed_scores), 2)
        self.assertAlmostEqual(passed_scores[0], 0.6)
        self.assertAlmostEqual(passed_scores[1], 0.9)
        self.assertEqual(passed_imgs.shape[0], 2)
        self.assertEqual(failed_imgs.shape[0], 2)

    def test_filter_less(self):
        images = torch.rand(4, 10, 10, 3)
        scores = [0.1, 0.4, 0.6, 0.9]
        threshold = 0.5

        # Expect indices 0 and 1 to pass (0.1, 0.4 < 0.5)
        res = IQA_ThresholdFilter.execute(images, scores, threshold, "less")
        passed_scores = res[1]
        self.assertEqual(len(passed_scores), 2)
        self.assertAlmostEqual(passed_scores[0], 0.1)

class TestIQABatchRanker(unittest.TestCase):
    def test_rank_descending(self):
        # 3 images
        images = torch.zeros(3, 10, 10, 3)
        # Mark images so we can identify them
        images[0, 0, 0, 0] = 1.0 # Index 0
        images[1, 0, 0, 0] = 2.0 # Index 1
        images[2, 0, 0, 0] = 3.0 # Index 2

        scores = [10.0, 30.0, 20.0]
        # Expected order: 30 (idx 1), 20 (idx 2), 10 (idx 0)

        # args: images, scores, order, take_top_n
        res = IQA_BatchRanker.execute(images, scores, "descending", 0)
        sorted_images = res[0]
        sorted_scores = res[1]

        self.assertEqual(sorted_scores, [30.0, 20.0, 10.0])
        self.assertEqual(sorted_images[0, 0, 0, 0].item(), 2.0)
        self.assertEqual(sorted_images[1, 0, 0, 0].item(), 3.0)
        self.assertEqual(sorted_images[2, 0, 0, 0].item(), 1.0)

    def test_rank_take_top_n(self):
        images = torch.zeros(3, 10, 10, 3)
        scores = [10.0, 30.0, 20.0]
        # Top 1: 30

        res = IQA_BatchRanker.execute(images, scores, "descending", 1)
        sorted_scores = res[1]
        self.assertEqual(len(sorted_scores), 1)
        self.assertEqual(sorted_scores[0], 30.0)

class TestIQAHeatmapVisualizer(unittest.TestCase):
    def test_heatmap_basic(self):
        # 1 image, 10x10, random
        image = torch.rand(1, 10, 10, 3)
        # execute(image, colormap, normalize_min, normalize_max, score_map_optional)
        res = IQA_HeatmapVisualizer.execute(image, "JET", 0.0, 1.0, None)
        output = res[0]
        self.assertEqual(output.shape, (1, 10, 10, 3))
        # Ensure it returns valid tensor
        self.assertTrue(torch.is_tensor(output))

if __name__ == "__main__":
    unittest.main()
