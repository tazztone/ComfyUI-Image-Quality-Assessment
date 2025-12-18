import unittest
import sys
import os
import torch
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
parent_of_repo = os.path.dirname(repo_root)
repo_name = os.path.basename(repo_root)

sys.path.insert(0, parent_of_repo)

try:
    opencv_nodes = __import__(f"{repo_name}.opencv_nodes", fromlist=["IQA_Blur_Estimation", "IQA_Brightness_Contrast"])
    IQA_Blur_Estimation = opencv_nodes.IQA_Blur_Estimation
    IQA_Brightness_Contrast = opencv_nodes.IQA_Brightness_Contrast
except ImportError:
    sys.path.insert(0, repo_root)
    from opencv_nodes import IQA_Blur_Estimation, IQA_Brightness_Contrast

class TestOpencvNodes(unittest.TestCase):
    def test_blur_estimation(self):
        # Create a sharp image (checkerboard)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::10, :] = 255
        img[:, ::10] = 255
        # Convert to tensor [1, H, W, C] normalized
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        res = IQA_Blur_Estimation.execute(tensor, "laplacian", "mean")
        # Output: (score, text, map, raw_scores)
        score = res['result'][0]
        self.assertGreater(score, 0.0)

        # Blur the image and expect lower score
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        tensor_blur = torch.from_numpy(img_blur).unsqueeze(0).float() / 255.0

        res_blur = IQA_Blur_Estimation.execute(tensor_blur, "laplacian", "mean")
        score_blur = res_blur['result'][0]

        self.assertLess(score_blur, score)

    def test_brightness_contrast(self):
        # Black image
        img = torch.zeros(1, 10, 10, 3)
        res = IQA_Brightness_Contrast.execute(img, "brightness", "mean")
        score = res['result'][0]
        self.assertAlmostEqual(score, 0.0)

        # White image
        img = torch.ones(1, 10, 10, 3)
        res = IQA_Brightness_Contrast.execute(img, "brightness", "mean")
        score = res['result'][0]
        self.assertAlmostEqual(score, 255.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()
