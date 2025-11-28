import cv2
import numpy as np
import torch

class IQA_Blur_Estimation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("score", "score_text", "blur_map")
    FUNCTION = "analyze"
    CATEGORY = "IQA/OpenCV"

    def analyze(self, image, aggregation):
        scores = []
        maps = []

        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Variance of Laplacian
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            score = laplacian.var()
            scores.append(score)

            # Create visualization map
            # Normalize laplacian absolute values to 0-1 range for visualization
            lap_abs = np.absolute(laplacian)
            lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
            lap_norm = lap_norm.astype(np.uint8)

            # Apply heatmap (JET is common for magnitude)
            heatmap = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Convert back to tensor [H, W, C] -> [1, H, W, C]
            map_tensor = torch.from_numpy(heatmap.astype(np.float32) / 255.0)
            maps.append(map_tensor)

        final_score = self._aggregate(scores, aggregation)
        score_text = f"Blur (Laplacian) {aggregation}: {final_score:.2f}"

        # Stack maps
        if maps:
            maps_out = torch.stack(maps)
        else:
            maps_out = torch.zeros_like(image)

        return {
            "ui": {"text": [score_text]},
            "result": (final_score, score_text, maps_out)
        }

    def _aggregate(self, scores, method):
        if not scores: return 0.0
        if method == "mean": return float(np.mean(scores))
        if method == "min": return float(np.min(scores))
        if method == "max": return float(np.max(scores))
        if method == "first": return float(scores[0])
        return float(np.mean(scores))

class IQA_Brightness_Contrast:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["brightness", "contrast"],),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "analyze"
    CATEGORY = "IQA/OpenCV"

    def analyze(self, image, mode, aggregation):
        scores = []
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            if mode == "brightness":
                score = np.mean(img_gray)
            else: # contrast
                score = img_gray.std()

            scores.append(score)

        final_score = self._aggregate(scores, aggregation)
        score_text = f"{mode.capitalize()} ({aggregation}): {final_score:.2f}"

        return {
            "ui": {"text": [score_text]},
            "result": (final_score, score_text)
        }

    def _aggregate(self, scores, method):
        if not scores: return 0.0
        if method == "mean": return float(np.mean(scores))
        if method == "min": return float(np.min(scores))
        if method == "max": return float(np.max(scores))
        if method == "first": return float(scores[0])
        return float(np.mean(scores))

class IQA_Colorfulness:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "analyze"
    CATEGORY = "IQA/OpenCV"

    def analyze(self, image, aggregation):
        scores = []
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            B, G, R = cv2.split(img_bgr.astype("float"))
            rg = np.absolute(R - G)
            yb = np.absolute(0.5 * (R + G) - B)
            (rbMean, rbStd) = (np.mean(rg), np.std(rg))
            (ybMean, ybStd) = (np.mean(yb), np.std(yb))
            stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
            meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
            score = stdRoot + (0.3 * meanRoot)

            scores.append(score)

        final_score = self._aggregate(scores, aggregation)
        score_text = f"Colorfulness ({aggregation}): {final_score:.2f}"

        return {
            "ui": {"text": [score_text]},
            "result": (final_score, score_text)
        }

    def _aggregate(self, scores, method):
        if not scores: return 0.0
        if method == "mean": return float(np.mean(scores))
        if method == "min": return float(np.min(scores))
        if method == "max": return float(np.max(scores))
        if method == "first": return float(scores[0])
        return float(np.mean(scores))
