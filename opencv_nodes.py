import cv2
import numpy as np
import torch

class OpenCV_IQA_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": ([
                    "laplacian_blur_score",
                    "brightness_mean",
                    "contrast_rms",
                    "colorfulness"
                ],),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "analyze"
    CATEGORY = "IQA"

    def analyze(self, image, method, aggregation):
        # image is [B, H, W, C]
        scores = []

        # Loop over batch
        for i in range(image.shape[0]):
            img_tensor = image[i] # [H, W, C]

            # Convert to numpy (0-255, uint8)
            img_np = 255. * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            score = 0.0

            if method == "laplacian_blur_score":
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(img_gray, cv2.CV_64F).var()

            elif method == "brightness_mean":
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                score = np.mean(img_gray)

            elif method == "contrast_rms":
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                score = img_gray.std()

            elif method == "colorfulness":
                # Metric by Hasler and Suesstrunk
                B, G, R = cv2.split(img_bgr.astype("float"))
                rg = np.absolute(R - G)
                yb = np.absolute(0.5 * (R + G) - B)
                (rbMean, rbStd) = (np.mean(rg), np.std(rg))
                (ybMean, ybStd) = (np.mean(yb), np.std(yb))
                stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
                meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
                score = stdRoot + (0.3 * meanRoot)

            scores.append(score)

        # Aggregation
        if aggregation == "mean":
            final_score = float(np.mean(scores))
        elif aggregation == "min":
            final_score = float(np.min(scores))
        elif aggregation == "max":
            final_score = float(np.max(scores))
        elif aggregation == "first":
            final_score = float(scores[0])
        else:
            final_score = float(np.mean(scores))

        text_output = f"{method} ({aggregation}): {final_score:.4f}"

        return (final_score, text_output)
