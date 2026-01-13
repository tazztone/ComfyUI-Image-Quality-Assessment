import cv2
import numpy as np
import torch

import skimage.restoration
from .iqa_core import aggregate_scores, tensor_to_numpy, get_hash, InferenceError
from .comfy_compat import io


class IQA_Blur_Estimation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_Blur_Estimation",
            display_name="IQA: Blur Estimation (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze for blur.\n\nHigher scores indicate sharper images.\nOutputs both a numeric score and a visual blur map.",
                ),
                io.Enum.Input(
                    "mode",
                    ["laplacian", "tenengrad"],
                    default="laplacian",
                    tooltip="Blur detection algorithm:\n\n• laplacian: Variance of Laplacian operator (recommended)\n  - Fast and reliable for general blur detection\n  - Higher variance = sharper image\n\n• tenengrad: Gradient magnitude using Sobel operator\n  - Better for detecting motion blur\n  - Measures average edge strength",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average of all scores (recommended)\n• min: Lowest score (blurriest image)\n• max: Highest score (sharpest image)\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Image.Output("blur_map"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, mode, aggregation, **kwargs):
        return get_hash([mode, aggregation])

    @classmethod
    def VALIDATE_INPUTS(cls, image, mode, aggregation, **kwargs):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation method: {aggregation}"
        if mode not in ["laplacian", "tenengrad"]:
            return f"Invalid mode: {mode}"
        return True

    @classmethod
    def execute(cls, image, mode, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)

            for img_np in img_list:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                score = 0.0
                heatmap = None

                if mode == "laplacian":
                    # Variance of Laplacian
                    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
                    score = laplacian.var()

                    # Map
                    lap_abs = np.absolute(laplacian)
                    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
                    lap_norm = lap_norm.astype(np.uint8)
                    heatmap = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)

                elif mode == "tenengrad":
                    # Gradient magnitude (Sobel)
                    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = np.sqrt(gx**2 + gy**2)
                    score = np.mean(magnitude)  # Average gradient magnitude

                    # Map
                    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    mag_norm = mag_norm.astype(np.uint8)
                    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

                scores.append(score)

                # Convert heatmap BGR -> RGB -> Tensor
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                map_tensor = torch.from_numpy(heatmap.astype(np.float32) / 255.0)
                maps.append(map_tensor)

        except Exception as e:
            raise InferenceError(f"Blur estimation failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"Blur ({mode.capitalize()}) {aggregation}: {final_score:.2f}"

        if maps:
            maps_out = torch.stack(maps)
        else:
            maps_out = torch.zeros_like(image)

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, maps_out, scores),
        }


class IQA_Brightness_Contrast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_Brightness_Contrast",
            display_name="IQA: Brightness & Contrast (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze.\n\nConverts to grayscale internally for brightness/contrast calculations.",
                ),
                io.Enum.Input(
                    "mode",
                    ["brightness", "contrast", "exposure_score"],
                    default="brightness",
                    tooltip="What to measure:\n\n• brightness: Average pixel intensity (0-255 scale)\n  - 0 = pure black, 255 = pure white\n  - Ideal range: 100-150 for most images\n\n• contrast: Standard deviation of pixel values\n  - Higher = more contrast\n  - Low values indicate flat/washed out images\n\n• exposure_score: Exposure quality (0-1 scale)\n  - Penalizes under/overexposed pixels\n  - 1.0 = well exposed, 0.0 = poorly exposed",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average of all scores (recommended)\n• min: Lowest score in batch\n• max: Highest score in batch\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, mode, aggregation, **kwargs):
        return get_hash([mode, aggregation])

    @classmethod
    def VALIDATE_INPUTS(cls, image, mode, aggregation, **kwargs):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation method: {aggregation}"
        return True

    @classmethod
    def execute(cls, image, mode, aggregation):
        scores = []
        try:
            img_list = tensor_to_numpy(image)

            for img_np in img_list:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                if mode == "brightness":
                    score = np.mean(img_gray)
                elif mode == "contrast":
                    score = img_gray.std()
                elif mode == "exposure_score":
                    # Calculate histogram
                    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                    # Simple exposure metric: penalize too many dark or bright pixels
                    # Low score = bad exposure (either under or over)
                    # Normalize hist
                    hist_norm = hist / hist.sum()

                    # Measure entropy or spread?
                    # Let's use a heuristic: score = 1 - (under_exposed + over_exposed)
                    # Under: < 20, Over: > 235
                    under = hist_norm[:20].sum()
                    over = hist_norm[235:].sum()
                    score = 1.0 - (under + over)
                    # Scale to 0-100 for consistency with others? Or 0-1.
                    # Let's keep 0-1.

                scores.append(score)
        except Exception as e:
            raise InferenceError(f"Brightness/Contrast analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"{mode.capitalize()} ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores),
        }


class IQA_Colorfulness(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_Colorfulness",
            display_name="IQA: Colorfulness (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze.\n\nMeasures color variety and saturation using the Hasler-Süsstrunk colorfulness metric.",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average of all scores (recommended)\n• min: Least colorful image\n• max: Most colorful image\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, aggregation, **kwargs):
        return get_hash([aggregation])

    @classmethod
    def VALIDATE_INPUTS(cls, image, aggregation, **kwargs):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation: {aggregation}"
        return True

    @classmethod
    def execute(cls, image, aggregation):
        scores = []
        try:
            img_list = tensor_to_numpy(image)
            for img_np in img_list:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                B, G, R = cv2.split(img_bgr.astype("float"))

                rg = np.absolute(R - G)
                yb = np.absolute(0.5 * (R + G) - B)
                (rbMean, rbStd) = (np.mean(rg), np.std(rg))
                (ybMean, ybStd) = (np.mean(yb), np.std(yb))
                stdRoot = np.sqrt((rbStd**2) + (ybStd**2))
                meanRoot = np.sqrt((rbMean**2) + (ybMean**2))
                score = stdRoot + (0.3 * meanRoot)
                scores.append(score)
        except Exception as e:
            raise InferenceError(f"Colorfulness analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"Colorfulness ({aggregation}): {final_score:.2f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores),
        }


class IQA_Noise_Estimation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_Noise_Estimation",
            display_name="IQA: Noise Estimation (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze for noise.\n\nUses Donoho's Median Absolute Deviation (MAD) method via wavelet decomposition.\nHigher values indicate more noise.",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average noise level (recommended)\n• min: Cleanest image in batch\n• max: Noisiest image in batch\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, aggregation, **kwargs):
        return get_hash([aggregation])

    @classmethod
    def VALIDATE_INPUTS(cls, image, aggregation, **kwargs):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation: {aggregation}"
        return True

    @classmethod
    def execute(cls, image, aggregation):
        scores = []
        try:
            img_list = tensor_to_numpy(image)
            for img_np in img_list:
                # Use scikit-image's estimate_sigma
                # It uses Median Absolute Deviation of Wavelet coefficients (Donoho's method)
                # It's robust and standard.
                # image is H,W,3 or H,W. estimate_sigma handles multichannel.
                # We assume average across channels or treat as multichannel.

                # Normalize to 0-1 for skimage consistency if needed?
                # skimage usually works with float 0-1 or uint8.
                # Let's pass the uint8 img_np directly, it handles it.
                sigma = skimage.restoration.estimate_sigma(
                    img_np, channel_axis=-1, average_sigmas=True
                )

                scores.append(float(sigma))
        except Exception as e:
            raise InferenceError(f"Noise estimation failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"Noise Level ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores),
        }


class IQA_EdgeDensity(io.ComfyNode):
    """Measures edge density using Canny edge detection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_EdgeDensity",
            display_name="IQA: Edge Density (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze.\n\nMeasures edge density as percentage of edge pixels.\nHigher scores indicate more detailed/complex images.",
                ),
                io.Int.Input(
                    "low_threshold",
                    default=50,
                    min=0,
                    max=255,
                    tooltip="Canny edge detector lower threshold (0-255).\n\n• Lower values detect more edges (including noise)\n• Higher values detect only strong edges\n• Must be less than high_threshold\n• Default 50 works well for most images",
                ),
                io.Int.Input(
                    "high_threshold",
                    default=150,
                    min=0,
                    max=255,
                    tooltip="Canny edge detector upper threshold (0-255).\n\n• Edges above this are always kept\n• Edges between low and high are kept if connected to strong edges\n• Higher values = fewer, stronger edges only\n• Default 150 works well for most images",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average edge density (recommended)\n• min: Image with least edges\n• max: Image with most edges\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Image.Output("edge_map"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, low_threshold, high_threshold, aggregation, **kwargs):
        return get_hash([low_threshold, high_threshold, aggregation])

    @classmethod
    def VALIDATE_INPUTS(
        cls, image, low_threshold, high_threshold, aggregation, **kwargs
    ):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation: {aggregation}"
        if low_threshold >= high_threshold:
            return "Low threshold must be less than high threshold."
        return True

    @classmethod
    def execute(cls, image, low_threshold, high_threshold, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)

            for img_np in img_list:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Canny edge detection
                edges = cv2.Canny(img_gray, low_threshold, high_threshold)

                # Edge density = percentage of edge pixels
                total_pixels = edges.size
                edge_pixels = np.sum(edges > 0)
                density = (edge_pixels / total_pixels) * 100  # As percentage

                scores.append(density)

                # Create RGB edge map for visualization
                edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                map_tensor = torch.from_numpy(edge_rgb.astype(np.float32) / 255.0)
                maps.append(map_tensor)

        except Exception as e:
            raise InferenceError(f"Edge density calculation failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"Edge Density ({aggregation}): {final_score:.2f}%"

        if maps:
            maps_out = torch.stack(maps)
        else:
            maps_out = torch.zeros_like(image)

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, maps_out, scores),
        }


class IQA_Saturation(io.ComfyNode):
    """Measures average color saturation using HSV colorspace."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_Saturation",
            display_name="IQA: Saturation (OpenCV)",
            category="IQA/OpenCV",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch of images to analyze.\n\nConverts to HSV colorspace and analyzes the Saturation channel.\nScore is on 0-100 scale (percentage saturation).",
                ),
                io.Enum.Input(
                    "mode",
                    ["mean", "std", "min", "max"],
                    default="mean",
                    tooltip="What statistic to compute from saturation channel:\n\n• mean: Average saturation (recommended)\n  - Higher = more vibrant colors overall\n\n• std: Standard deviation of saturation\n  - Higher = more variation in color intensity\n\n• min: Minimum saturation value\n  - Useful to detect desaturated regions\n\n• max: Maximum saturation value\n  - Useful to detect highly saturated areas",
                ),
                io.Enum.Input(
                    "aggregation",
                    ["mean", "min", "max", "median", "first"],
                    default="mean",
                    tooltip="How to combine scores from batch images:\n\n• mean: Average saturation (recommended)\n• min: Least saturated image\n• max: Most saturated image\n• median: Middle value, robust to outliers\n• first: Only use first image's score",
                ),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, mode, aggregation, **kwargs):
        return get_hash([mode, aggregation])

    @classmethod
    def VALIDATE_INPUTS(cls, image, mode, aggregation, **kwargs):
        if image is None:
            return "Image input is missing."
        if aggregation not in ["mean", "min", "max", "median", "first"]:
            return f"Invalid aggregation: {aggregation}"
        if mode not in ["mean", "std", "min", "max"]:
            return f"Invalid mode: {mode}"
        return True

    @classmethod
    def execute(cls, image, mode, aggregation):
        scores = []

        try:
            img_list = tensor_to_numpy(image)

            for img_np in img_list:
                # Convert RGB to HSV
                img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                saturation_channel = img_hsv[:, :, 1]  # S channel (0-255)

                # Normalize to 0-100 scale
                sat_normalized = saturation_channel / 255.0 * 100

                if mode == "mean":
                    score = np.mean(sat_normalized)
                elif mode == "std":
                    score = np.std(sat_normalized)
                elif mode == "min":
                    score = np.min(sat_normalized)
                elif mode == "max":
                    score = np.max(sat_normalized)

                scores.append(score)

        except Exception as e:
            raise InferenceError(f"Saturation analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"Saturation {mode} ({aggregation}): {final_score:.2f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores),
        }
