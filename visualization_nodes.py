import cv2
import numpy as np
import torch
from .comfy_compat import io
from .iqa_core import InferenceError, tensor_to_numpy

class IQA_HeatmapVisualizer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_HeatmapVisualizer",
            display_name="IQA: Heatmap Visualizer",
            category="IQA/Visualization",
            inputs=[
                io.Image.Input("image", tooltip="Base image for heatmap generation.\n\nIf no score_map is provided, this image will be converted to grayscale and used as the intensity map."),
                io.Enum.Input("colormap", ["JET", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "HOT"], default="JET", tooltip="Color scheme for the heatmap:\n\n• JET: Classic rainbow (blue→cyan→green→yellow→red)\n• VIRIDIS: Perceptually uniform (purple→blue→green→yellow)\n• PLASMA: Warm tones (purple→pink→orange→yellow)\n• INFERNO: Dark to bright (black→purple→red→yellow)\n• MAGMA: Similar to inferno, smoother\n• HOT: Heat colors (black→red→yellow→white)"),
                io.Float.Input("normalize_min", default=0.0, min=0.0, max=1.0, tooltip="Minimum value for normalization (0.0-1.0).\n\n• Values at or below this become the colormap's minimum color\n• Use to adjust dynamic range of visualization\n• Lower value = more sensitivity to low intensities"),
                io.Float.Input("normalize_max", default=1.0, min=0.0, max=1.0, tooltip="Maximum value for normalization (0.0-1.0).\n\n• Values at or above this become the colormap's maximum color\n• Use to adjust dynamic range of visualization\n• Higher value = more sensitivity to high intensities"),
                io.Image.Input("score_map_optional", optional=True, tooltip="Optional: Pre-computed score/intensity map.\n\n• Connect from IQA nodes that output maps (e.g., blur_map, edge_map)\n• If not connected, the main 'image' input is used\n• Will be converted to grayscale before colormap application")
            ],
            outputs=[
                io.Image.Output("heatmap_image")
            ]
        )

    @classmethod
    def execute(cls, image, colormap, normalize_min, normalize_max, score_map_optional=None):
        # If score_map is provided, use it. Else, we assume the input 'image' IS the map (grayscale)?
        # Or maybe this node is intended to TAKE a score map from IQA_Blur and colorize it?
        # IQA_Blur returns "blur_map" which is already RGB heatmap.
        # But maybe we want to visualize a raw grayscale map?

        target_map = score_map_optional if score_map_optional is not None else image

        # We process batch
        # Assuming target_map is [B, H, W, C]

        results = []

        # Colormap mapping
        cm_code = getattr(cv2, f"COLORMAP_{colormap}", cv2.COLORMAP_JET)

        try:
            map_list = tensor_to_numpy(target_map)
            for img_np in map_list:
                # Convert to grayscale if it's RGB
                if img_np.shape[2] == 3:
                     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                     gray = img_np[:, :, 0]

                # Normalize to 0-255 based on input range (which is usually 0-1 for tensors, but tensor_to_numpy makes it 0-255)
                # But if we want to custom normalize:
                # The input 'gray' is 0-255.
                # The user params normalize_min/max are 0.0-1.0.

                # Rescale based on user range
                # value = (value - min) / (max - min)
                gray_f = gray.astype(np.float32) / 255.0
                gray_f = (gray_f - normalize_min) / (normalize_max - normalize_min + 1e-6)
                gray_f = np.clip(gray_f, 0.0, 1.0)
                gray_u8 = (gray_f * 255).astype(np.uint8)

                # Apply colormap
                heatmap = cv2.applyColorMap(gray_u8, cm_code)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                results.append(torch.from_numpy(heatmap.astype(np.float32) / 255.0))

        except Exception as e:
            raise InferenceError(f"Heatmap generation failed: {str(e)}")

        return io.NodeOutput(torch.stack(results))
