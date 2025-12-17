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
                io.Image.Input("image"),
                io.Enum.Input("colormap", ["JET", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "HOT"], default="JET"),
                io.Float.Input("normalize_min", default=0.0, min=0.0, max=1.0),
                io.Float.Input("normalize_max", default=1.0, min=0.0, max=1.0),
                io.Image.Input("score_map_optional", optional=True) # If upstream node provides a map?
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
