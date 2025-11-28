import torch
import numpy as np
import sys

# Safe import for pyiqa
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

# Global cache for loaded models
# Key: (metric_name, device), Value: model_instance
LOADED_MODELS = {}

class PyIQANode:
    @classmethod
    def INPUT_TYPES(s):
        # Fallback list if pyiqa is not installed or import fails
        metrics = ["hyperiqa", "musique", "nima", "lpips", "fid", "ssim", "psnr", "brisque", "clip_score"]

        if PYIQA_AVAILABLE:
            try:
                metrics = sorted(pyiqa.list_models())
            except Exception:
                pass

        return {
            "required": {
                "image": ("IMAGE",),  # The distorted/generated image
                "metric": (metrics, {"default": "hyperiqa"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),  # Required for FR metrics like LPIPS/SSIM
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "process"
    CATEGORY = "IQA"

    def process(self, image, metric, device, aggregation, keep_model_loaded, reference_image=None):
        global LOADED_MODELS

        if not PYIQA_AVAILABLE:
            return (0.0, "Error: 'pyiqa' library not found. Please install it via requirements.txt")

        # 1. Handle Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Model Retrieval / Initialization
        model_key = (metric, device)
        iqa_model = None

        # Check if model is already loaded
        if model_key in LOADED_MODELS:
            iqa_model = LOADED_MODELS[model_key]
        else:
            try:
                # create_metric downloads weights automatically
                iqa_model = pyiqa.create_metric(metric, device=device)
                if keep_model_loaded:
                    LOADED_MODELS[model_key] = iqa_model
            except Exception as e:
                print(f"Error loading metric {metric}: {e}")
                return (0.0, f"Error loading model: {e}")

        # 3. Prepare Inputs
        # ComfyUI: [B, H, W, C] (0-1 float)
        # PyIQA:   [B, C, H, W] (0-1 float)
        dist_tensor = image.permute(0, 3, 1, 2).to(device)

        ref_tensor = None
        if reference_image is not None:
            ref_tensor = reference_image.permute(0, 3, 1, 2).to(device)
            # Resize reference if dimensions don't match
            if dist_tensor.shape[2:] != ref_tensor.shape[2:]:
                import torch.nn.functional as F
                ref_tensor = F.interpolate(ref_tensor, size=dist_tensor.shape[2:], mode='bilinear', align_corners=False)

        # 4. Inference
        scores = []
        try:
            with torch.no_grad():
                # Handle batch mismatch if user provided 1 ref image for N dist images
                if ref_tensor is not None:
                    if ref_tensor.shape[0] == 1 and dist_tensor.shape[0] > 1:
                        ref_tensor = ref_tensor.repeat(dist_tensor.shape[0], 1, 1, 1)

                    raw_score = iqa_model(dist_tensor, ref_tensor)
                else:
                    raw_score = iqa_model(dist_tensor)

                # raw_score shape analysis
                if raw_score.dim() == 0:
                    scores = [raw_score.item()] * dist_tensor.shape[0]
                else:
                    scores = raw_score.flatten().cpu().tolist()

        except Exception as e:
            return (0.0, f"Inference Error: {e}")
        finally:
            if not keep_model_loaded:
                # If we are not caching, remove from cache if it exists
                if model_key in LOADED_MODELS:
                    del LOADED_MODELS[model_key]
                    del iqa_model
                    torch.cuda.empty_cache()

        # 5. Aggregation
        if not scores:
            return (0.0, "Error: No scores returned")

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

        return (final_score, f"{metric} ({aggregation}): {final_score:.4f}")
