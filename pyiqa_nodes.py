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

class PyIQA_Base:
    def _load_model(self, metric, device, keep_model_loaded):
        global LOADED_MODELS

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_key = (metric, device)
        iqa_model = None

        if model_key in LOADED_MODELS:
            iqa_model = LOADED_MODELS[model_key]
        else:
            try:
                iqa_model = pyiqa.create_metric(metric, device=device)
                if keep_model_loaded:
                    LOADED_MODELS[model_key] = iqa_model
            except Exception as e:
                print(f"Error loading metric {metric}: {e}")
                return None, f"Error loading model: {e}"

        return iqa_model, device

    def _aggregate(self, scores, method):
        if not scores: return 0.0
        if method == "mean": return float(np.mean(scores))
        if method == "min": return float(np.min(scores))
        if method == "max": return float(np.max(scores))
        if method == "first": return float(scores[0])
        return float(np.mean(scores))

    def _cleanup(self, metric, device, keep_model_loaded):
        if not keep_model_loaded:
            model_key = (metric, device)
            if model_key in LOADED_MODELS:
                del LOADED_MODELS[model_key]
                torch.cuda.empty_cache()

class PyIQA_NoReferenceNode(PyIQA_Base):
    @classmethod
    def INPUT_TYPES(s):
        # Common NR metrics
        metrics = ["hyperiqa", "musique", "nima", "brisque", "clip_score", "niqe", "piqe", "topiq_nr"]
        if PYIQA_AVAILABLE:
            try:
                all_models = pyiqa.list_models()
                # Just add any missing ones to the end if user knows what they are doing
                for m in all_models:
                    if m not in metrics:
                        metrics.append(m)
                metrics = sorted(metrics)
            except:
                pass

        return {
            "required": {
                "image": ("IMAGE",),
                "metric": (metrics, {"default": "hyperiqa"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "process"
    CATEGORY = "IQA/PyIQA"

    def process(self, image, metric, device, aggregation, keep_model_loaded):
        if not PYIQA_AVAILABLE:
            error_msg = "Error: 'pyiqa' library not found."
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}

        iqa_model, device = self._load_model(metric, device, keep_model_loaded)
        if iqa_model is None:
            error_msg = f"Failed to load {metric}"
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}

        dist_tensor = image.permute(0, 3, 1, 2).to(device)

        scores = []
        try:
            with torch.no_grad():
                raw_score = iqa_model(dist_tensor)

                if raw_score.dim() == 0:
                    scores = [raw_score.item()] * dist_tensor.shape[0]
                else:
                    scores = raw_score.flatten().cpu().tolist()
        except Exception as e:
            error_msg = f"Inference Error: {e}"
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}
        finally:
            self._cleanup(metric, device, keep_model_loaded)

        final_score = self._aggregate(scores, aggregation)
        score_text = f"{metric} ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": (final_score, score_text)
        }

class PyIQA_FullReferenceNode(PyIQA_Base):
    @classmethod
    def INPUT_TYPES(s):
        # Common FR metrics
        metrics = ["lpips", "fid", "ssim", "psnr", "ms_ssim", "dists", "fsim", "vif"]
        if PYIQA_AVAILABLE:
            try:
                all_models = pyiqa.list_models()
                for m in all_models:
                    if m not in metrics:
                        metrics.append(m)
                metrics = sorted(metrics)
            except:
                pass

        return {
            "required": {
                "distorted_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "metric": (metrics, {"default": "lpips"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "aggregation": (["mean", "min", "max", "first"], {"default": "mean"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "score_text")
    FUNCTION = "process"
    CATEGORY = "IQA/PyIQA"

    def process(self, distorted_image, reference_image, metric, device, aggregation, keep_model_loaded):
        if not PYIQA_AVAILABLE:
            error_msg = "Error: 'pyiqa' library not found."
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}

        iqa_model, device = self._load_model(metric, device, keep_model_loaded)
        if iqa_model is None:
            error_msg = f"Failed to load {metric}"
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}

        dist_tensor = distorted_image.permute(0, 3, 1, 2).to(device)
        ref_tensor = reference_image.permute(0, 3, 1, 2).to(device)

        # Resize reference if needed
        if dist_tensor.shape[2:] != ref_tensor.shape[2:]:
            import torch.nn.functional as F
            ref_tensor = F.interpolate(ref_tensor, size=dist_tensor.shape[2:], mode='bilinear', align_corners=False)

        scores = []
        try:
            with torch.no_grad():
                # Handle batch broadcasting
                if ref_tensor.shape[0] == 1 and dist_tensor.shape[0] > 1:
                    ref_tensor = ref_tensor.repeat(dist_tensor.shape[0], 1, 1, 1)

                raw_score = iqa_model(dist_tensor, ref_tensor)

                if raw_score.dim() == 0:
                    scores = [raw_score.item()] * dist_tensor.shape[0]
                else:
                    scores = raw_score.flatten().cpu().tolist()
        except Exception as e:
            error_msg = f"Inference Error: {e}"
            return {"ui": {"text": [error_msg]}, "result": (0.0, error_msg)}
        finally:
            self._cleanup(metric, device, keep_model_loaded)

        final_score = self._aggregate(scores, aggregation)
        score_text = f"{metric} ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": (final_score, score_text)
        }
