import torch
import numpy as np
import sys
from .iqa_core import get_model_cache, aggregate_scores, get_hash, ModelLoadError, InferenceError
from .comfy_compat import io

# Safe import for pyiqa
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

class PyIQA_Base:
    def _load_model(self, metric, device, keep_model_loaded):
        cache = get_model_cache()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_key = (metric, device)
        iqa_model = cache.get(model_key)

        if iqa_model is None:
            try:
                iqa_model = pyiqa.create_metric(metric, device=device)
                if keep_model_loaded:
                    cache.put(model_key, iqa_model)
            except Exception as e:
                raise ModelLoadError(f"Failed to load metric {metric}: {str(e)}")

        return iqa_model, device

    def _cleanup(self, metric, device, keep_model_loaded):
        # The cache handles cleanup automatically if max size is reached.
        pass

class PyIQA_NoReferenceNode(io.ComfyNode, PyIQA_Base):
    @classmethod
    def define_schema(cls):
        # Common NR metrics from research
        metrics = [
            "hyperiqa", "musiq", "nima", "brisque", "clip_score", "niqe", "piqe",
            "topiq_nr", "nrqm", "pi", "ilniqe", "clipiqa", "laion_aes",
            "dbcnn", "cnniqa", "paq2piq", "face_iqa"
        ]

        if PYIQA_AVAILABLE:
            try:
                all_models = pyiqa.list_models()
                for m in all_models:
                    if m not in metrics:
                        metrics.append(m)
                metrics = sorted(list(set(metrics)))
            except:
                pass

        return io.Schema(
            node_id="PyIQA_NoReferenceNode",
            display_name="IQA: PyIQA No-Reference",
            category="IQA/PyIQA",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("metric", metrics, default="hyperiqa"),
                io.Enum.Input("device", ["cuda", "cpu", "auto"], default="auto"),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
                io.Boolean.Input("keep_model_loaded", default=True),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def IS_CHANGED(cls, metric, device, keep_model_loaded, **kwargs):
        # Returns hash of parameters to control caching.
        # If parameters change, we rerun.
        return get_hash({"metric": metric, "device": device, "keep": keep_model_loaded})

    @classmethod
    def VALIDATE_INPUTS(cls, image, metric, device, aggregation, **kwargs):
        if not PYIQA_AVAILABLE: return "PyIQA library not installed"
        if aggregation not in ["mean", "min", "max", "median", "first"]: return f"Invalid aggregation: {aggregation}"
        return True

    @classmethod
    def execute(cls, image, metric, device, aggregation, keep_model_loaded):
        instance = cls() # Create instance to use helper methods
        return instance.process_instance(image, metric, device, aggregation, keep_model_loaded)

    def process_instance(self, image, metric, device, aggregation, keep_model_loaded):
        if not PYIQA_AVAILABLE:
             raise ModelLoadError("PyIQA library is not installed.")

        try:
            iqa_model, device = self._load_model(metric, device, keep_model_loaded)
        except Exception as e:
             # Return dummy or raise? ComfyUI handles exceptions by showing red node.
             raise e

        if metric == "face_iqa":
            pass

        dist_tensor = image.permute(0, 3, 1, 2).to(device)

        scores = []
        try:
            with torch.no_grad():
                raw_score = iqa_model(dist_tensor)

                if isinstance(raw_score, torch.Tensor):
                    if raw_score.dim() == 0:
                        scores = [raw_score.item()] * dist_tensor.shape[0]
                    else:
                        scores = raw_score.flatten().cpu().tolist()
                else:
                    scores = [float(raw_score)] * dist_tensor.shape[0]

        except Exception as e:
            raise InferenceError(f"Inference failed for {metric}: {str(e)}")
        finally:
            self._cleanup(metric, device, keep_model_loaded)

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"{metric} ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores)
        }

class PyIQA_FullReferenceNode(io.ComfyNode, PyIQA_Base):
    @classmethod
    def define_schema(cls):
        # Common FR metrics
        metrics = [
            "lpips", "fid", "ssim", "psnr", "ms_ssim", "dists", "fsim", "vif",
            "pieapp", "ahijk", "ckdn", "gmsd", "nlpd", "vsi", "mad"
        ]

        if PYIQA_AVAILABLE:
            try:
                all_models = pyiqa.list_models()
                for m in all_models:
                    if m not in metrics:
                        metrics.append(m)
                metrics = sorted(list(set(metrics)))
            except:
                pass

        return io.Schema(
            node_id="PyIQA_FullReferenceNode",
            display_name="IQA: PyIQA Full-Reference",
            category="IQA/PyIQA",
            inputs=[
                io.Image.Input("distorted_image"),
                io.Image.Input("reference_image"),
                io.Enum.Input("metric", metrics, default="lpips"),
                io.Enum.Input("device", ["cuda", "cpu", "auto"], default="auto"),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
                io.Boolean.Input("keep_model_loaded", default=True),
            ],
            outputs=[
                io.Float.Output("score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def IS_CHANGED(cls, metric, device, keep_model_loaded, **kwargs):
        return get_hash({"metric": metric, "device": device, "keep": keep_model_loaded})

    @classmethod
    def VALIDATE_INPUTS(cls, distorted_image, reference_image, metric, device, aggregation, **kwargs):
        if not PYIQA_AVAILABLE: return "PyIQA library not installed"
        if distorted_image.shape[0] != reference_image.shape[0] and reference_image.shape[0] != 1:
            return "Batch size mismatch. Reference must be batch size 1 or match distorted image."
        return True

    @classmethod
    def execute(cls, distorted_image, reference_image, metric, device, aggregation, keep_model_loaded):
        instance = cls()
        return instance.process_instance(distorted_image, reference_image, metric, device, aggregation, keep_model_loaded)

    def process_instance(self, distorted_image, reference_image, metric, device, aggregation, keep_model_loaded):
        if not PYIQA_AVAILABLE:
             raise ModelLoadError("PyIQA library is not installed.")

        iqa_model, device = self._load_model(metric, device, keep_model_loaded)

        dist_tensor = distorted_image.permute(0, 3, 1, 2).to(device)
        ref_tensor = reference_image.permute(0, 3, 1, 2).to(device)

        # Resize reference if needed to match distorted
        if dist_tensor.shape[2:] != ref_tensor.shape[2:]:
            import torch.nn.functional as F
            ref_tensor = F.interpolate(ref_tensor, size=dist_tensor.shape[2:], mode='bilinear', align_corners=False)

        scores = []
        try:
            with torch.no_grad():
                # Broadcast reference if needed
                if ref_tensor.shape[0] == 1 and dist_tensor.shape[0] > 1:
                    ref_tensor = ref_tensor.repeat(dist_tensor.shape[0], 1, 1, 1)

                raw_score = iqa_model(dist_tensor, ref_tensor)

                if isinstance(raw_score, torch.Tensor):
                    if raw_score.dim() == 0:
                        scores = [raw_score.item()] * dist_tensor.shape[0]
                    else:
                        scores = raw_score.flatten().cpu().tolist()
                else:
                    scores = [float(raw_score)] * dist_tensor.shape[0]
        except Exception as e:
            raise InferenceError(f"Inference failed for {metric}: {str(e)}")
        finally:
            self._cleanup(metric, device, keep_model_loaded)

        final_score = aggregate_scores(scores, aggregation)
        score_text = f"{metric} ({aggregation}): {final_score:.4f}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(final_score, score_text, scores)
        }
