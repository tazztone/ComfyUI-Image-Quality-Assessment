import torch
import numpy as np
from collections import OrderedDict
import hashlib
import json
import os

# Custom Exceptions
class IQAError(Exception):
    """Base class for IQA exceptions."""
    pass

class ModelLoadError(IQAError):
    """Raised when a model fails to load."""
    pass

class InferenceError(IQAError):
    """Raised when inference fails."""
    pass

class ValidationError(IQAError):
    """Raised when input validation fails."""
    pass

# LRU Cache for Models
class ModelCache:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, model):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = model
        if len(self.cache) > self.max_size:
            # Remove LRU item
            removed_key, removed_model = self.cache.popitem(last=False)
            # Explicit cleanup if possible (though PyTorch handles ref counting)
            del removed_model
            torch.cuda.empty_cache()

# Global Cache Instance (configurable via env var)
CACHE_SIZE = int(os.environ.get("COMFY_IQA_CACHE_SIZE", "3"))
GLOBAL_MODEL_CACHE = ModelCache(max_size=CACHE_SIZE)

def get_model_cache():
    return GLOBAL_MODEL_CACHE

# Shared Utilities
def aggregate_scores(scores, method="mean"):
    """
    Aggregates a list of scores using the specified method.
    """
    if not scores:
        return 0.0

    # Ensure scores is a list or numpy array
    if not isinstance(scores, (list, np.ndarray, tuple)):
        try:
            scores = [float(scores)]
        except:
            return 0.0

    # Filter out NaNs
    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return 0.0

    if method == "mean":
        return float(np.mean(scores))
    elif method == "min":
        return float(np.min(scores))
    elif method == "max":
        return float(np.max(scores))
    elif method == "median":
        return float(np.median(scores))
    elif method == "first":
        return float(scores[0])

    return float(np.mean(scores))

def tensor_to_numpy(image_tensor):
    """
    Converts a ComfyUI Image Tensor [B, H, W, C] to a list of Numpy arrays [H, W, C] (0-255, uint8).
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise ValidationError("Input must be a torch.Tensor")

    results = []
    for i in range(image_tensor.shape[0]):
        img = 255. * image_tensor[i].cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        results.append(img)
    return results

def get_hash(input_data):
    """
    Generates a hash for the input data.
    """
    if isinstance(input_data, (str, int, float, bool)):
        return hashlib.sha256(str(input_data).encode()).hexdigest()
    if isinstance(input_data, dict):
        return hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    # For tensors, we usually rely on ComfyUI's internal caching,
    # but here we might want to just return a constant if we trust the node system.
    return None
