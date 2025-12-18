# ComfyUI-IQA-Node Developer Guide

This repository contains custom nodes for ComfyUI that perform Image Quality Assessment (IQA) using `pyiqa`, `opencv`, and `scikit-image`.

## Repository Structure

- `pyiqa_nodes.py`: Deep Learning based metrics (uses `pyiqa`).
- `opencv_nodes.py`: Classical computer vision metrics (uses `opencv` and `scikit-image`).
- `logic_nodes.py`: Flow control nodes like `Threshold Filter` and `Batch Ranker`.
- `score_normalizer.py`: Utility for scaling and inverting scores.
- `visualization_nodes.py`: Heatmap generators and visualization helpers.
- `iqa_core.py`: Core shared logic: `ModelCache`, `aggregate_scores`, and exception types.
- `comfy_compat.py`: V3 Node Schema compatibility layer.
- `web/js/`: Javascript extensions for the ComfyUI frontend.
- `tests/`: Standalone unit tests for core logic.

---

## Core Architecture

### Node Definition (V3 Schema)
All nodes MUST inherit from `io.ComfyNode` (defined in `comfy_compat.py`). This allows using a modern schema-based definition that is automatically converted to ComfyUI's legacy `INPUT_TYPES` format.

**Key Requirements:**
- Must implement `define_schema()`.
- Must implement `execute()`.
- Use `io.NodeOutput` for returning results.
- **Tooltips**: Every input MUST have a descriptive `tooltip` parameter. Tooltips should explain:
    - What the parameter does.
    - Expected ranges or units (e.g. 0-255, 0.0-1.0).
    - Recommendations (e.g. "recommended for general use").
    - Use bullet points (`•`) for readability in the ComfyUI UI.

Example:
```python
io.Enum.Input("aggregation", ["mean", "min", "max"], default="mean", 
    tooltip="How to combine scores:\n• mean: Average\n• min: Lowest\n• max: Highest")
```

### Model Caching
Deep learning models are heavy. We use a global `ModelCache` (LRU) in `iqa_core.py` to keep models in VRAM.
- **Cache Size**: Configurable via `COMFY_IQA_CACHE_SIZE` environment variable (default: 3).
- **Cleanup**: `ModelCache` automatically calls `torch.cuda.empty_cache()` when symbols are evicted.

### Score Aggregation
Most nodes support batch processing. The `aggregate_scores` utility in `iqa_core.py` handles:
- **Methods**: `mean`, `max`, `min`, `median`, `first`.
- **Filtering**: Automatically excludes `NaN` values.
- **Type Safety**: Ensures output is a standard Python `float`.

---

## Coding Standards

### Dependencies & Environment
- **Numpy**: Pinned to `<2.0.0` to avoid breaking `imgaug` and other CV dependencies.
- **PyTorch**: Used for tensor manipulation and running PyIQA models.
- **Device Management**: Always support `cuda`, `cpu`, and `auto` selection.

### Handling Batch Tensors
ComfyUI images are `[B, H, W, C]` tensors in `0.0 - 1.0` range.
- Use `tensor_to_numpy()` in `iqa_core.py` to convert to list of `uint8` arrays for OpenCV processing.
- For DL models, permute to `[B, C, H, W]` before passing to `pyiqa`.

### Frontend Integration
The `web/js/iqa_score_display.js` script automatically adds a display widget to any node in the "IQA" category.
- Backend must return `{"ui": {"text": [display_string]}}`.
- The `io.NodeOutput` helper handles this automatically if passed a string as the second argument.

---

## Testing

### Standalone Unit Tests
Located in `tests/test_unit.py`. These tests validate core logic (aggregation, normalization, caching) without requiring a full ComfyUI install.

**Run Tests:**
```bash
python tests/test_unit.py -v
```

### Syntax Verification
Before committing, verify all files are syntactically correct:
```bash
python -m py_compile *.py
```

---

## Configuration

| Env Variable | Description | Default |
|--------------|-------------|---------|
| `COMFY_IQA_CACHE_SIZE` | Max number of DL models to keep in VRAM | `3` |

---

## Development Workflow

1. **Add Logic**: Implement your metric in `opencv_nodes.py` (classical) or `pyiqa_nodes.py` (DL).
2. **Handle Batches**: Ensure your `execute` method handles batch tensors correctly.
3. **Registry**: Register the new node class in `__init__.py`.
4. **Docs**: Update `README.md` and this guide if any new shared utilities are added.
