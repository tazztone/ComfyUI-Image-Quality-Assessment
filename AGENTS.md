# ComfyUI-IQA-Node Developer Guide

This repository contains custom nodes for ComfyUI that perform Image Quality Assessment (IQA) using `pyiqa` and `opencv`.

## Repository Structure

- `pyiqa_nodes.py`: Contains nodes that use the `pyiqa` library (Deep Learning models).
- `opencv_nodes.py`: Contains nodes that use `opencv` and `scikit-image` (Classical CV metrics).
- `logic_nodes.py`: Contains utility nodes for filtering and ranking.
- `score_normalizer.py`: Contains the `IQA_ScoreNormalizer` node.
- `visualization_nodes.py`: Contains nodes for visualizing results (e.g., heatmaps).
- `iqa_core.py`: Shared core logic, including the `ModelCache` and utilities.
- `comfy_compat.py`: A compatibility layer that provides the V3 Node Schema (`io.ComfyNode`, `io.Schema`, etc.).
- `web/js/`: Frontend extensions.
- `tests/`: Standalone unit tests.

## Coding Standards

### Node Definition (V3 Schema)
All nodes MUST be defined using the V3 schema provided by `comfy_compat.py`. Do not use the legacy `INPUT_TYPES` dictionary directly.

Example:
```python
from .comfy_compat import io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="IQA",
            inputs=[ io.Image.Input("image") ],
            outputs=[ io.Float.Output("score") ]
        )

    @classmethod
    def execute(cls, image):
        # ... logic ...
        return io.NodeOutput(score)
```

### Dependencies
- **Numpy**: We pin `numpy<2.0.0` in `requirements.txt`. Do NOT upgrade numpy to 2.x as it breaks `imgaug` (used by some dependencies).
- **PyIQA**: Used for DL metrics.
- **OpenCV/Scikit-Image**: Used for classical metrics.

### Outputs
- Nodes typically return a calculated score (float) and often a string representation for UI display.
- Some nodes return `raw_scores` (list) alongside aggregated scores to support batch processing logic.

## Testing & Development environment

- Use `scripts/setup_dev_env.sh` to set up a development environment. This script will:
  - Clone ComfyUI (if not present).
  - Install dependencies.
  - Generate `env_setup.sh` to set `PYTHONPATH`.

To run tests or scripts, source the environment first:
```bash
source env_setup.sh
python <your_script.py>
```

## Frontend
- `web/js/iqa_score_display.js` handles the display of scores on the nodes themselves.
- Nodes return a dictionary `{ "ui": { "text": ... }, "result": ... }`. The `io.NodeOutput` helper handles this structure automatically.
