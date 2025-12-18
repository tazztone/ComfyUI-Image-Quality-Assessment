# ComfyUI-IQA-Node

A comprehensive Image Quality Assessment (IQA) custom node collection for ComfyUI. This pack leverages both **PyIQA** for deep learning-based metrics and **OpenCV** for fast, classical computer vision metrics. It also includes utility nodes for filtering and ranking images based on these scores.

## Features

- **PyIQA Deep Image Analysis**:
  - Access to a vast zoo of state-of-the-art IQA models (HyperIQA, MUSIQ, NIMA, LPIPS, FID, SSIM, etc.).
  - **Smart Caching**: Models are cached in memory to avoid reloading, with an option to unload to save VRAM.
  - **Batch Support**: Processes batches of images with configurable aggregation (mean, min, max).
  - **Automatic Device Detection**: Runs on CUDA if available, falls back to CPU.

- **OpenCV Basic Image Analysis**:
  - Lightweight, deterministic metrics.
  - **Blur Estimation**: Detect blurry images using Laplacian variance.
  - **Brightness & Contrast**: Basic image statistics.
  - **Colorfulness**: Measures image vividness.
  - **Noise Estimation**: Estimates noise levels.

- **Logic & Visualization**:
  - **Filtering**: Automatically discard images that don't meet a quality threshold.
  - **Ranking**: Sort a batch of images by quality score.
  - **Ensemble**: Combine multiple scores with custom weights.
  - **Heatmaps**: Visualize metric maps (where supported).

- **Frontend Integration**:
  - Scores are displayed directly on the nodes in the workflow editor.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-IQA-Node.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `pyiqa`, `opencv-python`, `scikit-image`, `torch`, and `torchvision`.*

## Usage

### PyIQA Nodes
- **IQA: PyIQA No-Reference**:
  - Evaluate image quality without a reference image (e.g., aesthetics, technical quality).
  - **Models**: `hyperiqa`, `musiq`, `nima`, `topiq`, etc.
- **IQA: PyIQA Full-Reference**:
  - Compare a generated image against a reference image (ground truth).
  - **Models**: `lpips`, `ssim`, `psnr`, `fid`, etc.

**Common Inputs**:
- `device`: `auto`, `cuda`, or `cpu`.
- `keep_model_loaded`: Optimization for VRAM.

### OpenCV Nodes
- **IQA: Blur Estimation**: Higher score generally means sharper (Laplacian variance).
- **IQA: Brightness & Contrast**: Returns mean brightness and RMS contrast.
- **IQA: Colorfulness**: Returns a colorfulness index (metric by Hasler and Suesstrunk).
- **IQA: Noise Estimation**: Returns estimated noise sigma.
- **IQA: Edge Density**: Measures edge density using Canny detection (percentage of edge pixels).
- **IQA: Saturation**: Measures color saturation using HSV colorspace.

### Logic Nodes
- **IQA: Threshold Filter**:
  - Routes images to "Passed" or "Failed" outputs based on a score threshold.
  - Useful for creating workflows that only save or process high-quality generations.
- **IQA: Batch Ranker**:
  - Sorts a batch of images based on their scores (Ascending/Descending).
  - Optional `take_top_n` to keep only the best images.
- **IQA: Ensemble Scorer**:
  - weighted average of up to 4 different scores.
- **IQA: Score Normalizer**:
  - Normalizes scores to a consistent range (e.g., 0-100).
  - Supports inversion (lower is better → higher is better) and clamping.

### Visualization Nodes
- **IQA: Heatmap Visualizer**:
  - Colorizes single-channel maps (like blur maps or attention maps) using standard colormaps (JET, VIRIDIS, etc.).

## Tips
- Use **Laplacian Blur Score** to filter out blurry generations.
- Use **HyperIQA** or **MUSIQ** to score aesthetic quality.
- Use **LPIPS** to measure how different two images are (requires `reference_image`).
- Combine a scorer with **IQA: Threshold Filter** to automate quality control.

## Credits
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) for the deep learning metrics.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the node framework.
