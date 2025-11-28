# ComfyUI-IQA-Node

A comprehensive Image Quality Assessment (IQA) custom node collection for ComfyUI. This pack leverages both **PyIQA** for deep learning-based metrics and **OpenCV** for fast, classical computer vision metrics.

## Features

- **PyIQA Deep Image Analysis**:
  - Access to a vast zoo of state-of-the-art IQA models (HyperIQA, MUSIQ, NIMA, LPIPS, FID, SSIM, etc.).
  - **Smart Caching**: Models are cached in memory to avoid reloading, with an option to unload to save VRAM.
  - **Batch Support**: Processes batches of images with configurable aggregation (mean, min, max).
  - **Automatic Device Detection**: Runs on CUDA if available, falls back to CPU.

- **OpenCV Basic Image Analysis**:
  - Lightweight, deterministic metrics.
  - **Laplacian Blur Score**: Detect blurry images.
  - **Brightness / Contrast**: Basic image statistics.
  - **Colorfulness**: Measures image vividness.
  - **Fast**: No model loading required.

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
   *Note: This will install `pyiqa`, `opencv-python`, `torch`, and `torchvision`.*

## Usage

### PyIQA Node
- **Inputs**:
  - `image`: The image(s) to analyze.
  - `metric`: Select from available PyIQA models (e.g., `hyperiqa` for aesthetics, `lpips` for perceptual similarity).
  - `device`: `auto`, `cuda`, or `cpu`.
  - `aggregation`: How to combine scores if input is a batch (`mean`, `min`, `max`, `first`).
  - `keep_model_loaded`: Keep the model in VRAM for faster subsequent runs. Set to `False` to save memory.
  - `reference_image`: (Optional) Required for Full-Reference (FR) metrics like LPIPS, SSIM, PSNR.

### OpenCV Node
- **Inputs**:
  - `image`: The image(s) to analyze.
  - `method`: `laplacian_blur_score`, `brightness_mean`, `contrast_rms`, `colorfulness`.
  - `aggregation`: How to combine scores if input is a batch.

## Tips
- Use **Laplacian Blur Score** to filter out blurry generations. Higher is sharper.
- Use **HyperIQA** or **MUSIQ** to score aesthetic quality.
- Use **LPIPS** to measure how different two images are (requires `reference_image`).

## Credits
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) for the deep learning metrics.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the node framework.
