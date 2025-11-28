# ComfyUI-IQA-Node Documentation

This repository provides Image Quality Assessment (IQA) nodes for ComfyUI, leveraging **PyIQA** (Deep Learning based) and **OpenCV** (Classical Computer Vision based) metrics.

## Installation

1.  Clone this repository into `ComfyUI/custom_nodes/`.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: PyIQA requires PyTorch.*

## Nodes Overview

### Deep Learning Metrics (PyIQA)

*   **IQA: PyIQA No-Reference**: Evaluates image quality without a reference image.
    *   *Supported Metrics*: `hyperiqa`, `musique`, `nima`, `brisque`, `clip_score`, `niqe`, `piqe`, `topiq_nr`, `nrqm`, `ilniqe`, `clipiqa`, `laion_aes`, `dbcnn`, `cnniqa`, `paq2piq`, `face_iqa`.
    *   *Inputs*: Image, Metric Name, Device.
    *   *Outputs*: Aggregate Score, Score Text, Raw Scores (list).

*   **IQA: PyIQA Full-Reference**: Compares a distorted image against a reference image.
    *   *Supported Metrics*: `lpips`, `fid`, `ssim`, `psnr`, `ms_ssim`, `dists`, `fsim`, `vif`, `pieapp`, `ahijk`, `ckdn`, `gmsd`, `nlpd`, `vsi`, `mad`.
    *   *Inputs*: Distorted Image, Reference Image.

### Technical Metrics (OpenCV)

*   **IQA: Blur Estimation**: Estimates blurriness.
    *   *Modes*: `laplacian` (Variance of Laplacian), `tenengrad` (Gradient Magnitude).
    *   *Outputs*: Score, Text, Blur Heatmap.
*   **IQA: Brightness & Contrast**: Measures basic image properties.
    *   *Modes*: `brightness`, `contrast`, `exposure_score` (Histogram analysis).
*   **IQA: Colorfulness**: Measures image colorfulness metric.
*   **IQA: Noise Estimation**: Estimates noise level using fast variance estimation.

### Logic & Workflow Tools

*   **IQA: Threshold Filter**: Filters a batch of images based on quality scores.
    *   *Inputs*: Images, Scores (must link from IQA node), Threshold.
    *   *Outputs*: Passed Images, Failed Images.
*   **IQA: Batch Ranker**: Sorts a batch of images by quality score.
    *   *Inputs*: Images, Scores.
    *   *Outputs*: Sorted Images.
*   **IQA: Ensemble Scorer**: Weighted average of multiple scores.

## Caching

Models are cached in memory (LRU Cache, size=3) to speed up execution. Toggle `keep_model_loaded` to control persistence.
