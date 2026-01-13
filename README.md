# ComfyUI-IQA-Node

A comprehensive Image Quality Assessment (IQA) custom node collection for ComfyUI. This pack leverages **PyIQA** for deep learning-based metrics, **OpenCV** for classical computer vision metrics, and advanced **Analysis Tools** for detailed image inspection.

## Features

- **PyIQA Deep Image Analysis**:
  - Access to a vast zoo of state-of-the-art IQA models (HyperIQA, MUSIQ, NIMA, LPIPS, FID, SSIM, etc.).
  - **Smart Caching**: Models are cached in memory to avoid reloading.
  - **Batch Support**: Processes batches with configurable aggregation.
  - **Automatic Device Detection**: Runs on CUDA if available.

- **OpenCV & Analysis Tools**:
  - **Classical Metrics**: Blur, Brightness, Contrast, Colorfulness, Noise.
  - **Advanced Analysis**: Color Harmony, Color Temperature, Defocus (FFT), Clipping, Entropy.
  - **Visualizations**: Histograms, Heatmaps, Color Wheels, Edge Maps.

- **Logic & Visualization**:
  - **Filtering & Ranking**: Filter or sort images based on quality scores.
  - **Ensemble**: Combine multiple scores.
  - **Frontend Integration**: Real-time score display on nodes.

### Installation

1. Clone this repository into your `ComfyUI/custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-IQA-Node.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This version uses a **vendored** version of `pyiqa` to prevent dependency conflicts (like downgrading `transformers` or `numpy`). All original `pyiqa` metrics are available except for `qalign`, which is disabled due to its strict requirement for an older version of `transformers`.*

## Usage

### PyIQA Nodes
- **IQA: PyIQA No-Reference**: Evaluate aesthetics/quality (HyperIQA, MUSIQ, NIMA, etc.).
- **IQA: PyIQA Full-Reference**: Compare against reference (LPIPS, SSIM, PSNR, FID, etc.).

### OpenCV Nodes
- **IQA: Blur Estimation**: Global blur detection (Laplacian/Tenengrad).
- **IQA: Brightness & Contrast**: Basic statistics.
- **IQA: Colorfulness**: Vividness metric.
- **IQA: Noise Estimation**: Wavelet-based noise estimation (MAD).
- **IQA: Edge Density**: Edge complexity.
- **IQA: Saturation**: Average/Max saturation.

### Analysis Nodes
- **Analysis: Blur Detection**: Block-based blur analysis with heatmaps.
- **Analysis: Color Harmony**: Identifies color schemes (Complementary, Triadic) and displays color wheel.
- **Analysis: Color Cast**: Detects and visualizes unwanted color tints.
- **Analysis: Color Temperature**: Estimates Kelvin temperature and labels (Warm/Cool).
- **Analysis: Clipping**: Visualizes clipped highlights/shadows or saturation.
- **Analysis: Defocus**: FFT-based frequency analysis to detect defocus.
- **Analysis: Edge Density**: Detailed edge density analysis with maps.
- **Analysis: Entropy**: Measures information content/entropy (bits).
- **Analysis: Noise Estimation**: Variance-based noise mapping.
- **Analysis: RGB Histogram**: Renders RGB histograms for the batch.
- **Analysis: Sharpness/Focus**: Hybrid scoring (Laplacian + Tenengrad).

### Logic Nodes
- **IQA: Threshold Filter**: Route images based on score.
- **IQA: Batch Ranker**: Sort images by score.
- **IQA: Ensemble Scorer**: Weighted average of scores.
- **IQA: Score Normalizer**: Scale/Invert scores.

### Visualization Nodes
- **IQA: Heatmap Visualizer**: Apply colormaps to raw value maps.

## Tips
- Use **Analysis: Defocus** or **IQA: Blur Estimation** to filter bad generations.
- Use **Analysis: Color Harmony** to verify prompt adherence (e.g., "teal and orange").
- Use **IQA: Batch Ranker** to pick the best image from a large batch.

## Credits
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) for DL metrics.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- [ThatGlennD/ComfyUI-Image-Analysis-Tools](https://github.com/ThatGlennD/ComfyUI-Image-Analysis-Tools/) for analysis logic.
