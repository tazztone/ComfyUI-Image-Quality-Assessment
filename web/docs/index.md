# IQA Nodes Documentation

## OpenCV Nodes

### IQA: Blur Estimation (OpenCV)
Estimates image blur using the Variance of Laplacian method.
- **Input**: Image
- **Output**: Score (Higher is less blurry/more edges), Text, Heatmap
- **Interpretation**: Low values indicate blur. Threshold depends on image content but generally < 100 is blurry.

### IQA: Brightness & Contrast (OpenCV)
Calculates brightness (mean pixel value) or contrast (standard deviation).
- **Mode**: Brightness or Contrast

### IQA: Colorfulness (OpenCV)
Calculates image colorfulness metric.

## PyIQA Nodes

### IQA: PyIQA No-Reference
Uses deep learning models to assess quality without a reference image.
- **Metrics**: hyperiqa, maniqa, etc.
- **Device**: auto, cuda, cpu

### IQA: PyIQA Full-Reference
Compares distorted image against a reference image.
- **Metrics**: lpips, ssim, psnr, etc.
- **Input**: Distorted Image, Reference Image
- **Note**: Reference image batch size must be 1 or equal to distorted image batch size.
