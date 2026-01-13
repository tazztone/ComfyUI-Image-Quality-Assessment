from .pyiqa_nodes import PyIQA_NoReferenceNode, PyIQA_FullReferenceNode
from .opencv_nodes import (
    IQA_Blur_Estimation,
    IQA_Brightness_Contrast,
    IQA_Colorfulness,
    IQA_Noise_Estimation,
    IQA_EdgeDensity,
    IQA_Saturation,
)
from .logic_nodes import PyIQA_EnsembleNode, IQA_ThresholdFilter, IQA_BatchRanker
from .visualization_nodes import IQA_HeatmapVisualizer
from .score_normalizer import IQA_ScoreNormalizer
from .analysis_nodes import (
    Analysis_BlurDetection,
    Analysis_ColorHarmony,
    Analysis_Clipping,
    Analysis_ColorCast,
    Analysis_ColorTemperature,
    Analysis_Contrast,
    Analysis_Defocus,
    Analysis_EdgeDensity,
    Analysis_Entropy,
    Analysis_NoiseEstimation,
    Analysis_RGBHistogram,
    Analysis_SharpnessFocusScore,
)
from .dataset_nodes import (
    LoadImageDataSetFromFolderNode_Custom,
    LoadImageTextDataSetFromFolderNode_Custom,
)

NODE_CLASS_MAPPINGS = {
    "PyIQA_NoReferenceNode": PyIQA_NoReferenceNode,
    "PyIQA_FullReferenceNode": PyIQA_FullReferenceNode,
    "IQA_Blur_Estimation": IQA_Blur_Estimation,
    "IQA_Brightness_Contrast": IQA_Brightness_Contrast,
    "IQA_Colorfulness": IQA_Colorfulness,
    "IQA_Noise_Estimation": IQA_Noise_Estimation,
    "IQA_EdgeDensity": IQA_EdgeDensity,
    "IQA_Saturation": IQA_Saturation,
    "PyIQA_EnsembleNode": PyIQA_EnsembleNode,
    "IQA_ThresholdFilter": IQA_ThresholdFilter,
    "IQA_BatchRanker": IQA_BatchRanker,
    "IQA_HeatmapVisualizer": IQA_HeatmapVisualizer,
    "IQA_ScoreNormalizer": IQA_ScoreNormalizer,
    "Analysis_BlurDetection": Analysis_BlurDetection,
    "Analysis_ColorHarmony": Analysis_ColorHarmony,
    "Analysis_Clipping": Analysis_Clipping,
    "Analysis_ColorCast": Analysis_ColorCast,
    "Analysis_ColorTemperature": Analysis_ColorTemperature,
    "Analysis_Contrast": Analysis_Contrast,
    "Analysis_Defocus": Analysis_Defocus,
    "Analysis_EdgeDensity": Analysis_EdgeDensity,
    "Analysis_Entropy": Analysis_Entropy,
    "Analysis_NoiseEstimation": Analysis_NoiseEstimation,
    "Analysis_RGBHistogram": Analysis_RGBHistogram,
    "Analysis_SharpnessFocusScore": Analysis_SharpnessFocusScore,
    "LoadImageDataSetFromFolder_Custom": LoadImageDataSetFromFolderNode_Custom,
    "LoadImageTextDataSetFromFolder_Custom": LoadImageTextDataSetFromFolderNode_Custom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyIQA_NoReferenceNode": "IQA: PyIQA No-Reference",
    "PyIQA_FullReferenceNode": "IQA: PyIQA Full-Reference",
    "IQA_Blur_Estimation": "IQA: Blur Estimation (OpenCV)",
    "IQA_Brightness_Contrast": "IQA: Brightness & Contrast (OpenCV)",
    "IQA_Colorfulness": "IQA: Colorfulness (OpenCV)",
    "IQA_Noise_Estimation": "IQA: Noise Estimation (OpenCV)",
    "IQA_EdgeDensity": "IQA: Edge Density (OpenCV)",
    "IQA_Saturation": "IQA: Saturation (OpenCV)",
    "PyIQA_EnsembleNode": "IQA: Ensemble Scorer",
    "IQA_ThresholdFilter": "IQA: Threshold Filter",
    "IQA_BatchRanker": "IQA: Batch Ranker",
    "IQA_HeatmapVisualizer": "IQA: Heatmap Visualizer",
    "IQA_ScoreNormalizer": "IQA: Score Normalizer",
    "Analysis_BlurDetection": "Analysis: Blur Detection",
    "Analysis_ColorHarmony": "Analysis: Color Harmony",
    "Analysis_Clipping": "Analysis: Clipping",
    "Analysis_ColorCast": "Analysis: Color Cast",
    "Analysis_ColorTemperature": "Analysis: Color Temperature",
    "Analysis_Contrast": "Analysis: Contrast",
    "Analysis_Defocus": "Analysis: Defocus",
    "Analysis_EdgeDensity": "Analysis: Edge Density",
    "Analysis_Entropy": "Analysis: Entropy",
    "Analysis_NoiseEstimation": "Analysis: Noise Estimation",
    "Analysis_RGBHistogram": "Analysis: RGB Histogram",
    "Analysis_SharpnessFocusScore": "Analysis: Sharpness/Focus",
    "LoadImageDataSetFromFolder_Custom": "IQA: Load Image Dataset",
    "LoadImageTextDataSetFromFolder_Custom": "IQA: Load Image & Text Dataset",
}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
