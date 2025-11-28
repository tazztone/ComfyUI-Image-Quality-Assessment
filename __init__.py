from .pyiqa_nodes import PyIQA_NoReferenceNode, PyIQA_FullReferenceNode
from .opencv_nodes import IQA_Blur_Estimation, IQA_Brightness_Contrast, IQA_Colorfulness

NODE_CLASS_MAPPINGS = {
    "PyIQA_NoReferenceNode": PyIQA_NoReferenceNode,
    "PyIQA_FullReferenceNode": PyIQA_FullReferenceNode,
    "IQA_Blur_Estimation": IQA_Blur_Estimation,
    "IQA_Brightness_Contrast": IQA_Brightness_Contrast,
    "IQA_Colorfulness": IQA_Colorfulness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyIQA_NoReferenceNode": "PyIQA No-Reference Metric",
    "PyIQA_FullReferenceNode": "PyIQA Full-Reference Metric",
    "IQA_Blur_Estimation": "OpenCV Blur Estimation",
    "IQA_Brightness_Contrast": "OpenCV Brightness & Contrast",
    "IQA_Colorfulness": "OpenCV Colorfulness",
}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
