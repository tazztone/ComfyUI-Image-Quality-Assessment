from .pyiqa_nodes import PyIQANode
from .opencv_nodes import OpenCV_IQA_Node

NODE_CLASS_MAPPINGS = {
    "PyIQA_Metric_Node": PyIQANode,
    "OpenCV_Metric_Node": OpenCV_IQA_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyIQA_Metric_Node": "PyIQA Deep Image Analysis",
    "OpenCV_Metric_Node": "OpenCV Basic Image Analysis"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
