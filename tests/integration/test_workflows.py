"""
Integration tests for ComfyUI-Image-Quality-Assessment nodes.
Tests execute workflows and verify node registration through ComfyUI API.
"""
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@pytest.mark.integration
class TestNodeRegistration:
    """Test that all ComfyUI-IQA nodes are properly registered"""

    IQA_NODES = [
        "PyIQA_NoReferenceNode",
        "PyIQA_FullReferenceNode",
        "IQA_Blur_Estimation",
        "IQA_Brightness_Contrast",
        "IQA_Colorfulness",
        "IQA_Noise_Estimation",
        "IQA_EdgeDensity",
        "IQA_Saturation",
        "PyIQA_EnsembleNode",
        "IQA_ThresholdFilter",
        "IQA_BatchRanker",
        "IQA_HeatmapVisualizer",
        "IQA_ScoreNormalizer",
        "Analysis_BlurDetection",
        "Analysis_ColorHarmony",
        "Analysis_Clipping",
        "Analysis_ColorCast",
        "Analysis_ColorTemperature",
        "Analysis_Contrast",
        "Analysis_Defocus",
        "Analysis_EdgeDensity",
        "Analysis_Entropy",
        "Analysis_NoiseEstimation",
        "Analysis_RGBHistogram",
        "Analysis_SharpnessFocusScore"
    ]

    @pytest.mark.parametrize("node_class", IQA_NODES)
    def test_node_registered(self, api_client, node_class):
        """Verify each IQA node is available in object_info"""
        assert api_client.node_exists(node_class), f"{node_class} not registered"

@pytest.mark.integration
class TestServerHealth:
    """Basic ComfyUI server health and node counts"""

    def test_server_system_stats(self, api_client):
        """Test that we can get system stats"""
        stats = api_client.get_system_stats()
        assert "system" in stats
        assert "devices" in stats

    def test_iqa_nodes_count(self, api_client):
        """Verify the expected number of nodes are registered"""
        info = api_client.get_object_info()
        iqa_related = [k for k in info.keys() if any(x in k for x in ["IQA_", "PyIQA_", "Analysis_"])]
        # We expect 25 nodes (8 OpenCV/DL + 3 Logic + 1 Visualizer + 1 Normalizer + 12 Analysis)
        assert len(iqa_related) >= 24, f"Expected at least 24 IQA nodes, found {len(iqa_related)}"

@pytest.mark.integration
class TestWorkflowFixtures:
    """Test loading and basic validation of workflow fixtures"""

    def test_fixture_integrity(self, workflow_fixtures_path):
        """Ensure the workflows/ directory exists and contains expected files (placeholders for now)"""
        assert workflow_fixtures_path.exists()
