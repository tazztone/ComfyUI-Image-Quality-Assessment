"""
Integration smoke test that loads ComfyUI and verifies IQA nodes register correctly.
This test requires ComfyUI to be running or startable.
"""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestNodeRegistration:
    """Test that all ComfyUI-IQA nodes are properly registered."""

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
        """Verify each IQA node is available in object_info."""
        assert api_client.node_exists(node_class), f"{node_class} not registered"


@pytest.mark.integration
class TestServerHealth:
    """Basic ComfyUI server health and node counts."""

    def test_server_system_stats(self, api_client):
        """Test that we can get system stats."""
        stats = api_client.get_system_stats()
        assert "system" in stats
        assert "devices" in stats

    def test_iqa_nodes_count(self, api_client):
        """Verify the expected number of IQA nodes are registered."""
        info = api_client.get_object_info()
        iqa_related = [k for k in info.keys() if any(x in k for x in ["IQA_", "PyIQA_", "Analysis_"])]
        assert len(iqa_related) >= 24, f"Expected at least 24 IQA nodes, found {len(iqa_related)}: {iqa_related}"


@pytest.mark.integration
class TestWorkflowFixtures:
    """Test loading and basic validation of workflow fixtures."""

    def test_fixture_directory_exists(self, workflow_fixtures_path):
        """Ensure the workflows/ directory exists."""
        assert workflow_fixtures_path.exists()

    def test_opencv_workflow_valid(self, workflow_fixtures_path):
        """Test OpenCV workflow fixture is valid JSON."""
        workflow_file = workflow_fixtures_path / "test_opencv_metrics.json"
        if workflow_file.exists():
            with open(workflow_file) as f:
                workflow = json.load(f)
            assert isinstance(workflow, dict)

    def test_pyiqa_workflow_valid(self, workflow_fixtures_path):
        """Test PyIQA workflow fixture is valid JSON."""
        workflow_file = workflow_fixtures_path / "test_pyiqa_noreference.json"
        if workflow_file.exists():
            with open(workflow_file) as f:
                workflow = json.load(f)
            assert isinstance(workflow, dict)
