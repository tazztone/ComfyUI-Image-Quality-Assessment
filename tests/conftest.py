"""
ComfyUI-Image-Quality-Assessment Test Configuration

Mocks ComfyUI modules and provides fixtures for testing.
"""

import os
import sys
from unittest.mock import MagicMock

# ============================================================================
# CRITICAL: Mock ComfyUI modules BEFORE anything else
# ============================================================================
os.environ['COMFYUI_TESTING'] = '1'

_MOCK_MODULES = [
    'comfy',
    'comfy.model_management',
    'comfy.utils',
    'nodes',
    'folder_paths',
    'server',
    'execution',
    'comfy_extras',
]

for _module_name in _MOCK_MODULES:
    if _module_name not in sys.modules:
        sys.modules[_module_name] = MagicMock()

import pytest
from pathlib import Path
from typing import Dict, Any

# Tell pytest to ignore package __init__.py
collect_ignore = ["__init__.py"]

CUSTOM_NODE_ROOT = Path(__file__).parent.parent
COMFY_ROOT = CUSTOM_NODE_ROOT.parent.parent


class ComfyUIAPIClient:
    """Helper class for interacting with ComfyUI API during tests."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def get_system_stats(self) -> Dict[str, Any]:
        import requests
        response = requests.get(f"{self.base_url}/system_stats", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def get_object_info(self) -> Dict[str, Any]:
        import requests
        response = requests.get(f"{self.base_url}/object_info", timeout=30)
        response.raise_for_status()
        return response.json()
    
    def node_exists(self, node_class: str) -> bool:
        try:
            object_info = self.get_object_info()
            return node_class in object_info
        except Exception:
            return False


@pytest.fixture(scope="session")
def comfyui_server():
    """Connect to running ComfyUI server or skip."""
    import requests
    for port in [8188, 8199]:
        try:
            url = f"http://127.0.0.1:{port}"
            response = requests.get(f"{url}/system_stats", timeout=2)
            if response.status_code == 200:
                yield {"url": url}
                return
        except (requests.ConnectionError, requests.Timeout):
            continue
    pytest.skip("ComfyUI server not running on port 8188 or 8199")


@pytest.fixture
def api_client(comfyui_server) -> ComfyUIAPIClient:
    return ComfyUIAPIClient(comfyui_server["url"])


@pytest.fixture
def workflow_fixtures_path() -> Path:
    return CUSTOM_NODE_ROOT / "tests" / "fixtures" / "workflows"
