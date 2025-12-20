"""
ComfyUI-Image-Quality-Assessment Test Configuration

Session-scoped fixtures for ComfyUI server management and API testing.
"""

# ============================================================================
# CRITICAL: Mock ComfyUI modules BEFORE anything else
# This prevents import errors when pytest discovers and imports test modules
# ============================================================================
import os
import sys
from unittest.mock import MagicMock

# Set testing environment
os.environ['COMFYUI_TESTING'] = '1'
os.environ['PYTEST_CURRENT_TEST'] = 'true'

# Mock ComfyUI modules at module level - BEFORE any imports can trigger __init__.py
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

# Now safe to import pytest and other modules
import pytest
import subprocess
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Tell pytest to ignore certain files during collection
collect_ignore = ["__init__.py"]

# Path configuration
CUSTOM_NODE_ROOT = Path(__file__).parent.parent
COMFY_ROOT = CUSTOM_NODE_ROOT.parent.parent
VENV_PYTHON = COMFY_ROOT / "venv" / "Scripts" / "python.exe"


class ComfyUIAPIClient:
    """Helper class for interacting with ComfyUI API during tests"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client_id = "pytest-iqa-client"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system stats to verify server is running"""
        response = requests.get(f"{self.base_url}/system_stats", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def get_object_info(self) -> Dict[str, Any]:
        """Get all registered node info - useful for verifying nodes loaded"""
        response = requests.get(f"{self.base_url}/object_info", timeout=30)
        response.raise_for_status()
        return response.json()
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a workflow for execution"""
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        response = requests.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for workflow to complete"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    
                    # Check if completed
                    if status.get("completed", False):
                        return history[prompt_id]
                    
                    # Check for error
                    status_str = status.get("status_str", "")
                    if status_str == "error":
                        raise RuntimeError(
                            f"Workflow execution failed: {status}"
                        )
            except requests.RequestException:
                pass  # Server might be busy
            
            time.sleep(0.5)
        
        raise TimeoutError(
            f"Workflow did not complete within {timeout}s (prompt_id: {prompt_id})"
        )
    
    def execute_workflow(self, workflow: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Queue workflow and wait for completion"""
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        return self.wait_for_completion(prompt_id, timeout)
    
    def node_exists(self, node_class: str) -> bool:
        """Check if a node class is registered"""
        try:
            object_info = self.get_object_info()
            return node_class in object_info
        except Exception:
            return False


@pytest.fixture(scope="session")
def comfyui_server():
    """
    Start ComfyUI server for the entire test session.
    Automatically shuts down when tests complete.
    """
    server_url = "http://127.0.0.1:8199"
    
    # Check if server is already running
    try:
        response = requests.get(f"{server_url}/system_stats", timeout=2)
        if response.status_code == 200:
            yield {"url": server_url, "process": None, "external": True}
            return
    except (requests.ConnectionError, requests.Timeout):
        pass
    
    # Prepare environment for subprocess (remove testing flags so nodes register)
    env = os.environ.copy()
    env.pop('COMFYUI_TESTING', None)
    env.pop('PYTEST_CURRENT_TEST', None)

    # Start ComfyUI in subprocess
    process = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "main.py",
            "--listen", "127.0.0.1",
            "--port", "8199",
            "--disable-auto-launch",
            "--cpu"
        ],
        cwd=str(COMFY_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for server to be ready
    max_retries = 60
    retry_count = 0
    ready = False
    
    while retry_count < max_retries:
        try:
            response = requests.get(f"{server_url}/system_stats", timeout=2)
            if response.status_code == 200:
                ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1)
            retry_count += 1
    
    if not ready:
        process.terminate()
        pytest.skip("ComfyUI server not available (could not start within timeout)")
    
    yield {"url": server_url, "process": process, "external": False}
    
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture
def api_client(comfyui_server) -> ComfyUIAPIClient:
    return ComfyUIAPIClient(comfyui_server["url"])


@pytest.fixture
def workflow_fixtures_path() -> Path:
    return CUSTOM_NODE_ROOT / "tests" / "fixtures" / "workflows"


# Unit test fixtures
@pytest.fixture
def sample_image_tensor():
    """Returns a dummy [1, 64, 64, 3] float32 tensor representing a ComfyUI image"""
    import torch
    return torch.rand(1, 64, 64, 3)
