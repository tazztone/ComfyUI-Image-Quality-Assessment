"""
Syntax and structure validation tests.
These tests verify Python files are valid and __init__.py is correct.
"""
import pytest
import subprocess
import sys
from pathlib import Path

CUSTOM_NODE_ROOT = Path(__file__).parent.parent.parent

PYTHON_FILES = [
    "__init__.py",
    "iqa_core.py",
    "utils/iqa_core.py",
    "pyiqa_nodes.py",
    "opencv_nodes.py",
    "logic_nodes.py",
    "analysis_nodes.py",
    "visualization_nodes.py",
    "score_normalizer.py",
    "comfy_compat.py",
]


@pytest.mark.unit
class TestSyntaxValidation:
    """Verify all Python files are syntactically correct."""

    @pytest.mark.parametrize("filename", PYTHON_FILES)
    def test_file_syntax(self, filename):
        """Check that file compiles without syntax errors."""
        filepath = CUSTOM_NODE_ROOT / filename
        if not filepath.exists():
            pytest.skip(f"{filename} not found")

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(filepath)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Syntax error in {filename}:\n{result.stderr}"


@pytest.mark.unit
class TestNodeMappings:
    """Verify __init__.py defines expected node mappings."""

    def test_init_uses_relative_imports(self):
        """Verify __init__.py uses relative imports (required for ComfyUI)."""
        init_file = CUSTOM_NODE_ROOT / "__init__.py"
        content = init_file.read_text()

        assert "from .pyiqa_nodes import" in content, "__init__.py must use relative imports"
        assert "from .opencv_nodes import" in content, "__init__.py must use relative imports"

    def test_node_mappings_defined(self):
        """Check NODE_CLASS_MAPPINGS is defined."""
        init_file = CUSTOM_NODE_ROOT / "__init__.py"
        content = init_file.read_text()

        assert "NODE_CLASS_MAPPINGS" in content
        assert "NODE_DISPLAY_NAME_MAPPINGS" in content
