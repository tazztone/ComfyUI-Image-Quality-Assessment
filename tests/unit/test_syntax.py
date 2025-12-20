"""
Syntax, import, and structure validation tests for ComfyUI-Image-Quality-Assessment.
These tests verify:
1. All Python files are syntactically correct
2. All node files use relative imports (required for ComfyUI)
3. No absolute imports of internal modules
"""
import pytest
import subprocess
import sys
import re
from pathlib import Path

CUSTOM_NODE_ROOT = Path(__file__).parent.parent.parent

# Node files that are loaded by __init__.py and must use relative imports
NODE_FILES = [
    "pyiqa_nodes.py",
    "opencv_nodes.py",
    "logic_nodes.py",
    "analysis_nodes.py",
    "visualization_nodes.py",
    "score_normalizer.py",
]

# Internal modules that must be imported with relative syntax
INTERNAL_MODULES = [
    "comfy_compat",
    "iqa_core",
    "pyiqa_nodes",
    "opencv_nodes",
    "logic_nodes",
    "analysis_nodes",
    "visualization_nodes",
    "score_normalizer",
]

# All Python files that should be syntactically valid
ALL_PYTHON_FILES = [
    "__init__.py",
    "iqa_core.py",
    "comfy_compat.py",
] + NODE_FILES


@pytest.mark.unit
class TestSyntaxValidation:
    """Verify all Python files are syntactically correct."""

    @pytest.mark.parametrize("filename", ALL_PYTHON_FILES)
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
class TestImportPatterns:
    """Verify all node files use relative imports for internal modules."""

    @pytest.mark.parametrize("filename", NODE_FILES)
    def test_uses_relative_imports(self, filename):
        """Check that node files use relative imports for internal modules."""
        filepath = CUSTOM_NODE_ROOT / filename
        if not filepath.exists():
            pytest.skip(f"{filename} not found")

        content = filepath.read_text()
        errors = []

        for module in INTERNAL_MODULES:
            # Check for absolute imports like "from iqa_core import" or "import iqa_core"
            # But NOT "from .iqa_core import" (which is correct)
            
            # Pattern: "from module import" at start of line (absolute import)
            absolute_from = re.search(rf'^from\s+{module}\s+import', content, re.MULTILINE)
            if absolute_from:
                errors.append(f"Found absolute import 'from {module} import' - should be 'from .{module} import'")
            
            # Pattern: "import module" at start of line (absolute import)
            absolute_import = re.search(rf'^import\s+{module}($|\s)', content, re.MULTILINE)
            if absolute_import:
                errors.append(f"Found absolute import 'import {module}' - should use relative import")

        assert not errors, f"Import errors in {filename}:\n" + "\n".join(errors)

    def test_init_uses_relative_imports(self):
        """Verify __init__.py uses relative imports."""
        init_file = CUSTOM_NODE_ROOT / "__init__.py"
        content = init_file.read_text()

        # Must have relative imports
        assert "from .pyiqa_nodes import" in content, "__init__.py must use relative imports"
        assert "from .opencv_nodes import" in content, "__init__.py must use relative imports"
        assert "from .analysis_nodes import" in content, "__init__.py must use relative imports"


@pytest.mark.unit
class TestNodeMappings:
    """Verify __init__.py defines expected node mappings."""

    def test_node_mappings_defined(self):
        """Check NODE_CLASS_MAPPINGS is defined."""
        init_file = CUSTOM_NODE_ROOT / "__init__.py"
        content = init_file.read_text()

        assert "NODE_CLASS_MAPPINGS" in content
        assert "NODE_DISPLAY_NAME_MAPPINGS" in content

    def test_expected_nodes_registered(self):
        """Check all expected nodes are in NODE_CLASS_MAPPINGS."""
        init_file = CUSTOM_NODE_ROOT / "__init__.py"
        content = init_file.read_text()

        expected_nodes = [
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
        ]

        missing = []
        for node in expected_nodes:
            if f'"{node}"' not in content:
                missing.append(node)

        assert not missing, f"Missing nodes in NODE_CLASS_MAPPINGS: {missing}"
