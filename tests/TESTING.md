# ComfyUI-Image-Quality-Assessment Testing Guide

## Quick Start

```bash
cd C:\_stability_matrix\Data\Packages\Comfy-new\custom_nodes\ComfyUI-Image-Quality-Assessment

# Run all tests via the test runner script
..\..\venv\Scripts\python run_tests.py

# Run only unit tests (no ComfyUI server needed)
..\..\venv\Scripts\python run_tests.py -m unit

# Run integration tests (requires ComfyUI on port 8188)
..\..\venv\Scripts\python run_tests.py -m integration
```

---

## Test Categories

### Unit Tests
Fast tests that validate utility functions without ComfyUI.

| Module | Tests | Description |
|--------|-------|-------------|
| `test_aggregation.py` | 11 | Score aggregation, normalization logic |
| `test_caching.py` | 5 | ModelCache LRU, hash generation |
| `test_metrics.py` | 2 | Tensor conversion, colorfulness math |
| `test_syntax.py` | 12 | File syntax validation, __init__.py structure |

### Integration Tests
Tests that run against a live ComfyUI server.

| Category | Tests | Description |
|----------|-------|-------------|
| Node Registration | 25 | Verify all IQA nodes are registered |
| Server Health | 2 | API endpoints, node counts |
| Workflow Fixtures | 3 | Load and validate workflow JSON files |

---

## Test Markers

```bash
python run_tests.py -m unit           # Fast, no server
python run_tests.py -m integration    # Requires ComfyUI
```

---

## Running Integration Tests

Integration tests require ComfyUI to be running:

```bash
# Terminal 1: Start ComfyUI
cd C:\_stability_matrix\Data\Packages\Comfy-new
venv\Scripts\python main.py

# Terminal 2: Run tests
cd custom_nodes\ComfyUI-Image-Quality-Assessment
..\..\venv\Scripts\python run_tests.py -m integration
```

---

## For Developers

### Adding Unit Tests

1. Create test file in `tests/unit/test_*.py`
2. Mark tests with `@pytest.mark.unit`
3. Import modules using `importlib` to bypass package `__init__.py`

```python
import pytest
import sys
import importlib.util
from pathlib import Path

# Load module directly
custom_node_root = Path(__file__).parent.parent.parent
iqa_core_path = custom_node_root / "utils" / "iqa_core.py"
spec = importlib.util.spec_from_file_location("iqa_core", iqa_core_path)
iqa_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iqa_core)

@pytest.mark.unit
class TestMyFeature:
    def test_something(self):
        result = iqa_core.aggregate_scores([1, 2, 3], "mean")
        assert result == 2.0
```

### Adding Integration Tests

1. Create test in `tests/integration/test_*.py`
2. Mark with `@pytest.mark.integration`
3. Use provided fixtures: `api_client`, `workflow_fixtures_path`

```python
@pytest.mark.integration
def test_node_registered(self, api_client):
    assert api_client.node_exists("PyIQA_NoReferenceNode")
```

---

## Architectural Notes & Roadblocks

This testing suite encountered several critical roadblocks during implementation that are documented here for future maintainers:

### 1. Pytest Package Discovery vs. ComfyUI
The main `__init__.py` uses relative imports (`from .pyiqa_nodes import ...`). If `pytest` is run from the project root, it attempts to import the package, which fails with `ImportError: attempted relative import with no known parent package`.

**Solution:** The `run_tests.py` runner changes directory to `tests/` before executing pytest. This prevents pytest from treating the root as a package and discovering the root `__init__.py`.

### 2. Parent pytest.ini Interference
In some environments (like Stability Matrix), a parent `pytest.ini` might exist with `pythonpath = .`. This forces pytest to add the parent directory to `sys.path`, again triggering package discovery issues.

**Solution:** A local `tests/pytest.ini` with `pythonpath = ` (empty) overrides any parent settings and ensures a clean test environment.

### 3. Module Dependency Isolation
Most node modules in this repo depend on `comfy_compat.py`. Standalone unit tests can only safely test the `utils/iqa_core.py` module, which is kept free of ComfyUI dependencies.

### 4. Implementation summary
*   **TTS Pattern:** This repo strictly follows the `run_tests.py` and `tests/` isolation pattern established in TTS-Audio-Suite.
*   **Env Check:** Do **not** use `COMFYUI_TESTING` inside `__init__.py` to skip imports; this breaks node registration in the actual ComfyUI server. Isolation must be handled at the test runner level.
