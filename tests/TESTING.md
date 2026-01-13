# Testing Guide: ComfyUI-Image-Quality-Assessment

This repository uses a comprehensive test suite to ensure the stability of IQA nodes. The tests are designed to run both **standalone** (unit tests) and **integrated** (with a live ComfyUI server).

---

## 🚀 Quick Start

### 1. Configure VS Code (Recommended)
This repository includes a `.vscode/settings.json` that pre-configures the correct Python environment and test arguments.

1.  Open this folder in VS Code.
2.  Go to the **Testing** tab (flask icon).
3.  Click "Refresh Tests".
4.  Run any test directly from the UI.

> **Note:** If tests are not found, ensure your Python interpreter is set to the ComfyUI virtual environment: `venv\Scripts\python.exe`.

### 2. Run from Command Line
You can use the provided `run_tests.py` script which handles environment setup automatically.

```powershell
cd C:\_stability_matrix\Data\Packages\Comfy-new\custom_nodes\ComfyUI-Image-Quality-Assessment

# Run ALL tests
..\..\venv\Scripts\python run_tests.py

# Run ONLY Unit Tests (Fast, no server needed)
..\..\venv\Scripts\python run_tests.py -m unit

# Run ONLY Integration Tests (Requires ComfyUI running)
..\..\venv\Scripts\python run_tests.py -m integration
```

---

## 🧪 Test Suite Structure

The tests are split into two categories to ensure fast feedback loops while still verifying full functionality.

### Unit Tests (`tests/unit/`)
*   **Speed:** Fast (< 1s)
*   **Dependencies:** None (Standalone)
*   **Purpose:** Verify internal logic, math, and syntax without loading ComfyUI.

| File | Description |
| :--- | :--- |
| `test_logic_nodes.py` | Validates core node logic (Score Normalizer, Ensemble, Ranker, Filter). |
| `test_aggregation.py` | Checks score aggregation math and basic normalization. |
| `test_caching.py` | Verifies the caching mechanism (LRU eviction, hashing). |
| `test_metrics.py` | Tests tensor-to-numpy conversion and standalone metric math. |
| `test_syntax.py` | Ensures all files are valid Python and follow strict import rules. |

### Integration Tests (`tests/integration/`)
*   **Speed:** Slow (Requires network)
*   **Dependencies:** Running ComfyUI Server (Port 8188)
*   **Purpose:** Verify node registration, API endpoints, and workflow validity.

| Category | Description |
| :--- | :--- |
| **Node Registration** | Checks that all 25+ IQA nodes are correctly registered with the server. |
| **Server Health** | Verifies the ComfyUI API is reachable and healthy. |
| **Workflows** | Loads sample workflows (`fixtures/workflows/*.json`) to ensure they are valid. |

---

## 🛠 Developer Guide

### Writing Unit Tests
Unit tests must differ from standard Python tests because they cannot import the main package directly (due to relative imports in ComfyUI nodes).

**Pattern for testing nodes:**
Use the custom loader shown in `tests/unit/test_logic_nodes.py`:
```python
# Helper to load modules with relative imports
def load_module_into_package(filename, submodule_name):
    # ... (See test_logic_nodes.py for full helper)
```
This isolates the node logic from the rest of the ComfyUI system.

### Writing Integration Tests
Integration tests use the `api_client` fixture to talk to the server.
```python
@pytest.mark.integration
def test_my_node_exists(self, api_client):
    assert api_client.node_exists("MyNewNode")
```

### Running Integration Tests Manually
1.  **Terminal 1:** Start ComfyUI
    ```powershell
    cd C:\_stability_matrix\Data\Packages\Comfy-new
    venv\Scripts\python main.py
    ```
2.  **Terminal 2:** Run Tests
    ```powershell
    cd custom_nodes\ComfyUI-Image-Quality-Assessment
    ..\..\venv\Scripts\python run_tests.py -m integration
    ```

---

## 🧩 Architecture & Troubleshooting

### Why is the setup complex?
ComfyUI custom nodes use **relative imports** (e.g., `from . import node`). This works great inside ComfyUI but breaks standard `pytest` discovery because `pytest` doesn't treat the root folder as a package.

### Key Solutions Implemented
1.  **Test Runner (`run_tests.py`)**:
    *   Changes the working directory to `tests/`.
    *   This forces pytest to see `tests/` as the root, avoiding the "attempted relative import" error.
2.  **VS Code Configuration**:
    *   `"python.testing.cwd": "${workspaceFolder}/tests"`: Mimics the runner behavior.
    *   `"python.testing.pytestArgs": ["."]` : Tells pytest to look in the *current* directory (which is `tests/`).
    *   If you see "No tests found", verify these settings in `.vscode/settings.json`.
3.  **Lazy Imports in `conftest.py`**:
    *   The `requests` library is imported *inside* fixtures, not at the top level.
    *   This ensures Unit Tests can run even if `requests` is missing or the environment is minimal.

### Common Issues
*   **Error:** `ImportError: attempted relative import with no known parent package`
    *   **Fix:** Ensure you are running tests via `run_tests.py` or have VS Code configured to run from the `tests/` directory. Do **not** run `pytest` directly from the project root.
*   **Error:** `ModuleNotFoundError: No module named 'requests'`
    *   **Fix:** Ensure you are using the ComfyUI virtual environment (`venv\Scripts\python.exe`). Unit tests should still run, but integration tests will fail.
