# ComfyUI-Image-Quality-Assessment - Testing Guide

This guide covers running tests for the IQA custom node suite.

## Quick Start

```bash
cd C:\_stability_matrix\Data\Packages\Comfy-new\custom_nodes\ComfyUI-Image-Quality-Assessment

# Run all tests
..\..\venv\Scripts\python -m pytest tests/ -v

# Run only fast unit tests (no server needed)
..\..\venv\Scripts\python -m pytest tests/unit/ -m unit -v

# Run integration tests (requires ComfyUI running)
..\..\venv\Scripts\python -m pytest tests/integration/ -m integration -v
```

---

## Test Categories

### Unit Tests
Fast tests that validate core logic without requiring ComfyUI.

| Module | Description |
|--------|-------------|
| `test_aggregation.py` | Score aggregation (mean, max, etc.) and normalization |
| `test_caching.py` | Model caching (LRU) and hash generation |
| `test_metrics.py` | Tensor conversions and standalone metric logic |

### Integration Tests
Tests that run against a live ComfyUI server.

| Category | Description |
|----------|-------------|
| Node Registration | Verify all 25+ IQA nodes are properly registered |
| Workflow Fixtures | Load and validate workflow JSON files |
| Server Health | API endpoints and system stats |

---

## Test Markers

```bash
# Run by marker
pytest -m unit           # Fast, no server
pytest -m integration    # Requires ComfyUI
```

---

## Running Integration Tests

Integration tests require ComfyUI to be running. If it's not running, the test suite will attempt to start it automatically on port 8199.

```bash
# Manual Start (Terminal 1)
cd C:\_stability_matrix\Data\Packages\Comfy-new
venv\Scripts\python main.py

# Run tests (Terminal 2)
cd custom_nodes\ComfyUI-Image-Quality-Assessment
..\..\venv\Scripts\python -m pytest tests/integration/ -m integration -v
```

---

## Workflow Fixtures

Located in `tests/fixtures/workflows/`:

- `test_opencv_metrics.json`: Classical CV metrics
- `test_pyiqa_noreference.json`: Deep Learning metrics
- `test_analysis_nodes.json`: Advanced image analysis

---

## For Developers

### Adding Unit Tests
1. Create a file in `tests/unit/test_*.py`.
2. Mark class/test with `@pytest.mark.unit`.
3. Use the `sample_image_tensor` fixture if you need a dummy ComfyUI image.

### Adding Integration Tests
1. Create a file in `tests/integration/test_*.py`.
2. Mark with `@pytest.mark.integration`.
3. Use `api_client` fixture to interact with the server.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: folder_paths` | Ensure `tests/conftest.py` is mocking ComfyUI modules |
| Integration tests hang | Check ComfyUI is running at `http://127.0.0.1:8199` |
| `ImportError` in unit tests | Use `sys.path.insert(0, ...)` pattern in test file |
