# Development Environment Setup

This repository includes a setup script to facilitate testing and development within a real ComfyUI environment.

## Quick Start

1.  Run the setup script:
    ```bash
    ./scripts/setup_dev_env.sh
    ```
    This will:
    *   Clone ComfyUI to `.comfyui_core/`.
    *   Install all dependencies (CPU-optimized by default if no GPU found).
    *   Symlink this repository into `.comfyui_core/custom_nodes/`.
    *   Generate `env_setup.sh`.

2.  Activate the environment:
    ```bash
    source env_setup.sh
    ```

3.  Run tests or Python scripts:
    ```bash
    python my_test_script.py
    ```

## Why use this?

This setup allows you to import `comfy` modules (like `comfy.sd`, `comfy_api`) directly, enabling:
*   Testing V3 Schema nodes (`io.ComfyNode`) without shims.
*   Verifying integration with ComfyUI's core logic.
*   Running automated integration tests.
