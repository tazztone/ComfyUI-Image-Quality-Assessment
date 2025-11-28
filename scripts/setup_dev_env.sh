#!/bin/bash
set -e

# Directory to install ComfyUI Core
COMFY_DIR=".comfyui_core"
REPO_DIR=$(pwd)

echo ">>> Setting up ComfyUI Development Environment..."

# 1. Clone ComfyUI if not exists
if [ -d "$COMFY_DIR" ]; then
    echo ">>> ComfyUI directory exists. Pulling latest changes..."
    cd "$COMFY_DIR"
    git reset --hard HEAD
    git pull
    cd "$REPO_DIR"
else
    echo ">>> Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    cd "$COMFY_DIR"
    git reset --hard HEAD
    cd "$REPO_DIR"
fi

# 2. Install Dependencies
echo ">>> Installing Dependencies..."
# Detect if CUDA is available, otherwise install CPU versions to be lightweight
if ! command -v nvidia-smi &> /dev/null; then
    echo ">>> CUDA not found. Installing CPU-only PyTorch and dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo ">>> CUDA found. Installing standard dependencies..."
fi

# Install ComfyUI requirements
pip install -r "$COMFY_DIR/requirements.txt"

# Install Custom Node requirements
if [ -f "requirements.txt" ]; then
    echo ">>> Installing Custom Node requirements..."
    pip install -r requirements.txt --only-binary=:all:
fi

# 3. Link Custom Node
echo ">>> Linking Custom Node..."
NODES_DIR="$COMFY_DIR/custom_nodes"
# Use a fixed valid package name for the link to ensure importability
TARGET_LINK="$NODES_DIR/ComfyUI_Image_Quality_Assessment"

# Remove existing link/dir if exists to ensure clean link
if [ -e "$TARGET_LINK" ] || [ -L "$TARGET_LINK" ]; then
    rm -rf "$TARGET_LINK"
fi

ln -s "$REPO_DIR" "$TARGET_LINK"
echo ">>> Symlinked $REPO_DIR to $TARGET_LINK"

# 4. Generate Activation Helper
echo ">>> Generating env_setup.sh..."
cat <<EOF > env_setup.sh
# Add ComfyUI to python path
export PYTHONPATH="\$PYTHONPATH:$REPO_DIR/$COMFY_DIR"
# Add custom_nodes to python path so we can import the node package
export PYTHONPATH="\$PYTHONPATH:$REPO_DIR/$COMFY_DIR/custom_nodes"
echo "ComfyUI Environment Loaded. V3 API should be available."
EOF

# At the end of setup script, before final message:
echo ">>> Cleaning git working tree..."
git add .gitignore 2>/dev/null || true
git reset --hard HEAD

echo ">>> Setup Complete!"


echo ">>> Setup Complete!"
echo "Run 'source env_setup.sh' to activate the environment for testing."
