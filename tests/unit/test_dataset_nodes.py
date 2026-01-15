"""
Unit tests for dataset nodes.
Mocks ComfyUI internals (folder_paths, node_helpers) to verify file finding and data loading logic.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

# ===================================================================================
# IMPORT HELPER
# ===================================================================================
TEST_PKG_NAME = "iqa_test_pkg_dataset"
CUSTOM_NODE_ROOT = Path(__file__).parent.parent.parent


def load_module_into_package(filename, submodule_name):
    """Load a file as a submodule of our dummy package."""
    full_pkg_name = f"{TEST_PKG_NAME}.{submodule_name}"
    filepath = CUSTOM_NODE_ROOT / filename

    spec = importlib.util.spec_from_file_location(full_pkg_name, filepath)
    if spec is None:
        raise ImportError(f"Could not load {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_pkg_name] = module
    spec.loader.exec_module(module)
    return module


# ===================================================================================
# SETUP: MOCK COMFYUI DEPENDENCIES
# ===================================================================================

# 1. Create dummy package
if TEST_PKG_NAME not in sys.modules:
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(TEST_PKG_NAME, None, is_package=True)
    )
    sys.modules[TEST_PKG_NAME] = pkg

# 2. Mock external dependencies (folder_paths, node_helpers)
# We must mock these BEFORE loading dataset_nodes because it imports them at top level.
mock_folder_paths = MagicMock()
mock_node_helpers = MagicMock()

sys.modules["folder_paths"] = mock_folder_paths
sys.modules["node_helpers"] = mock_node_helpers

# 3. Load local dependencies
# dataset_nodes imports from .comfy_compat
comfy_compat = load_module_into_package("comfy_compat.py", "comfy_compat")

# 4. Load dataset_nodes
dataset_nodes = load_module_into_package("dataset_nodes.py", "dataset_nodes")

# Extract functions and classes
get_image_files_from_folder = dataset_nodes.get_image_files_from_folder
LoadImageDataSetNode = dataset_nodes.LoadImageDataSetFromFolderNode_Custom
LoadImageTextDataSetNode = dataset_nodes.LoadImageTextDataSetFromFolderNode_Custom


@pytest.fixture
def mock_filesystem(tmp_path):
    """Create a temporary directory structure for testing."""
    # Create structure:
    # /root
    #   /my_images
    #     img1.png
    #     img2.jpg
    #     nested/
    #       img3.webp
    #   /empty_folder

    root = tmp_path / "comfy_input"
    root.mkdir()

    my_images = root / "my_images"
    my_images.mkdir()
    (my_images / "img1.png").touch()
    (my_images / "img2.jpg").touch()
    (my_images / "not_an_image.txt").touch()

    nested = my_images / "nested"
    nested.mkdir()
    (nested / "img3.webp").touch()

    empty = root / "empty_folder"
    empty.mkdir()

    return root


@pytest.mark.unit
class TestGetImageFiles:
    def test_basic_flat_folder(self, mock_filesystem):
        """Test finding images in a flat directory."""
        mock_folder_paths.get_input_directory.return_value = str(mock_filesystem)

        folder_name = "my_images"
        files, path = get_image_files_from_folder(folder_name, 0, 0, folder_path="")

        # Should find img1.png and img2.jpg
        # img3.webp is in nested, usually not found unless recursive logic is triggered specifically
        # But wait, looking at code: it iterates subdirs!
        # "elif os.path.isdir(path): ... for f in sorted(os.listdir(path))"
        # So it IS recursive for one level depth.

        # Expected: img1.png, img2.jpg, nested/img3.webp (if logic supports it)
        # Let's check the code logic again:
        # It appends relative path "item/f" so "nested\\img3.webp" (win) or "nested/img3.webp" (linux)

        # We check simply that we got 3 images
        # Default recursive=False, so only top level files (img1.png, img2.jpg)
        assert len(files) == 2
        assert "img1.png" in files
        assert "img2.jpg" in files

        # Test with recursive=True
        files_rec, _ = get_image_files_from_folder(
            folder_name, 0, 0, folder_path="", recursive=True
        )
        assert len(files_rec) == 3
        assert any("img3.webp" in f for f in files_rec)

    def test_start_index(self, mock_filesystem):
        """Test skipping images."""
        mock_folder_paths.get_input_directory.return_value = str(mock_filesystem)

        folder_name = "my_images"
        # sorted: img1.png, img2.jpg, nested/...
        files, _ = get_image_files_from_folder(folder_name, start_index=1, max_items=0)

        # Should skip first one
        # Should skip first one. Total 2 files (non-recursive), so 1 left.
        assert len(files) == 1
        assert "img1.png" not in files

    def test_max_items(self, mock_filesystem):
        """Test limiting result count."""
        mock_folder_paths.get_input_directory.return_value = str(mock_filesystem)

        folder_name = "my_images"
        files, _ = get_image_files_from_folder(folder_name, start_index=0, max_items=1)

        assert len(files) == 1
        assert "img1.png" in files

    def test_absolute_path_override(self, mock_filesystem):
        """Test providing an explicit 'folder_path'."""
        override_path = mock_filesystem / "my_images" / "nested"

        # Should only find things in 'nested' (img3.webp)
        # NOTE: get_input_directory won't be called or used for base if folder_path provided
        files, used_path = get_image_files_from_folder(
            "garbage", 0, 0, folder_path=str(override_path), recursive=False
        )

        assert str(used_path) == str(override_path)
        assert len(files) == 1
        assert "img3.webp" in files[0]

    def test_invalid_folder_graceful(self):
        """Test handling of non-existent folders."""
        mock_folder_paths.get_input_directory.return_value = "/tmp/does_not_exist"

        files, _ = get_image_files_from_folder("bad_folder", 0, 0)
        assert files == []


@pytest.mark.unit
class TestLoadImageNodes:
    def test_dataset_node_structure(self, mock_filesystem):
        """Test execution flow of LoadImageDataSetFromFolderNode_Custom."""
        mock_folder_paths.get_input_directory.return_value = str(mock_filesystem)

        # Mock pillow opening images
        # We need to mock Image.open to return a valid dummy image
        with patch("PIL.Image.open"):
            mock_img = MagicMock()
            mock_img.mode = "RGB"
            mock_img.convert.return_value = mock_img
            # Mock numpy conversion
            # dataset_nodes does: np.array(img).astype...
            # We can patch os.path.join or just let it fail?
            # It actually reads the file.

            # Better approach: The function `load_and_process_images` attempts to open file.
            # We should probably mock `load_and_process_images` inside dataset_nodes if we want to isolate Node logic vs File logic
            # But the Node calls `get_image_files...` then `load_and_process_images`.
            pass

        # Let's test the helper `load_and_process_images` separately by mocking its internals if needed
        # Or mock `dataset_nodes.load_and_process_images`

        # Mocking the internal function call in the module
        original_loader = dataset_nodes.load_and_process_images
        dataset_nodes.load_and_process_images = MagicMock(return_value="TENSOR_RESULT")

        try:
            res = LoadImageDataSetNode.execute("my_images", 0, 0, "", recursive=True)
            # Expect NodeOutput wrapper which is a tuple
            assert res[0] == "TENSOR_RESULT"

            # Verify it was called with correct file list
            args, _ = dataset_nodes.load_and_process_images.call_args
            assert len(args[0]) == 3  # 3 images found
        finally:
            dataset_nodes.load_and_process_images = original_loader

    def test_text_dataset_loading(self, mock_filesystem):
        """Test loading images AND text captions."""
        mock_folder_paths.get_input_directory.return_value = str(mock_filesystem)

        # Create a caption file for img1.png
        # my_images/img1.txt
        img_dir = mock_filesystem / "my_images"
        (img_dir / "img1.txt").write_text("A cute cat", encoding="utf-8")

        # Mock the image loader part to skip actual image loading
        original_loader = dataset_nodes.load_and_process_images
        dataset_nodes.load_and_process_images = MagicMock(return_value="TENSOR_RESULT")

        try:
            # We assume alphabetically sorted: img1.png, img2.jpg, img3.webp
            # So index 0 is img1
            res = LoadImageTextDataSetNode.execute(
                "my_images", 0, 0, "", recursive=True
            )

            # Output is (tensor, list_of_texts) inside NodeOutput
            # NodeOutput is a tuple

            # res[0] is the output_tensor
            texts = res[1]

            assert len(texts) == 3
            assert texts[0] == "A cute cat"
            assert texts[1] == ""  # img2 has no text

        finally:
            dataset_nodes.load_and_process_images = original_loader
