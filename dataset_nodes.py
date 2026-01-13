import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

import folder_paths
import node_helpers
from .comfy_compat import io


def load_and_process_images(image_files, input_dir):
    """Utility function to load and process a list of images."""
    if not image_files:
        raise ValueError("No valid images found in input")

    output_images = []

    for file in image_files:
        image_path = os.path.join(input_dir, file)
        img = node_helpers.pillow(Image.open, image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        output_images.append(img_tensor)

    return output_images


def get_image_files_from_folder(folder, start_index, max_items, folder_path=None):
    if folder_path and os.path.exists(folder_path):
        sub_input_dir = folder_path
        logging.info(f"Loading images from custom path: {folder_path}")
    else:
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        logging.info(f"Loading images from folder: {folder}")

    valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    image_files = []

    try:
        items = sorted(os.listdir(sub_input_dir))
    except (FileNotFoundError, IndexError):
        logging.warning(f"Could not load images/find folder: {sub_input_dir}")
        return [], sub_input_dir

    for item in items:
        path = os.path.join(sub_input_dir, item)
        if any(item.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(item)
        elif os.path.isdir(path):
            # Support kohya-ss/sd-scripts folder structure
            repeat = 1
            if item.split("_")[0].isdigit():
                repeat = int(item.split("_")[0])

            try:
                # For nested folders, we need the relative path from sub_input_dir
                for f in sorted(os.listdir(path)):
                    if any(f.lower().endswith(ext) for ext in valid_extensions):
                        # Store relative path from sub_input_dir
                        relative_path = os.path.join(item, f)
                        image_files.extend([relative_path] * repeat)
            except (FileNotFoundError, IndexError):
                pass

    if max_items > 0:
        image_files = image_files[start_index : start_index + max_items]
    else:
        image_files = image_files[start_index:]

    return image_files, sub_input_dir


class LoadImageDataSetFromFolderNode_Custom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageDataSetFromFolder_Custom",
            display_name="Load Image Dataset (Custom)",
            category="IQA/Dataset",
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images from. Ignored if 'folder_path' is set.",
                ),
                io.String.Input(
                    "folder_path",
                    default="",
                    tooltip="Optional absolute path to a specific folder. If provided, this overrides the 'folder' selection.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    control_after_generate=True,
                    tooltip="Index of the first image to load (0-based). Files are sorted alphabetically.",
                ),
                io.Int.Input(
                    "max_items",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Maximum number of images to load. Set to 0 to load all images.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="List of loaded images",
                )
            ],
        )

    @classmethod
    def execute(cls, folder, start_index, max_items, folder_path):
        image_files, sub_input_dir = get_image_files_from_folder(
            folder, start_index, max_items, folder_path
        )
        output_tensor = load_and_process_images(image_files, sub_input_dir)
        return io.NodeOutput(output_tensor)


class LoadImageTextDataSetFromFolderNode_Custom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageTextDataSetFromFolder_Custom",
            display_name="Load Image and Text Dataset (Custom)",
            category="IQA/Dataset",
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images from. Ignored if 'folder_path' is set.",
                ),
                io.String.Input(
                    "folder_path",
                    default="",
                    tooltip="Optional absolute path to a specific folder. If provided, this overrides the 'folder' selection.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    control_after_generate=True,
                    tooltip="Index of the first image to load (0-based). Files are sorted alphabetically.",
                ),
                io.Int.Input(
                    "max_items",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Maximum number of images to load. Set to 0 to load all images.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="List of loaded images",
                ),
                io.String.Output(
                    display_name="texts",
                    is_output_list=True,
                    tooltip="List of text captions",
                ),
            ],
        )

    @classmethod
    def execute(cls, folder, start_index, max_items, folder_path):
        image_files, sub_input_dir = get_image_files_from_folder(
            folder, start_index, max_items, folder_path
        )

        caption_file_path = [
            f.replace(os.path.splitext(f)[1], ".txt") for f in image_files
        ]
        captions = []
        for caption_file in caption_file_path:
            caption_path = os.path.join(sub_input_dir, caption_file)
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                    captions.append(caption)
            else:
                captions.append("")

        output_tensor = load_and_process_images(image_files, sub_input_dir)

        logging.info(f"Loaded {len(output_tensor)} images from {sub_input_dir}.")
        return io.NodeOutput(output_tensor, captions)
