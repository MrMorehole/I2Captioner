"""
ComfyUI Custom Node: I2Caption Loader
---------------------------------------
A drop-in replacement for Batch Load Images that re-scans the source
folder on every queue execution, so adding, removing, or replacing
images in the folder is picked up immediately without needing to
restart ComfyUI or manually refresh.

The force_refresh integer (wired from a RandomNoise seed or any
incrementing value) acts as a cache-buster — changing it tells
ComfyUI this node's output may have changed and must be re-evaluated.
Wire a RandomNoise seed or simply increment manually between runs.
"""

import os
import glob

import numpy as np
import torch
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def load_image_to_tensor(path: str) -> torch.Tensor:
    """Load a single image file and return a (1, H, W, C) float32 tensor."""
    img = Image.open(path).convert("RGBA")
    img = ImageOps.exif_transpose(img)          # respect EXIF rotation
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)   # (1, H, W, C)


class I2CaptionLoader:

    CATEGORY = "I2Caption"
    RETURN_TYPES  = ("IMAGE", "STRING", "INT")
    RETURN_NAMES  = ("image",  "filename", "total_images")
    FUNCTION = "load"
    OUTPUT_IS_LIST = (True, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "/mnt/More/ComfyUI/output/batch/images",
                        "multiline": False,
                        "placeholder": "Absolute path to image folder",
                    },
                ),
                "force_refresh": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "Change this value to force a folder re-scan. "
                            "Wire from a RandomNoise seed for automatic refresh every run."
                        ),
                    },
                ),
            },
            "optional": {
                "sort_by": (
                    ["name", "date_modified", "date_created"],
                    {"default": "name"},
                ),
                "reverse_order": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Newest first", "label_off": "Oldest first"},
                ),
            },
        }

    # IS_CHANGED tells ComfyUI to re-execute this node whenever
    # force_refresh changes value — this is the cache-busting mechanism.
    @classmethod
    def IS_CHANGED(cls, folder_path, force_refresh, sort_by="name", reverse_order=False):
        return force_refresh

    def load(self, folder_path, force_refresh, sort_by="name", reverse_order=False):
        if not os.path.isdir(folder_path):
            raise ValueError(f"[I2Caption Loader] Folder not found: {folder_path}")

        # Scan folder for supported image files
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

        # Deduplicate (glob can return duplicates on case-insensitive filesystems)
        all_files = list(set(all_files))

        if not all_files:
            raise ValueError(f"[I2Caption Loader] No supported images found in: {folder_path}")

        # Sort
        if sort_by == "date_modified":
            all_files.sort(key=lambda p: os.path.getmtime(p), reverse=reverse_order)
        elif sort_by == "date_created":
            all_files.sort(key=lambda p: os.path.getctime(p), reverse=reverse_order)
        else:
            all_files.sort(key=lambda p: os.path.basename(p).lower(), reverse=reverse_order)

        total = len(all_files)
        print(f"[I2Caption Loader] Found {total} images in {folder_path} (refresh={force_refresh})")

        images    = []
        filenames = []

        for path in all_files:
            fname = os.path.splitext(os.path.basename(path))[0]
            try:
                tensor = load_image_to_tensor(path)
                images.append(tensor)
                filenames.append(fname)
                print(f"[I2Caption Loader] Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"[I2Caption Loader] Warning: could not load {path}: {e}")

        if not images:
            raise RuntimeError(f"[I2Caption Loader] All images failed to load from: {folder_path}")

        return (images, filenames, total)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS        = {"I2Caption Loader": I2CaptionLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"I2Caption Loader": "I2Caption Loader"}
