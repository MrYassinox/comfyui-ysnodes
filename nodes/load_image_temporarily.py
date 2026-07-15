"""
nodes/load_image_temporarily.py
====================================
Load Image Temporarily
— converted from a community "Load Image Temporarily" node (originally
  published under "⚡ MNeMiC Nodes"), with a default-directory fix and
  clearer validation messages.

What it does
------------
Lists image files already present in ComfyUI's /temp folder (instead of
/input) and loads the selected one — same core logic as the built-in
LoadImage node: EXIF transpose, multi-frame (GIF/MPO) support, alpha-channel
mask extraction.

A companion JS file (web/load_image_temporarily_sg.js) adds a
"choose file to upload" button that uploads new files directly into /temp
(type=temp), so this node never writes to /input.

Bug fix vs the original reference code
-----------------------------------------
The original called folder_paths.get_annotated_filepath(image) and
folder_paths.exists_annotated_filepath(image) with no default_dir.
ComfyUI's folder_paths defaults to the INPUT directory when a filename has
no "[temp]"/"[input]"/"[output]" annotation suffix — which is the case for
every file already sitting in /temp at node-creation time (the dropdown
lists raw filenames, unannotated). Without the fix, selecting any
pre-existing temp file would silently look in the wrong folder.

Verified against ComfyUI core source (folder_paths.py):
  - get_annotated_filepath(name, default_dir=None)  → supports default_dir
  - exists_annotated_filepath(name)                 → does NOT support it

So get_annotated_filepath(image, default_dir=temp_dir) is used everywhere
a path is resolved, and existence is checked directly with os.path.exists()
on that resolved path instead of calling the (non-overridable)
exists_annotated_filepath().
"""

import hashlib
import os

import comfy.model_management
import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence


def _temp_dir() -> str:
    d = folder_paths.get_temp_directory()
    os.makedirs(d, exist_ok=True)
    return d


def _list_temp_images() -> list:
    """List image files currently in the temp folder, blank-first."""
    temp_dir = _temp_dir()
    files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
    files = folder_paths.filter_files_content_types(files, ["image"])
    # Keep new node instances clear by default instead of preselecting
    # whichever temp file currently exists first.
    return [""] + sorted(files)


class LoadImageTemporarily:
    """
    Load Image Temporarily

    Loads an image from ComfyUI's /temp folder instead of /input — useful
    for throwaway reference images you don't want cluttering your permanent
    input library.

    Use the "choose file to upload" button (added by the companion JS file)
    to upload a new image directly into /temp, or pick an existing /temp
    file from the dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_list_temp_images(), {
                    "tooltip": (
                        "Image file in ComfyUI's /temp folder. "
                        "Use the upload button to add a new one."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_TOOLTIPS = (
        "The loaded image tensor.",
        "The image alpha mask (inverted like ComfyUI LoadImage).",
        "Image width.",
        "Image height.",
    )
    FUNCTION = "load_image"
    CATEGORY = "YSNodes/image"
    DESCRIPTION = "Loads an image from ComfyUI's /temp folder instead of /input."

    def load_image(self, image):
        if not image:
            raise ValueError(
                "[LoadImageTemporarily] No image selected. "
                "Use the upload button or pick a file from the dropdown."
            )

        image_path = folder_paths.get_annotated_filepath(image, default_dir=_temp_dir())
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None
        dtype = comfy.model_management.intermediate_dtype()

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 255))
            image_rgb = frame.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if "A" in frame.getbands():
                mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif frame.mode == "P" and "transparency" in frame.info:
                mask = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor.to(dtype=dtype))
            output_masks.append(mask.unsqueeze(0).to(dtype=dtype))

            if img.format == "MPO":
                break

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, w, h)

    @classmethod
    def IS_CHANGED(cls, image):
        if not image:
            return ""
        image_path = folder_paths.get_annotated_filepath(image, default_dir=_temp_dir())
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image:
            return "No image selected. Upload a file or pick one from the dropdown."
        # exists_annotated_filepath() does NOT support default_dir, so we
        # resolve the path ourselves with get_annotated_filepath() (which does)
        # and check existence directly.
        image_path = folder_paths.get_annotated_filepath(image, default_dir=_temp_dir())
        if not os.path.exists(image_path):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "LoadImageTemporarily": LoadImageTemporarily,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageTemporarily": "Load Image Temporarily",
}
