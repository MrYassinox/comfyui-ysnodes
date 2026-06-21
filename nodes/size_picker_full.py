"""
nodes/size_picker_full.py
=============================
Size Picker Full — extends Size Picker with VAE encoding.

Difference from size_picker
--------------------------------
  One new optional input:
    • vae  — VAE model for encoding the image into latent space

  Two latent outputs instead of one:
    • latent_empty      — always an empty latent at final (w, h, batch).
                          Never uses VAE. Ideal for txt2img at a chosen size.
    • latent_from_image — VAE-encoded latent of the image resized to (w, h).
                          Falls back to empty latent when VAE is not connected.

Base width/height are computed from aspect_ratio + megapixels + multiple
(same formula as ComfyUI core's ResolutionSelector), then optionally
overridden by width_override/height_override or the connected image's
own dimensions — identical logic to size_picker_sg.
"""

import math
import torch
import torch.nn.functional as F
from enum import Enum

# ---------------------------------------------------------------------------
# Aspect ratios — covers print, social media, video/cinema, and AI-gen ratios
# ---------------------------------------------------------------------------

class AspectRatio(str, Enum):
    SQUARE              = "1:1 (Square)"
    PORTRAIT_SOCIAL     = "4:5 (Portrait Social)"
    PORTRAIT_PRINT      = "5:7 (Portrait Print)"
    PORTRAIT_PHOTO      = "2:3 (Portrait Photo)"
    PORTRAIT_STANDARD   = "3:4 (Portrait Standard)"
    PORTRAIT_TALL       = "5:12 (Portrait Tall)"
    PORTRAIT_WIDESCREEN = "9:16 (Portrait Widescreen)"
    PORTRAIT_ULTRAWIDE  = "9:21 (Portrait Ultrawide)"
    PORTRAIT_PANORAMIC  = "1:2 (Portrait Panoramic)"
    LANDSCAPE_SOCIAL    = "5:4 (Landscape Social)"
    LANDSCAPE_PRINT     = "7:5 (Landscape Print)"
    PHOTO               = "3:2 (Photo)"
    STANDARD            = "4:3 (Standard)"
    LANDSCAPE_WIDE      = "12:5 (Landscape Wide)"
    WIDESCREEN          = "16:9 (Widescreen)"
    ULTRAWIDE           = "21:9 (Ultrawide)"
    PANORAMIC           = "2:1 (Panoramic)"


ASPECT_RATIOS: dict = {
    AspectRatio.SQUARE:              (1, 1),
    AspectRatio.PORTRAIT_SOCIAL:     (4, 5),
    AspectRatio.PORTRAIT_PRINT:      (5, 7),
    AspectRatio.PORTRAIT_PHOTO:      (2, 3),
    AspectRatio.PORTRAIT_STANDARD:   (3, 4),
    AspectRatio.PORTRAIT_TALL:       (5, 12),
    AspectRatio.PORTRAIT_WIDESCREEN: (9, 16),
    AspectRatio.PORTRAIT_ULTRAWIDE:  (9, 21),
    AspectRatio.PORTRAIT_PANORAMIC:  (1, 2),
    AspectRatio.LANDSCAPE_SOCIAL:    (5, 4),
    AspectRatio.LANDSCAPE_PRINT:     (7, 5),
    AspectRatio.PHOTO:               (3, 2),
    AspectRatio.STANDARD:            (4, 3),
    AspectRatio.LANDSCAPE_WIDE:      (12, 5),
    AspectRatio.WIDESCREEN:          (16, 9),
    AspectRatio.ULTRAWIDE:           (21, 9),
    AspectRatio.PANORAMIC:           (2, 1),
}

_ASPECT_RATIO_KEYS = [e.value for e in AspectRatio]
_DEFAULT_ASPECT_RATIO = AspectRatio.SQUARE.value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_base_size(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    """Same formula as ComfyUI core ResolutionSelector."""
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


def _image_size(image) -> tuple[int, int]:
    _, h, w, _ = image.shape
    return w, h


def _apply_overrides(base_w, base_h, width_override, height_override, multiple=8):
    w = width_override  if width_override  > 0 else base_w
    h = height_override if height_override > 0 else base_h
    w = max(multiple, (w // multiple) * multiple)
    h = max(multiple, (h // multiple) * multiple)
    return w, h


def _empty_latent(width, height, batch_size) -> dict:
    samples = torch.zeros([batch_size, 4, height // 8, width // 8],
                          dtype=torch.float32)
    return {"samples": samples}


def _vae_encode(vae, image: torch.Tensor, target_w: int, target_h: int) -> dict:
    """Resize image to (target_w, target_h) then VAE-encode it."""
    samples = image.movedim(-1, 1)
    samples = F.interpolate(samples, size=(target_h, target_w),
                            mode="bicubic", antialias=True)
    samples = samples.movedim(1, -1)
    encoded = vae.encode(samples[:, :, :, :3])
    return {"samples": encoded}


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SizePickerFull:
    """
    Size Picker Full

    Identical to Size Picker with one extra optional input (vae)
    and two latent outputs instead of one:

      latent_empty      — empty latent at (w, h, batch). No VAE needed.
      latent_from_image — VAE-encoded image resized to (w, h).
                          Falls back to empty latent if VAE not connected.
    """

    CATEGORY = "YSNodes/utility"
    FUNCTION = "pick"

    RETURN_TYPES  = ("INT",   "INT",    "INT",   "LATENT",       "LATENT")
    RETURN_NAMES  = ("width", "height", "batch", "latent_empty", "latent_from_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    _ASPECT_RATIO_KEYS,
                    {
                        "default": _DEFAULT_ASPECT_RATIO,
                        "tooltip": "The aspect ratio used to compute width and height.",
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                        "tooltip": "Target total megapixels. 1.0 MP ≈ 1024×1024 for a square aspect ratio.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 8, "min": 8, "max": 128, "step": 4,
                        "tooltip": "Nearest multiple to round the computed width/height to. 8 = VAE-safe (recommended).",
                    },
                ),
                "width_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 8,
                     "tooltip": "Override width in pixels. 0 = use the value computed from aspect_ratio + megapixels."},
                ),
                "height_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 8,
                     "tooltip": "Override height in pixels. 0 = use the value computed from aspect_ratio + megapixels."},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "step": 1},
                ),
                "use_image_size": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "When True, width/height are taken from the connected image instead of the computed value."},
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Source image. Required when use_image_size=True. Also used for VAE encoding."},
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "VAE model for encoding. When not connected, latent_from_image falls back to empty latent."},
                ),
            },
        }

    # ------------------------------------------------------------------
    def pick(self, aspect_ratio, megapixels, multiple, width_override, height_override,
             batch_size, use_image_size, image=None, vae=None):

        # Shared base — computed from aspect_ratio + megapixels + multiple
        base_w, base_h = _calculate_base_size(aspect_ratio, megapixels, multiple)

        # ── Computed path (on_false) ────────────────────────────────────
        preset_w, preset_h = _apply_overrides(
            base_w, base_h, width_override, height_override, multiple
        )

        # ── Image path (on_true) ────────────────────────────────────────
        if image is not None:
            img_w, img_h = _image_size(image)
            image_w, image_h = _apply_overrides(base_w, base_h, img_w, img_h, multiple)
        else:
            image_w, image_h = preset_w, preset_h

        # ── Switch — same as size_picker_sg ────────────────────────────
        if use_image_size and image is not None:
            out_w, out_h = image_w, image_h
        else:
            out_w, out_h = preset_w, preset_h

        # ── latent_empty — always empty, never uses VAE ─────────────────
        latent_empty = _empty_latent(out_w, out_h, batch_size)

        # ── latent_from_image — VAE encode or empty fallback ─────────────
        if vae is not None and image is not None:
            latent_from_image = _vae_encode(vae, image, out_w, out_h)
        else:
            if vae is None and image is not None:
                print("[SizePickerFull] WARNING: VAE not connected — "
                      "latent_from_image is an empty latent.")
            latent_from_image = _empty_latent(out_w, out_h, batch_size)

        return (out_w, out_h, batch_size, latent_empty, latent_from_image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS       = {"SizePickerFull": SizePickerFull}
NODE_DISPLAY_NAME_MAPPINGS = {"SizePickerFull": "Size Picker Full"}
