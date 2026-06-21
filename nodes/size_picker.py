"""
nodes/size_picker.py
========================
Size Picker — converted from the "Size Picker" subgraph.

Logic summary
-------------
- Base width/height are computed from aspect_ratio + megapixels + multiple,
  using the same formula as ComfyUI core's ResolutionSelector:
      total_pixels = megapixels * 1024 * 1024
      scale        = sqrt(total_pixels / (w_ratio * h_ratio))
      width        = round(w_ratio * scale / multiple) * multiple
      height       = round(h_ratio * scale / multiple) * multiple
- Two parallel paths share that same base:
    • preset path (on_false): base + user width/height overrides
    • image path  (on_true):  base + image pixel dims as overrides
- Switch reads use_image_size:
    • False → preset path
    • True  → image path (falls back to preset path if no image connected)
- The latent is switched independently, just like width and height.

Zero dependency on ComfyUI-nhknodes or any other external pack.
"""

import math
import torch
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
# Helper – compute base (width, height) from aspect_ratio + megapixels
# (same formula as ComfyUI core ResolutionSelector)
# ---------------------------------------------------------------------------

def _calculate_base_size(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


# ---------------------------------------------------------------------------
# Helper – get image dimensions
# ---------------------------------------------------------------------------

def _image_size(image) -> tuple[int, int]:
    """Return (width, height) of a ComfyUI IMAGE tensor (B, H, W, C)."""
    _, h, w, _ = image.shape
    return w, h


# ---------------------------------------------------------------------------
# Helper – apply width/height overrides, snapped to `multiple`
# ---------------------------------------------------------------------------

def _apply_overrides(
    base_w: int,
    base_h: int,
    width_override: int,
    height_override: int,
    multiple: int = 8,
) -> tuple[int, int]:
    """
    Override rules:
      • override == 0  → keep base value
      • override  > 0  → use override value (snapped to `multiple`)
    """
    w = (width_override  if width_override  > 0 else base_w)
    h = (height_override if height_override > 0 else base_h)
    w = max(multiple, (w // multiple) * multiple)
    h = max(multiple, (h // multiple) * multiple)
    return w, h


# ---------------------------------------------------------------------------
# Helper – build an EmptyLatentImage tensor
# ---------------------------------------------------------------------------

def _empty_latent(width: int, height: int, batch_size: int) -> dict:
    """Return a latent dict identical to EmptyLatentImage output."""
    latent = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        dtype=torch.float32,
    )
    return {"samples": latent}


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SizePicker:
    """
    Size Picker

    Computes width/height from an aspect ratio + target megapixels + a
    rounding multiple (same formula as ComfyUI core's ResolutionSelector),
    then optionally switches to the connected image's own dimensions.

    width_override / height_override non-zero values always win over the
    computed base, snapped to `multiple`.

    Zero dependency on ComfyUI-nhknodes or any other external pack.
    """

    CATEGORY = "YSNodes/utility"
    FUNCTION = "pick"
    RETURN_TYPES = ("INT", "INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "batch", "latent")

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
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": (
                            "Target total megapixels. "
                            "1.0 MP ≈ 1024×1024 for a square aspect ratio."
                        ),
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 4,
                        "tooltip": (
                            "Nearest multiple to round the computed width/height to. "
                            "8 = VAE-safe (recommended)."
                        ),
                    },
                ),
                "width_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 8,
                        "tooltip": (
                            "Override width in pixels. 0 = use the value computed "
                            "from aspect_ratio + megapixels."
                        ),
                    },
                ),
                "height_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 8,
                        "tooltip": (
                            "Override height in pixels. 0 = use the value computed "
                            "from aspect_ratio + megapixels."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "step": 1},
                ),
                "use_image_size": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When True, width/height are taken from the "
                            "connected image instead of the computed value."
                        ),
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------
    def pick(
        self,
        aspect_ratio: str,
        megapixels: float,
        multiple: int,
        width_override: int,
        height_override: int,
        batch_size: int,
        use_image_size: bool,
        image=None,
    ):
        # Shared base — computed from aspect_ratio + megapixels + multiple
        base_w, base_h = _calculate_base_size(aspect_ratio, megapixels, multiple)

        # ── Computed path → on_false of all switches ────────────────────
        preset_w, preset_h = _apply_overrides(
            base_w, base_h, width_override, height_override, multiple
        )
        preset_latent = _empty_latent(preset_w, preset_h, batch_size)

        # ── Image path → on_true of all switches ────────────────────────
        # Image pixel dims feed in as overrides on top of the same base.
        # Falls back silently to the computed path when no image is connected.
        if image is not None:
            img_w, img_h = _image_size(image)
            image_w, image_h = _apply_overrides(
                base_w, base_h, img_w, img_h, multiple
            )
            image_latent = _empty_latent(image_w, image_h, batch_size)
        else:
            image_w, image_h = preset_w, preset_h
            image_latent = preset_latent

        # ── Switch — width / height / latent ────────────────────────────
        # use_image_size=False → computed path
        # use_image_size=True  → image path
        if use_image_size and image is not None:
            out_w, out_h, out_latent = image_w, image_h, image_latent
        else:
            out_w, out_h, out_latent = preset_w, preset_h, preset_latent

        return (out_w, out_h, batch_size, out_latent)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SizePicker": SizePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SizePicker": "Size Picker",
}
