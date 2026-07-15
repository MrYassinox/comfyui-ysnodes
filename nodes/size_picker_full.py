"""
nodes/size_picker_full.py
=============================
Size Picker Full — extends Size Picker with VAE encoding.

Same width/height logic as size_picker_sg, plus:
  • Optional `vae` input
  • Two latent outputs instead of one:
      latent_empty      — empty latent at the final (w, h, batch).
                          Never uses VAE. Ideal for txt2img.
      latent_from_image — VAE-encoded image resized to (w, h).
                          Falls back to empty latent if VAE not connected.

Two independent modes, selected by use_image_size
----------------------------------------------------
  use_image_size=False:
    width/height computed from aspect_ratio + megapixels + multiple

  use_image_size=True (image connected):
    fit_megapixels=False → image's exact pixel size, unchanged
    fit_megapixels=True  → image's own aspect ratio, resized to hit
                            the megapixels target

width_override / height_override (if > 0) always win over the computed
value, snapped to `multiple`. When no override is set, the computed/exact
value passes through with no extra snapping.
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
# Latent type registry — channels + spatial divisor per model architecture
# ---------------------------------------------------------------------------

LATENT_TYPES: dict[str, tuple[int, int]] = {
    "SD / SDXL":      (4,   8),   # SD 1.x, SD 2.x, SDXL
    "SD3 / AuraFlow": (16,  8),   # Stable Diffusion 3, AuraFlow
    "Flux":           (16,  8),   # Flux (standard)
    "Flux2":          (128, 16),  # Flux 2 (packed)
}

_LATENT_TYPE_KEYS    = list(LATENT_TYPES.keys())
_DEFAULT_LATENT_TYPE = "SD / SDXL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_size_from_ratio(w_ratio: float, h_ratio: float, megapixels: float, multiple: int) -> tuple[int, int]:
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


def _calculate_base_size(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    """Computed base for the aspect_ratio dropdown path."""
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    return _calculate_size_from_ratio(w_ratio, h_ratio, megapixels, multiple)


def _image_size(image) -> tuple[int, int]:
    _, h, w, _ = image.shape
    return w, h


def _apply_overrides(base_w, base_h, width_override, height_override, multiple=8):
    """
    Override rules:
      • override == 0  → keep base value EXACTLY as-is (no snapping)
      • override  > 0  → use override value, snapped to `multiple`
    """
    if width_override > 0:
        w = max(multiple, (width_override // multiple) * multiple)
    else:
        w = base_w

    if height_override > 0:
        h = max(multiple, (height_override // multiple) * multiple)
    else:
        h = base_h

    return w, h


def _empty_latent(width, height, batch_size,
                  latent_type: str = _DEFAULT_LATENT_TYPE) -> dict:
    """
    Return a latent dict matching the spatial format of the given model architecture.
      SD / SDXL      → 4ch,   pixels ÷  8
      SD3 / AuraFlow → 16ch,  pixels ÷  8
      Flux           → 16ch,  pixels ÷  8
      Flux2          → 128ch, pixels ÷ 16
    """
    channels, divisor = LATENT_TYPES[latent_type]
    samples = torch.zeros([batch_size, channels, height // divisor, width // divisor],
                          dtype=torch.float32)
    return {"samples": samples}


def _empty_image(width: int, height: int, batch_size: int) -> torch.Tensor:
    """Return a blank black IMAGE tensor (B, H, W, C) at the given size."""
    return torch.zeros([batch_size, height, width, 3], dtype=torch.float32)


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

    Same width/height logic as Size Picker (SG), plus an optional `vae`
    input and two latent outputs instead of one:

      latent_empty      — empty latent at (w, h, batch). No VAE needed.
      latent_from_image — VAE-encoded image resized to (w, h).
                          Falls back to empty latent if VAE not connected.
    """

    CATEGORY = "YSNodes/utility"
    FUNCTION = "pick"

    RETURN_TYPES  = ("IMAGE", "INT",   "INT",    "INT",   "LATENT",       "LATENT")
    RETURN_NAMES  = ("image", "width", "height", "batch", "latent_empty", "latent_from_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_image_size": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When True, width/height are taken from the "
                            "connected image instead of aspect_ratio + megapixels."
                        ),
                    },
                ),
                "fit_megapixels": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Only applies when use_image_size=True.\n"
                            "False → use the image's exact pixel size, unchanged.\n"
                            "True  → keep the image's aspect ratio but resize it "
                            "to hit the megapixels target below."
                        ),
                    },
                ),
                "aspect_ratio": (
                    _ASPECT_RATIO_KEYS,
                    {
                        "default": _DEFAULT_ASPECT_RATIO,
                        "tooltip": (
                            "The aspect ratio used to compute width and height. "
                            "Ignored when use_image_size=True."
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                        "tooltip": (
                            "Target total megapixels. Used for the aspect_ratio path, "
                            "and for the image path when fit_megapixels=True."
                        ),
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
                     "tooltip": "Override width in pixels. 0 = use the computed value."},
                ),
                "height_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 8,
                     "tooltip": "Override height in pixels. 0 = use the computed value."},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "step": 1},
                ),
                "latent_type": (
                    _LATENT_TYPE_KEYS,
                    {
                        "default": _DEFAULT_LATENT_TYPE,
                        "tooltip": (
                            "Latent format matching your model architecture.\n"
                            "SD / SDXL      → 4ch,   pixels ÷  8\n"
                            "SD3 / AuraFlow → 16ch,  pixels ÷  8\n"
                            "Flux           → 16ch,  pixels ÷  8\n"
                            "Flux2          → 128ch, pixels ÷ 16"
                        ),
                    },
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
    def pick(self, use_image_size, fit_megapixels, aspect_ratio, megapixels, multiple,
             width_override, height_override, batch_size, latent_type,
             image=None, vae=None):

        # ── Computed path (aspect_ratio + megapixels + multiple) ────────
        base_w, base_h = _calculate_base_size(aspect_ratio, megapixels, multiple)
        preset_w, preset_h = _apply_overrides(
            base_w, base_h, width_override, height_override, multiple
        )

        # ── Image path ────────────────────────────────────────────────
        if image is not None:
            img_w, img_h = _image_size(image)

            if fit_megapixels:
                fitted_w, fitted_h = _calculate_size_from_ratio(
                    img_w, img_h, megapixels, multiple
                )
            else:
                fitted_w, fitted_h = img_w, img_h

            image_w, image_h = _apply_overrides(
                fitted_w, fitted_h, width_override, height_override, multiple
            )
        else:
            image_w, image_h = preset_w, preset_h

        # ── Switch ────────────────────────────────────────────────────
        if use_image_size and image is not None:
            out_w, out_h = image_w, image_h
        else:
            out_w, out_h = preset_w, preset_h

        # ── latent_empty — always empty, never uses VAE ─────────────────
        latent_empty = _empty_latent(out_w, out_h, batch_size, latent_type)

        # ── latent_from_image — VAE encode or empty fallback ─────────────
        if vae is not None and image is not None:
            latent_from_image = _vae_encode(vae, image, out_w, out_h)
        else:
            if vae is None and image is not None:
                print("[SizePickerFull] WARNING: VAE not connected — "
                      "latent_from_image is an empty latent.")
            latent_from_image = _empty_latent(out_w, out_h, batch_size, latent_type)

        # ── image output: passthrough unchanged, or blank placeholder ───
        if image is not None:
            image_out = image
        else:
            image_out = _empty_image(out_w, out_h, batch_size)

        return (image_out, out_w, out_h, batch_size, latent_empty, latent_from_image)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS       = {"SizePickerFull": SizePickerFull}
NODE_DISPLAY_NAME_MAPPINGS = {"SizePickerFull": "Size Picker Full"}
