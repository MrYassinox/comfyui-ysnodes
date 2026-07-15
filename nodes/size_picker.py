"""
nodes/size_picker.py
========================
Size Picker — converted from the "Size Picker" subgraph.

Two independent modes, selected by use_image_size
----------------------------------------------------
  use_image_size=False:
    width/height computed from aspect_ratio + megapixels + multiple
    (same formula as ComfyUI core's ResolutionSelector)

  use_image_size=True (image connected):
    fit_megapixels=False → image's exact pixel size, unchanged
    fit_megapixels=True  → image's own aspect ratio, resized to hit
                            the megapixels target (multiple still applies)

In both modes, width_override / height_override (if > 0) always win over
the computed value, snapped to `multiple`. When no override is set, the
computed/exact value passes through with NO extra snapping — this matters
for fit_megapixels=False, where the image's exact pixels must stay exact.

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
# Helper – compute (width, height) from any w_ratio:h_ratio + megapixels
# (same formula as ComfyUI core ResolutionSelector)
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


# ---------------------------------------------------------------------------
# Helper – get image dimensions
# ---------------------------------------------------------------------------

def _image_size(image) -> tuple[int, int]:
    """Return (width, height) of a ComfyUI IMAGE tensor (B, H, W, C)."""
    _, h, w, _ = image.shape
    return w, h


# ---------------------------------------------------------------------------
# Helper – apply width/height overrides
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
      • override == 0  → keep base value EXACTLY as-is (no snapping)
      • override  > 0  → use override value, snapped to `multiple`

    Not snapping the base when there's no override matters for
    fit_megapixels=False: the image's exact pixel size must stay exact.
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


# ---------------------------------------------------------------------------
# Helper – build an EmptyLatentImage tensor (dynamic latent type)
# ---------------------------------------------------------------------------

def _empty_latent(width: int, height: int, batch_size: int,
                  latent_type: str = _DEFAULT_LATENT_TYPE) -> dict:
    """
    Return a latent dict matching the spatial format of the given model architecture.
      SD / SDXL      → 4ch,   pixels ÷  8
      SD3 / AuraFlow → 16ch,  pixels ÷  8
      Flux           → 16ch,  pixels ÷  8
      Flux2          → 128ch, pixels ÷ 16
    """
    channels, divisor = LATENT_TYPES[latent_type]
    latent = torch.zeros(
        [batch_size, channels, height // divisor, width // divisor],
        dtype=torch.float32,
    )
    return {"samples": latent}


# ---------------------------------------------------------------------------
# Helper – build a blank IMAGE tensor (fallback when no image is connected)
# ---------------------------------------------------------------------------

def _empty_image(width: int, height: int, batch_size: int) -> torch.Tensor:
    """Return a blank black IMAGE tensor (B, H, W, C) at the given size."""
    return torch.zeros([batch_size, height, width, 3], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SizePicker:
    """
    Size Picker

    use_image_size=False:
        width/height = aspect_ratio + megapixels + multiple (computed)

    use_image_size=True (image connected):
        fit_megapixels=False → image's exact pixel size, unchanged
        fit_megapixels=True  → image's own aspect ratio, resized to hit
                                the megapixels target

    width_override / height_override always win over the computed value
    when > 0 (snapped to multiple). aspect_ratio is ignored whenever
    use_image_size=True.

    Zero dependency on ComfyUI-nhknodes or any other external pack.
    """

    CATEGORY = "YSNodes/utility"
    FUNCTION = "pick"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "LATENT")
    RETURN_NAMES = ("image", "width", "height", "batch", "latent")

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
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": (
                            "Target total megapixels. Used for the aspect_ratio path, "
                            "and for the image path when fit_megapixels=True."
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
                        "tooltip": "Override width in pixels. 0 = use the computed value.",
                    },
                ),
                "height_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Override height in pixels. 0 = use the computed value.",
                    },
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
                "image": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------
    def pick(
        self,
        use_image_size: bool,
        fit_megapixels: bool,
        aspect_ratio: str,
        megapixels: float,
        multiple: int,
        width_override: int,
        height_override: int,
        batch_size: int,
        latent_type: str,
        image=None,
    ):
        # ── Computed path (aspect_ratio + megapixels + multiple) ────────
        base_w, base_h = _calculate_base_size(aspect_ratio, megapixels, multiple)
        preset_w, preset_h = _apply_overrides(
            base_w, base_h, width_override, height_override, multiple
        )
        preset_latent = _empty_latent(preset_w, preset_h, batch_size, latent_type)

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
            image_latent = _empty_latent(image_w, image_h, batch_size, latent_type)
        else:
            image_w, image_h = preset_w, preset_h
            image_latent = preset_latent

        # ── Switch ────────────────────────────────────────────────────
        if use_image_size and image is not None:
            out_w, out_h, out_latent = image_w, image_h, image_latent
        else:
            out_w, out_h, out_latent = preset_w, preset_h, preset_latent

        # ── image output: passthrough unchanged, or blank placeholder ───
        if image is not None:
            image_out = image
        else:
            image_out = _empty_image(out_w, out_h, batch_size)

        return (image_out, out_w, out_h, batch_size, out_latent)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SizePicker": SizePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SizePicker": "Size Picker",
}
