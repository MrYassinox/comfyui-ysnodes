"""
nodes/redraw_size.py
========================
Automatically Calculate Redraw Size

Subgraph pipeline (faithfully reproduced)
------------------------------------------
  GetImageSize(image)               → orig_w, orig_h
  simpleMath(orig_w * orig_h)       → total_pixels
  clamp_megapixels(total_pixels,    → target_mp  (node213 formula)
      min_pixel, max_pixel)
  ImageScaleToTotalPixels(image,    → image_resized
      upscale_method, target_mp)
  MaskToImage → ImageScaleToTotalPixels
      → ImageToMask                 → mask_resized  (same target_mp)
  GetImageSize(image_resized)       → resize_w, resize_h

Improvements over the original subgraph
-----------------------------------------
  • Preset COMBO for min / max MP — same UX as "Size Picker"
  • Override FLOAT inputs — non-zero value wins over the preset
  • Clear input / output names (snake_case, self-documenting)
  • Extra output: target_megapixels (FLOAT) for debugging / chaining
  • Mask is optional — zero masks returned when not connected
  • Snap-to-8 on output dimensions (VAE safe)
  • Pure Python / torch — zero external dependencies
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Upscale methods (mirrors ComfyUI core exactly)
# ---------------------------------------------------------------------------

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

# ---------------------------------------------------------------------------
# MP presets  —  label → exact float megapixels
# Formula: round(width * height / 1_048_576, 4)
# ---------------------------------------------------------------------------

MP_PRESETS: dict = {
    # ── SD 1.5 ──────────────────────────────────────────────────────────────
    "0.25 MP  ( 512x 512  - SD1.5)":   0.2500,
    "0.31 MP  ( 640x 512  - SD1.5)":   0.3125,
    "0.38 MP  ( 512x 768  - SD1.5)":   0.3750,
    "0.56 MP  ( 768x 768  - SD1.5)":   0.5625,
    # ── SDXL / Flux ─────────────────────────────────────────────────────────
    "0.77 MP  ( 896x 896  - SDXL)":    0.7656,
    "0.94 MP  (1536x 640  - SDXL)":    0.9375,
    "0.98 MP  (1216x 832  - SDXL)":    0.9648,
    "1.00 MP  (1024x1024  - SDXL)":    1.0000,
    "1.50 MP  (1024x1536  - Flux)":    1.5000,
    "1.56 MP  (1280x1280  - Flux)":    1.5625,
    # ── Larger ──────────────────────────────────────────────────────────────
    "1.98 MP  (1920x1080  - HD)":      1.9775,
    "2.25 MP  (1536x1536  - 2xSDXL)":  2.2500,
    "4.00 MP  (2048x2048  - 4K)":      4.0000,
    "8.29 MP  (3840x2160  - 4K UHD)":  7.9102,
}

_MP_PRESET_KEYS = list(MP_PRESETS.keys())
_DEFAULT_MIN_PRESET = "0.25 MP  ( 512x 512  - SD1.5)"
_DEFAULT_MAX_PRESET = "1.50 MP  (1024x1536  - Flux)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_image(image, target_w, target_h, method):
    """Scale a (B, H, W, C) IMAGE tensor to (target_w, target_h)."""
    samples = image.movedim(-1, 1)   # BHWC -> BCHW
    mode_map = {
        "lanczos":       "bicubic",
        "bicubic":       "bicubic",
        "bilinear":      "bilinear",
        "nearest-exact": "nearest",
        "area":          "area",
    }
    mode = mode_map.get(method, "bicubic")
    antialias = method in ("lanczos", "bicubic", "bilinear")
    resized = F.interpolate(samples, size=(target_h, target_w), mode=mode, antialias=antialias)
    return resized.movedim(1, -1)    # BCHW -> BHWC


def _scale_to_megapixels(image, megapixels, method):
    """
    Replicate ImageScaleToTotalPixels:
    scale so that w*h ~= megapixels * 1_048_576, preserving aspect ratio.
    Output dimensions snapped to multiples of 8.
    """
    _, h, w, _ = image.shape
    aspect = w / h
    target_total = megapixels * 1_048_576
    new_h = math.sqrt(target_total / aspect)
    new_w = new_h * aspect
    new_w = max(8, round(new_w / 8) * 8)
    new_h = max(8, round(new_h / 8) * 8)
    return _scale_image(image, new_w, new_h, method)


def _mask_to_image(mask):
    """MASK (B,H,W) or (H,W) -> IMAGE (B,H,W,3)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask.unsqueeze(-1).expand(-1, -1, -1, 3)


def _image_to_mask(image):
    """IMAGE (B,H,W,C) -> MASK (B,H,W) via red channel."""
    return image[..., 0]


def _resolve_mp(preset_key, override):
    """
    Override logic (mirrors nhknodes Size Picker pattern):
      override == 0.0  ->  use preset
      override  > 0.0  ->  use override value
    """
    if override > 0.0:
        return override
    return MP_PRESETS[preset_key]


def _clamp_megapixels(total_pixels, min_mp, max_mp):
    """
    Replicate the node213 formula exactly:
      (a<=c*MP)*c + ((a>c*MP)&(a<b*MP))*round(a/MP,2) + (a>=b*MP)*b
    Plain English: clamp(actual_mp, min_mp, max_mp), rounded to 2 dp.
    """
    MP = 1_048_576
    actual_mp = total_pixels / MP
    if actual_mp <= min_mp:
        return min_mp
    elif actual_mp >= max_mp:
        return max_mp
    else:
        return round(actual_mp, 2)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class RedrawSize:
    """
    Redraw Size Calculator

    Computes a target MP clamped between min and max, then scales both
    image and mask preserving aspect ratio.

    Min / max MP are chosen from presets (same style as Size Picker SG).
    Setting the override to any value > 0 overrides the preset for that bound.
    """

    CATEGORY = "YSNodes/image"
    FUNCTION = "calculate"

    RETURN_TYPES  = ("IMAGE", "MASK",  "IMAGE",          "MASK",          "INT",           "INT",            "INT",           "INT",             "FLOAT")
    RETURN_NAMES  = ("image_resized", "mask_resized", "image_original", "mask_original", "width_resized", "height_resized", "width_original", "height_original", "target_megapixels")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to resize for redrawing.",
                }),
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method used when scaling image and mask.",
                }),
                # ── Min MP ────────────────────────────────────────────────
                "min_preset": (_MP_PRESET_KEYS, {
                    "default": _DEFAULT_MIN_PRESET,
                    "tooltip": "Preset minimum MP. Images below this are upscaled. Ignored when min_megapixels_override > 0.",
                }),
                "min_megapixels_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.01, "round": 0.001,
                    "tooltip": "Override min preset. Set to 0 to use preset. Example: 0.26 ~ 512x512px.",
                }),
                # ── Max MP ────────────────────────────────────────────────
                "max_preset": (_MP_PRESET_KEYS, {
                    "default": _DEFAULT_MAX_PRESET,
                    "tooltip": "Preset maximum MP. Images above this are downscaled. Ignored when max_megapixels_override > 0.",
                }),
                "max_megapixels_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.01, "round": 0.001,
                    "tooltip": "Override max preset. Set to 0 to use preset. Example: 1.5 ~ 1024x1536px.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask resized to match image_resized exactly.",
                }),
            },
        }

    # ------------------------------------------------------------------
    def calculate(self, image, upscale_method,
                  min_preset, min_megapixels_override,
                  max_preset, max_megapixels_override,
                  mask=None):

        # ── Resolve min / max MP ────────────────────────────────────────
        min_mp = _resolve_mp(min_preset, min_megapixels_override)
        max_mp = _resolve_mp(max_preset, max_megapixels_override)
        if min_mp > max_mp:          # safety swap
            min_mp, max_mp = max_mp, min_mp

        # ── Original dimensions ─────────────────────────────────────────
        _, orig_h, orig_w, _ = image.shape
        total_pixels = orig_w * orig_h

        # ── Compute target MP (clamped) ─────────────────────────────────
        target_mp = _clamp_megapixels(total_pixels, min_mp, max_mp)

        # ── Scale image ─────────────────────────────────────────────────
        image_resized = _scale_to_megapixels(image, target_mp, upscale_method)
        _, res_h, res_w, _ = image_resized.shape

        # ── Scale mask (optional) ───────────────────────────────────────
        if mask is not None:
            mask_resized  = _image_to_mask(
                _scale_to_megapixels(_mask_to_image(mask), target_mp, upscale_method)
            )
            mask_original = mask
        else:
            mask_resized  = torch.zeros((image_resized.shape[0], res_h,  res_w),  dtype=torch.float32)
            mask_original = torch.zeros((image.shape[0],         orig_h, orig_w), dtype=torch.float32)

        return (
            image_resized,
            mask_resized,
            image,
            mask_original,
            res_w,
            res_h,
            orig_w,
            orig_h,
            float(target_mp),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RedrawSize": RedrawSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RedrawSize": "Redraw Size Calculator",
}
