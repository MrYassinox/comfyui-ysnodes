"""
nodes/image_scale_to_norm_pixels.py
=========================================
ImageScaleToNormPixels
— duplicated from ImageScaleTotalPixels, with VAE encoding removed.

Pipeline (two sequential steps, both optional independently)
----------------------------------------------------------------
  Step 1 — scale_by (inspired by ImageScaleToNormPixels):
    scale_by == 1.0  → pixels unchanged, byte-for-byte
    scale_by != 1.0  → image's current size × scale_by, snapped to `multiple`

  Step 2 — megapixels (original ImageScaleToTotalPixels logic):
    enable_megapixels=False → result of step 1 passes through unchanged
    enable_megapixels=True  → result of step 1 is further scaled to hit
                               the megapixels target, snapped to `multiple`

  The mask path (if connected) mirrors the image path exactly, using the
  same scale_by / megapixels / multiple values, so image and mask always
  end up at matching dimensions.

Difference from ImageScaleTotalPixels
------------------------------------------
  • No vae input
  • No latent_image / latent_mask outputs
  • Outputs are just: image, mask

Fallbacks
---------
  mask not connected → mask output = zeros

Zero external dependencies.
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_image(image: torch.Tensor, target_w: int, target_h: int, method: str) -> torch.Tensor:
    """Scale (B, H, W, C) to (target_w, target_h)."""
    samples = image.movedim(-1, 1)                        # BHWC → BCHW
    mode_map = {
        "lanczos":       "bicubic",
        "bicubic":       "bicubic",
        "bilinear":      "bilinear",
        "nearest-exact": "nearest",
        "area":          "area",
    }
    mode = mode_map.get(method, "bicubic")
    antialias = method in ("lanczos", "bicubic", "bilinear")
    resized = F.interpolate(
        samples, size=(target_h, target_w), mode=mode, antialias=antialias
    )
    return resized.movedim(1, -1)                         # BCHW → BHWC


def _scale_by_factor(image: torch.Tensor, scale_by: float, method: str, multiple: int) -> torch.Tensor:
    """
    Multiply the image's current size by scale_by, preserving aspect ratio.
    scale_by == 1.0 is handled by the caller (no-op, no snapping at all) —
    this function is only called when scale_by != 1.0.
    """
    _, h, w, _ = image.shape
    new_w = max(multiple, round(w * scale_by / multiple) * multiple)
    new_h = max(multiple, round(h * scale_by / multiple) * multiple)
    return _scale_image(image, new_w, new_h, method)


def _scale_to_megapixels(image: torch.Tensor, megapixels: float, method: str, multiple: int) -> torch.Tensor:
    """
    Replicate ImageScaleToTotalPixels (comfy-core):
    scale so w×h ≈ megapixels × 1_048_576, preserving aspect ratio.
    Output dimensions snapped to the nearest `multiple`.
    """
    _, h, w, _ = image.shape
    aspect = w / h
    target_total = megapixels * 1_048_576
    new_h = math.sqrt(target_total / aspect)
    new_w = new_h * aspect
    new_w = max(multiple, round(new_w / multiple) * multiple)
    new_h = max(multiple, round(new_h / multiple) * multiple)
    return _scale_image(image, new_w, new_h, method)


def _mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """MASK (B,H,W) or (H,W) → IMAGE (B,H,W,3)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask.unsqueeze(-1).expand(-1, -1, -1, 3)


def _image_to_mask(image: torch.Tensor) -> torch.Tensor:
    """IMAGE (B,H,W,C) → MASK (B,H,W) via red channel."""
    return image[..., 0]


def _empty_mask(image: torch.Tensor) -> torch.Tensor:
    """Return a zero mask matching the spatial size of image (B,H,W,C)."""
    _, h, w, _ = image.shape
    return torch.zeros([image.shape[0], h, w], dtype=torch.float32)


def _apply_scale_steps(image: torch.Tensor, scale_by: float, megapixels: float,
                       multiple: int, enable_megapixels: bool, method: str) -> torch.Tensor:
    """
    Shared two-step scaling pipeline used by both the image and mask paths,
    so they always end up at matching dimensions.

    Step 1: scale_by (skipped entirely, no snapping, when scale_by == 1.0)
    Step 2: megapixels constraint (skipped entirely when enable_megapixels=False)
    """
    if scale_by != 1.0:
        result = _scale_by_factor(image, scale_by, method, multiple)
    else:
        result = image

    if enable_megapixels:
        result = _scale_to_megapixels(result, megapixels, method, multiple)

    return result


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class ImageScaleToNormPixels:
    """
    ImageScaleToNormPixels

    Two independent, stackable scaling steps applied to an image (and
    optionally a mask):

      1. scale_by   — simple size multiplier (1.0 = unchanged, 1.5 = 50% bigger)
      2. megapixels — constrains the result to a target megapixel count,
                      only when enable_megapixels=True

    scale_by=1.0 and enable_megapixels=False together reproduce the
    original input completely unchanged. scale_by alone (with
    enable_megapixels=False) behaves exactly like the standalone
    ImageScaleToNormPixels node.

    No VAE encoding — outputs are just image and mask.
    mask is optional — returns zeros when not connected.
    """

    CATEGORY = "YSNodes/image"
    FUNCTION = "run"

    RETURN_TYPES  = ("IMAGE",  "MASK")
    RETURN_NAMES  = ("image",  "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE", {
                    "tooltip": "Input image to scale.",
                }),
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method used when scaling.",
                }),
                "scale_by": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 8.0,
                    "step": 0.01,
                    "tooltip": (
                        "Multiplies the image's current size. "
                        "1.0 = unchanged. 1.5 = 50% bigger. 0.5 = half size. "
                        "Applied before the megapixels step below."
                    ),
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": (
                        "Target resolution in megapixels, applied after scale_by. "
                        "The image is scaled so width × height ≈ megapixels × 1,048,576, "
                        "preserving the original aspect ratio. "
                        "Example: 1.0 ≈ 1024×1024, 1.5 ≈ 1024×1536. "
                        "Only takes effect when enable_megapixels=True."
                    ),
                }),
                "multiple": ("INT", {
                    "default": 8,
                    "min": 8,
                    "max": 128,
                    "step": 4,
                    "tooltip": (
                        "Nearest multiple to round scaled width/height to "
                        "(applies to both scale_by and megapixels). "
                        "8 = VAE-safe (recommended)."
                    ),
                }),
                "enable_megapixels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True  → further scale the scale_by result to target megapixels.\n"
                        "When False → use the scale_by result as-is, no megapixels constraint."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional mask. Goes through the same scale_by + megapixels "
                        "steps as the image. Returns zeros when not connected."
                    ),
                }),
            },
        }

    # ------------------------------------------------------------------
    def run(
        self,
        pixels: torch.Tensor,
        upscale_method: str,
        scale_by: float,
        megapixels: float,
        multiple: int,
        enable_megapixels: bool,
        mask=None,
    ):
        # ── Step 1+2: scale_by then megapixels (image) ──────────────────
        image_out = _apply_scale_steps(
            pixels, scale_by, megapixels, multiple, enable_megapixels, upscale_method
        )

        # ── Step 1+2: scale_by then megapixels (mask, if connected) ─────
        if mask is not None:
            mask_img = _mask_to_image(mask)
            mask_img = _apply_scale_steps(
                mask_img, scale_by, megapixels, multiple, enable_megapixels, upscale_method
            )
            mask_out = _image_to_mask(mask_img)
        else:
            mask_out = _empty_mask(image_out)

        return (image_out, mask_out)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ImageScaleToNormPixels": ImageScaleToNormPixels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleToNormPixels": "ImageScaleToNormPixels",
}
