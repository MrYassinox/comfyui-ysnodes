"""
nodes/image_scale_total_pixels.py
======================================
ImageScaleToTotalPixels

Subgraph pipeline (faithfully reproduced)
------------------------------------------
Two parallel paths controlled by enable_megapixels (switch):

  on_false (enable_megapixels=False) → passthrough — no scaling applied
  on_true  (enable_megapixels=True)  → scale to target megapixels

  Image path:
    ImageScaleToTotalPixels(pixels, upscale_method, megapixels)
    → Switch(enable_megapixels) → image

  VAE encode path:
    VAEEncode(scaled_image, vae)
    → Switch(enable_megapixels) → latent_image

  Mask path:
    MaskToImage → ImageScaleToTotalPixels → ImageToMask
    → Switch(enable_megapixels) → mask

  Latent mask path:
    VAEEncode(scaled_mask_as_image, vae)
    → Switch(enable_megapixels) → latent_mask

Fallbacks
---------
  vae  not connected → latent_image / latent_mask = zeros
  mask not connected → mask / latent_mask          = zeros

Improvements over the original subgraph
-----------------------------------------
  • Single node replaces 7 (2× ImageScaleToTotalPixels, 2× VAEEncode,
    MaskToImage, ImageToMask, 2× Switch)
  • mask and vae are optional
  • Lowercase output names
  • Clear tooltips on all inputs
  • Snap-to-8 on scaled dimensions
  • Zero external dependencies
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


def _scale_to_megapixels(image: torch.Tensor, megapixels: float, method: str) -> torch.Tensor:
    """
    Replicate ImageScaleToTotalPixels (comfy-core):
    scale so w×h ≈ megapixels × 1_048_576, preserving aspect ratio.
    Output dimensions snapped to multiples of 8 (VAE safe).
    """
    _, h, w, _ = image.shape
    aspect = w / h
    target_total = megapixels * 1_048_576
    new_h = math.sqrt(target_total / aspect)
    new_w = new_h * aspect
    new_w = max(8, round(new_w / 8) * 8)
    new_h = max(8, round(new_h / 8) * 8)
    return _scale_image(image, new_w, new_h, method)


def _mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """MASK (B,H,W) or (H,W) → IMAGE (B,H,W,3)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask.unsqueeze(-1).expand(-1, -1, -1, 3)


def _image_to_mask(image: torch.Tensor) -> torch.Tensor:
    """IMAGE (B,H,W,C) → MASK (B,H,W) via red channel."""
    return image[..., 0]


def _vae_encode(vae, image: torch.Tensor) -> dict:
    """Encode image with VAE — mirrors VAEEncode core node."""
    samples = vae.encode(image[:, :, :, :3])
    return {"samples": samples}


def _empty_latent(image: torch.Tensor) -> dict:
    """Return a zero latent matching the spatial size of image (B,H,W,C)."""
    _, h, w, _ = image.shape
    samples = torch.zeros(
        [image.shape[0], 4, h // 8, w // 8], dtype=torch.float32
    )
    return {"samples": samples}


def _empty_mask(image: torch.Tensor) -> torch.Tensor:
    """Return a zero mask matching the spatial size of image (B,H,W,C)."""
    _, h, w, _ = image.shape
    return torch.zeros([image.shape[0], h, w], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class ImageScaleTotalPixels:
    """
    🖼 ImageScaleToTotalPixels

    Scales an image (and optionally a mask) to a target megapixel count,
    then outputs both the scaled image/mask and their VAE-encoded latents.

    When enable_megapixels=False all outputs are the original inputs unchanged.
    When enable_megapixels=True  all outputs are scaled to target megapixels.

    Optional inputs:
      mask → MASK and LATENT_MASK return zeros when not connected
      vae  → LATENT_IMAGE and LATENT_MASK return zeros when not connected
    """

    CATEGORY = "YSNodes/image"
    FUNCTION = "run"

    RETURN_TYPES  = ("IMAGE",  "LATENT",        "MASK",  "LATENT")
    RETURN_NAMES  = ("image",  "latent_image",  "mask",  "latent_mask")

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
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": (
                        "Target resolution in megapixels. "
                        "The image is scaled so width × height ≈ megapixels × 1,048,576, "
                        "preserving the original aspect ratio. "
                        "Example: 1.0 ≈ 1024×1024, 1.5 ≈ 1024×1536."
                    ),
                }),
                "enable_megapixels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True  → scale image and mask to target megapixels.\n"
                        "When False → pass all inputs through unchanged."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional mask. Scaled with the same method and target MP as the image. "
                        "Returns zeros when not connected."
                    ),
                }),
                "vae": ("VAE", {
                    "tooltip": (
                        "Optional VAE model for encoding outputs to latent space. "
                        "latent_image and latent_mask return zeros when not connected."
                    ),
                }),
            },
        }

    # ------------------------------------------------------------------
    def run(
        self,
        pixels: torch.Tensor,
        upscale_method: str,
        megapixels: float,
        enable_megapixels: bool,
        mask=None,
        vae=None,
    ):
        # ── Step 1: Scale or passthrough ────────────────────────────────
        if enable_megapixels:
            image_out = _scale_to_megapixels(pixels, megapixels, upscale_method)
        else:
            image_out = pixels

        # ── Step 2: Scale mask (if connected) ───────────────────────────
        if mask is not None:
            if enable_megapixels:
                mask_img   = _mask_to_image(mask)
                mask_img   = _scale_to_megapixels(mask_img, megapixels, upscale_method)
                mask_out   = _image_to_mask(mask_img)
            else:
                mask_out   = mask
        else:
            mask_out = _empty_mask(image_out)

        # ── Step 3: VAE encode (if connected) ───────────────────────────
        if vae is not None:
            latent_image = _vae_encode(vae, image_out)

            if mask is not None:
                # Encode mask as single-channel image
                mask_as_img  = _mask_to_image(mask_out)
                latent_mask  = _vae_encode(vae, mask_as_img)
            else:
                latent_mask = _empty_latent(image_out)
        else:
            latent_image = _empty_latent(image_out)
            latent_mask  = _empty_latent(image_out)

        return (image_out, latent_image, mask_out, latent_mask)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ImageScaleTotalPixels": ImageScaleTotalPixels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleTotalPixels": "ImageScaleToTotalPixels",
}
