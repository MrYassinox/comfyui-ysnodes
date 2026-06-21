"""
nodes/reference_conditioning.py
=====================================
Reference Conditioning

pipeline
------------------------------------------
  Image path:
    enable_megapixels=True  → ImageScaleToTotalPixels(pixels, upscale_method, megapixels)
    enable_megapixels=False → pixels unchanged
    → VAEEncode(vae) → latent
    → ReferenceLatent(conditioning, latent) → conditioning_out

  Mask path (parallel, only runs when mask is connected):
    enable_megapixels=True  → MaskToImage → ImageScaleToTotalPixels (same megapixels)
                              → ImageToMask
    enable_megapixels=False → mask unchanged

ReferenceLatent (core ComfyUI node) attaches the encoded latent to the
conditioning's "reference_latents" list — used by models that support
reference-image conditioning (e.g. Flux Kontext-style workflows).

"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

# ---------------------------------------------------------------------------
# Helpers — image / mask scaling (same logic as ImageScaleTotalPixelsSG)
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


def _scale_to_megapixels(image: torch.Tensor, megapixels: float, method: str, multiple: int = 8) -> torch.Tensor:
    """
    Replicate ImageScaleToTotalPixels (comfy-core):
    scale so w×h ≈ megapixels × 1_048_576, preserving aspect ratio.
    Output dimensions snapped to the nearest `multiple` (8 = VAE safe).
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


# ---------------------------------------------------------------------------
# Helpers — VAE encode + ReferenceLatent (mirrors comfy-core exactly)
# ---------------------------------------------------------------------------

def _vae_encode(vae, image: torch.Tensor) -> dict:
    """Encode image with VAE — mirrors the VAEEncode core node."""
    samples = vae.encode(image[:, :, :, :3])
    return {"samples": samples}


def _attach_reference_latent(conditioning, latent_samples) -> list:
    """
    Mirrors comfy-core ReferenceLatent.append():
    appends latent_samples to each conditioning entry's "reference_latents"
    list (creating it if it doesn't exist yet).
    """
    result = []
    for cond_tensor, extra in conditioning:
        extra = extra.copy()
        if "reference_latents" in extra:
            extra["reference_latents"] = extra["reference_latents"] + [latent_samples]
        else:
            extra["reference_latents"] = [latent_samples]
        result.append([cond_tensor, extra])
    return result


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class ReferenceConditioning:
    """
    🔗 Reference Conditioning

    Attaches a reference image's latent to a conditioning, for models that
    support reference-image conditioning (e.g. Flux Kontext-style workflows).

    The image (and optional mask) can be scaled to a target megapixel count
    before encoding, controlled by enable_megapixels. When disabled, the
    image/mask pass through unchanged.

    mask is optional — when not connected, the mask output returns zeros.
    vae is required — matches the original subgraph; ComfyUI will not queue
    this node without a VAE connected.
    """

    CATEGORY = "YSNodes/conditioning"
    FUNCTION = "run"

    RETURN_TYPES  = ("CONDITIONING",   "IMAGE",  "MASK",  "LATENT")
    RETURN_NAMES  = ("conditioning",   "image",  "mask",  "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": (
                        "Conditioning to attach the reference latent to. "
                        "Works for positive or negative conditioning."
                    ),
                }),
                "pixels": ("IMAGE", {
                    "tooltip": "Reference image to encode and attach as a latent reference.",
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE model used to encode the reference image. Required.",
                }),
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "nearest-exact",
                    "tooltip": "Interpolation method used when scaling.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": (
                        "Target resolution in megapixels when enable_megapixels=True. "
                        "The image is scaled so width × height ≈ megapixels × 1,048,576, "
                        "preserving aspect ratio."
                    ),
                }),
                "multiple": ("INT", {
                    "default": 8,
                    "min": 8,
                    "max": 128,
                    "step": 4,
                    "tooltip": (
                        "Nearest multiple to round the scaled width/height to. "
                        "8 = VAE-safe (recommended)."
                    ),
                }),
                "enable_megapixels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True  → scale pixels and mask to target megapixels before encoding.\n"
                        "False → use pixels and mask unchanged."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional mask. Scaled with the same method and target MP as pixels. "
                        "Returns zeros when not connected."
                    ),
                }),
            },
        }

    # ------------------------------------------------------------------
    def run(
        self,
        conditioning,
        pixels: torch.Tensor,
        vae,
        upscale_method: str,
        megapixels: float,
        multiple: int,
        enable_megapixels: bool,
        mask=None,
    ):
        # ── Step 1: Scale or passthrough image ──────────────────────────
        if enable_megapixels:
            image_out = _scale_to_megapixels(pixels, megapixels, upscale_method, multiple)
        else:
            image_out = pixels

        # ── Step 2: Scale or passthrough mask (if connected) ────────────
        if mask is not None:
            if enable_megapixels:
                mask_img = _mask_to_image(mask)
                mask_img = _scale_to_megapixels(mask_img, megapixels, upscale_method, multiple)
                mask_out = _image_to_mask(mask_img)
            else:
                mask_out = mask
        else:
            mask_out = _empty_mask(image_out)

        # ── Step 3: VAE encode the (scaled) image ────────────────────────
        latent = _vae_encode(vae, image_out)

        # ── Step 4: Attach latent to conditioning (ReferenceLatent) ──────
        conditioning_out = _attach_reference_latent(conditioning, latent["samples"])

        return (conditioning_out, image_out, mask_out, latent)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ReferenceConditioning": ReferenceConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceConditioning": "Reference Conditioning",
}
