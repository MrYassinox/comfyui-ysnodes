<p align="center">
  <img src="https://img.shields.io/badge/version-1.2.0-blue.svg" alt="Version 1.2.0">
  <img src="https://img.shields.io/badge/type-skill-8A2BE2.svg" alt="Skill">
  <img src="https://img.shields.io/badge/dependencies-none-brightgreen.svg" alt="No Dependencies">
  <img src="https://img.shields.io/badge/nodes-9-orange.svg" alt="Nodes">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>
<p align="center">
  <h1 align="center">comfyui-ysnodes</h1>
  <p align="center">A lightweight ComfyUI custom node pack — every node is self-contained with zero dependency on external custom node packs.</p>
</p>

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Nodes](#-nodes) • [FAQ](#-faq) • [Contributing](#-contributing) • [Changelog](#-changelog) • [Credits](#-credits) • [License](#-license)

---

## 📖 Overview

**comfyui-ysnodes** is a personal pack of custom ComfyUI nodes covering resolution picking, sigma scheduling, reference conditioning, image scaling, and workflow utilities. Every node is a self-contained Python class — no external custom node packs required.

**Who it's for:** ComfyUI users who want a clean, dependency-free toolkit for common workflow tasks.

---

## ✨ Features

- **Zero external dependencies** — every node is a self-contained Python class. No nhknodes, no extra installs
- **Multi-model latent support** — `latent_type` selector for SD/SDXL (4ch), SD3/AuraFlow (16ch), Flux (16ch), and Flux2 (128ch)
- **Dynamic resolution picking** — 17 aspect ratios × megapixel target × rounding multiple (replaces 35 fixed presets)
- **Browser notifications** — workflow completion alerts with sound + OS notification
- **Temp folder image loading** — load throwaway reference images from `/temp` with a built-in upload button

---

## Node Index

| # | Node | Category | Description |
|---|---|---|---|
| 1 | [⚡ Sigma Generator](#-sigma-generator) | `sampling` | KSamplerSelect + BasicScheduler + SplitSigmas in one node |
| 2 | [📐 Size Picker](#-size-picker) | `utility` | Dynamic aspect-ratio resolution picker with megapixel-based sizing |
| 3 | [📐 Size Picker Full](#-size-picker-full) | `utility` | Size Picker with VAE encoding — outputs empty latent and image latent |
| 4 | [🔗 Reference Conditioning](#-reference-conditioning) | `conditioning` | Attaches a reference image latent to conditioning (Flux Kontext-style) |
| 5 | [🔁 Redraw Size Calculator](#-redraw-size-calculator) | `image` | Auto-scale image and mask to a clamped megapixel range |
| 6 | [🔔 Purge & Notification](#-purge--notification) | `utility` | Purge VRAM cache and send browser sound + OS notification |
| 7 | [🖼 ImageScaleToTotalPixels](#-imagescaletotalpixels) | `image` | Scale image + mask to target MP with VAE encoding and passthrough switch |
| 8 | [🖼 ImageScaleToNormPixels](#-imagescaletonormpixels) | `image` | Scale image + mask with `scale_by` + megapixels (no VAE encoding) |
| 9 | [📂 Load Image Temporarily](#-load-image-temporarily) | `image` | Load images from ComfyUI's `/temp` folder with upload button |


## 📦 Installation

### From GitHub

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MrYassinox/comfyui-ysnodes.git
```

### Manual

1. Download and extract the `comfyui-ysnodes/` folder
2. Place it in `ComfyUI/custom_nodes/`
3. Restart ComfyUI — all nodes appear under **YSNodes/** in the node menu

> [!NOTE]
> No additional dependencies. The pack uses only Python stdlib, PyTorch, and ComfyUI built-ins.


## 📁 Project Structure

```
ComfyUI/
└── custom_nodes/
    └── comfyui-ysnodes/
        ├── __init__.py                    # Package entry, auto-discovers nodes/, version="1.1.0"
        ├── README.md                      # This file
        ├── LICENSE                        # MIT License
        ├── requirements.txt               # Empty (stdlib + comfy + torch only)
        ├── web/
        │   ├── purge_notification_sg.js   # Frontend: sound + OS notification via WebSocket
        │   ├── load_image_temporarily_sg.js  # Frontend: upload button for /temp images
        │   └── assets/
        │       ├── notify.mp3             ← add your own sound file here
        │       └── notify.wav
        └── nodes/
            ├── __init__.py                # Auto-exports NODE_CLASS_MAPPINGS from all *.py
            ├── sigma_generator.py
            ├── size_picker.py
            ├── size_picker_full.py
            ├── redraw_size.py
            ├── purge_notification.py
            ├── image_scale_total_pixels.py
            ├── image_scale_to_norm_pixels.py
            ├── reference_conditioning.py
            └── load_image_temporarily.py
```

---

## 🧩 Nodes

### ⚡ Sigma Generator
> `YSNodes/sampling`

KSamplerSelect + BasicScheduler + SplitSigmas in one node. Connect directly to **SamplerCustom**.

| Input | Output |
|-------|--------|
| `model`, `sampler_name`, `scheduler`, `steps`, `skip_start_step`, `denoise` | `SIGMAS`, `SAMPLER` |

---

### 📐 Size Picker
> `YSNodes/utility`

Dynamic aspect-ratio resolution picker. Two modes: **computed** (`aspect_ratio` + `megapixels`) or **image-driven** (`use_image_size=true`). Supports `fit_megapixels` to resize the image to the megapixel target while preserving its aspect ratio.

| Input | Output |
|-------|--------|
| `use_image_size`, `fit_megapixels`, `aspect_ratio`, `megapixels`, `multiple`, `width_override`, `height_override`, `batch_size`, `latent_type`, `image` *(opt)* | `image`, `width`, `height`, `batch`, `latent` |

`latent_type`: `SD / SDXL` (4ch ÷8) · `SD3 / AuraFlow` (16ch ÷8) · `Flux` (16ch ÷8) · `Flux2` (128ch ÷16)

<details>
<summary>17 aspect ratios</summary>

`1:1` · `4:5` · `5:7` · `2:3` · `3:4` · `5:12` · `9:16` · `9:21` · `1:2` · `5:4` · `7:5` · `3:2` · `4:3` · `12:5` · `16:9` · `21:9` · `2:1`
</details>

---

### 📐 Size Picker Full
> `YSNodes/utility`

Size Picker + VAE encoding. Outputs two latents: `latent_empty` (txt2img) and `latent_from_image` (VAE-encoded, for img2img/inpaint).

| Input | Output |
|-------|--------|
| Same as Size Picker + `vae` *(opt)* | `image`, `width`, `height`, `batch`, `latent_empty`, `latent_from_image` |

---

### 🔗 Reference Conditioning
> `YSNodes/conditioning`

Attaches a reference image latent to conditioning — for Flux Kontext-style workflows. Optional megapixel scaling before encoding.

| Input | Output |
|-------|--------|
| `conditioning`, `pixels`, `vae`, `upscale_method`, `megapixels`, `multiple`, `enable_megapixels`, `mask` *(opt)* | `conditioning`, `image`, `mask`, `latent` |

---

### 🔁 Redraw Size Calculator
> `YSNodes/image`

Scales an image to fit within a clamped megapixel range (14 presets from SD1.5 to 4K UHD). Outputs both resized and original versions.

| Input | Output |
|-------|--------|
| `image`, `upscale_method`, `min_preset`, `max_preset`, `min_override`, `max_override`, `mask` *(opt)* | `image_resized`, `mask_resized`, `image_original`, `mask_original`, `width_resized`, `height_resized`, `width_original`, `height_original`, `target_megapixels` |

---

### 🔔 Purge & Notification
> `YSNodes/utility`

End-of-workflow node: purge VRAM cache + browser sound + OS notification. Place after your last Save Image node.

| Input | Output |
|-------|--------|
| `purge_cache`, `purge_models`, `alert_mode`, `system_notification`, `notification_text`, `play_sound`, `sound_volume`, `sound_file`, `signal` *(opt)* | `signal` |

> [!TIP]
> Place your sound file in `comfyui-ysnodes/web/assets/`. Supports `.mp3`, `.wav`, or any browser-playable URL.

---

### 🖼 ImageScaleToTotalPixels
> `YSNodes/image`

Scale image + mask to target megapixels **with VAE encoding**. Outputs both pixel and latent versions.

| Input | Output |
|-------|--------|
| `pixels`, `upscale_method`, `megapixels`, `enable_megapixels`, `mask` *(opt)*, `vae` *(opt)* | `image`, `latent_image`, `mask`, `latent_mask` |

---

### 🖼 ImageScaleToNormPixels
> `YSNodes/image`

Two-step scaling (`scale_by` + `megapixels`) **without VAE encoding**. `scale_by=1.0` + `enable_megapixels=false` = passthrough.

| Input | Output |
|-------|--------|
| `pixels`, `upscale_method`, `scale_by`, `megapixels`, `multiple`, `enable_megapixels`, `mask` *(opt)* | `image`, `mask` |

> [!NOTE]
> **TotalPixels vs NormPixels:** TotalPixels includes VAE encoding (outputs latents). NormPixels skips VAE — use it when you only need resized image/mask.

---

### 📂 Load Image Temporarily
> `YSNodes/image`

Loads images from ComfyUI's `/temp` folder. Companion JS adds an upload button — files go directly to `/temp`, never cluttering `/input`.

| Input | Output |
|-------|--------|
| `image` (dropdown + upload) | `image`, `mask`, `width`, `height` |

---

## ❓ FAQ

<details>
<summary><strong>Why don't I hear the notification sound?</strong></summary>

Two things must be in place:

1. `web/purge_notification.js` must be present in the `comfyui-ysnodes/` folder — ComfyUI auto-serves it
2. A sound file (e.g. `notify.mp3`) must exist in `comfyui-ysnodes/web/assets/`

If using a custom filename or URL, update the `sound_file` input accordingly.
</details>

<details>
<summary><strong>What is the difference between Size Picker and Size Picker Full?</strong></summary>

- **Size Picker** → outputs an **empty latent** only. Use for txt2img
- **Size Picker Full** → outputs **two latents**: `latent_empty` (txt2img) and `latent_from_image` (VAE-encoded image for img2img / inpainting)

Size Picker Full has one extra optional input: `vae`. All other inputs are identical.
</details>

<details>
<summary><strong>Do I need to edit `__init__.py` when adding a new node?</strong></summary>

No. The main `__init__.py` auto-discovers every `.py` file inside `nodes/`. Just drop a new file there, restart ComfyUI, and your node appears automatically.
</details>

<details>
<summary><strong>What is the difference between ImageScaleToTotalPixels and ImageScaleToNormPixels?</strong></summary>

- **ImageScaleToTotalPixels** → scales to a target megapixel count **with VAE encoding** — outputs `image`, `latent_image`, `mask`, `latent_mask`
- **ImageScaleToNormPixels** → scales with `scale_by` + megapixels **without VAE encoding** — outputs `image`, `mask` only

Use TotalPixels when you need latents for sampling. Use NormPixels when you just need a resized image/mask (e.g. for conditioning inputs, previews, or chaining into other nodes).
</details>

<details>
<summary><strong>Which latent_type should I use?</strong></summary>

| Model | latent_type |
|-------|-------------|
| SD 1.5, SD 2.x, SDXL | `SD / SDXL` |
| SD3, AuraFlow | `SD3 / AuraFlow` |
| Flux (standard) | `Flux` |
| Flux 2 | `Flux2` |
</details>

---

## 🤝 Contributing

1. Create a new `.py` file in `nodes/`
2. Define your node class + register at the bottom:

```python
NODE_CLASS_MAPPINGS = {"MyNewNode": MyNewNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyNewNode": "✨ My New Node"}
```

3. Restart ComfyUI — done ✅

---

## 📝 Changelog

### v1.2.0

- **NEW:** `ImageScaleToTotalPixels` — scale image + mask to target MP with VAE encoding and passthrough switch
- **NEW:** `ImageScaleToNormPixels` — scale image + mask with `scale_by` + megapixels (no VAE encoding)
- **NEW:** `Load Image Temporarily` — load images from ComfyUI's `/temp` folder with upload button
- **Added:** `latent_type` input to Size Picker / Size Picker Full — selects latent format per model architecture (SD/SDXL, SD3/AuraFlow, Flux, Flux2)
- **Added:** `fit_megapixels` input to Size Picker / Size Picker Full — controls whether connected image keeps exact size or gets resized to megapixel target
- **Added:** `image` output to Size Picker / Size Picker Full — passthrough of connected image, or blank placeholder at computed size
- **Fixed:** Size Picker / Size Picker Full — latent output was always SD/SDXL format regardless of target model, causing dimension mismatches with Flux2 and other non-SD architectures

### v1.1.0

- **BREAKING:** `Size Picker` — replaced 35 fixed resolution presets with dynamic aspect-ratio + megapixel + multiple system (17 aspect ratios)
- **BREAKING:** `Size Picker Full` — same dynamic sizing as Size Picker, with `latent_empty` and `latent_from_image` outputs
- **NEW:** `Reference Conditioning` — attaches reference image latent to conditioning (Flux Kontext-style)
- **Updated:** README fully rewritten with per-node I/O tables, presets, and credits

---

## 🙏 Credits

Many of these nodes are extracted, merged, and improved from existing open-source ComfyUI nodes and subgraphs. This pack would not exist without the work of the original authors.

| Node | Based on | Author |
|------|----------|--------|
| Size Picker | `📐 Size Picker` | [Enashka](https://github.com/Enashka) |
| Purge & Notification | `Unified Notification` + `PurgeVRAM` + `LayerUtility` | [royceschultz](https://github.com/royceschultz) · [T8mars](https://github.com/T8mars) · [chflame163](https://github.com/chflame163) |
| ImageScaleToTotalPixels | `ImageScaleToTotalPixels` | [comfyanonymous](https://github.com/comfyanonymous) |
| Redraw Size Calculator | `ImageScaleToTotalPixels` + subgraph | [comfyanonymous](https://github.com/comfyanonymous) |
| Sigma Generator | `KSamplerSelect` + `BasicScheduler` + `SplitSigmas` | [comfyanonymous](https://github.com/comfyanonymous) |
| Reference Conditioning | `VAEEncode` + `ReferenceLatent` | [comfyanonymous](https://github.com/comfyanonymous) |
| ImageScaleToNormPixels | `ImageScaleToNormPixels` + `ImageScaleToTotalPixels` | [comfyanonymous](https://github.com/comfyanonymous) |
| Load Image Temporarily | `Load Image Temporarily` | MNeMiC Nodes |

> All nodes rewritten as self-contained Python classes with zero external dependencies.

---

## 📄 License

MIT © Mr Yassinox — use it, modify it, share it. Attribution is appreciated but not required.

---

<p align="center">
  <b>⭐ Star on GitHub if you find this useful</b>
</p>