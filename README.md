# comfyui-ysnodes

A lightweight ComfyUI custom node pack — Every node is self-contained with zero dependency on external custom node packs.

> **Author:** Mr. Yassin NM (Mr. Yassinox)  
> **License:** MIT

---

## Features

- **Zero external dependencies** — every node is a self-contained Python class. No nhknodes, no extra installs
- **Browser notifications** — workflow completion alerts with sound + OS notification (bundled JS + Python)

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


---

## 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MrYassinox/comfyui-ysnodes.git
# or download and extract the folder directly
```

1. Drop the `comfyui-ysnodes/` folder into `ComfyUI/custom_nodes/`
2. Restart ComfyUI — all nodes appear under **YSNodes/** in the node menu.


## 📁 Project Structure

```
ComfyUI/
└── custom_nodes/
    └── comfyui-ysnodes/
        ├── __init__.py                    # Package entry, auto-discovers nodes/, version="1.0.0"
        ├── README.md                      # This file
        ├── LICENSE                        # MIT License
        ├── requirements.txt               # Empty (stdlib + comfy + torch only)
        ├── web/
        │   ├── purge_notification_sg.js   # Frontend: sound + OS notification via WebSocket
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
            └── reference_conditioning.py
```

---

## Nodes Details

### ⚡ Sigma Generator
> Category: `YSNodes/sampling`

Combines `KSamplerSelect`, `BasicScheduler`, and `SplitSigmas` into a single node.
Generates a sigma schedule clamped to the low-sigma range after an optional start-step skip.
Connect the outputs directly to a **SamplerCustom** node.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `model` | MODEL | — | Diffusion model |
| **IN** | `sampler_name` | COMBO | `dpmpp_2m` | Sampler algorithm |
| **IN** | `scheduler` | COMBO | `simple` | Noise schedule type |
| **IN** | `steps` | INT | `4` | Total number of sampling steps |
| **IN** | `skip_start_step` | INT | `0` | Number of leading sigma steps to discard (SplitSigmas). `0` keeps all sigmas |
| **IN** | `denoise` | FLOAT | `1.0` | Denoising strength — values below 1.0 reduce the effective number of steps |
| **OUT** | `SIGMAS` | SIGMAS | — | Low sigmas after the split, ready for SamplerCustom |
| **OUT** | `SAMPLER` | SAMPLER | — | Sampler object, ready for SamplerCustom |

---

### 📐 Size Picker
> Category: `YSNodes/utility`

Picks a canvas resolution from an **aspect ratio** combined with a **target megapixel count** and a **rounding multiple**. Dimensions are computed using the same formula as ComfyUI core's `ResolutionSelector`. When `use_image_size` is enabled, the dimensions are taken from the connected image instead.

All 17 aspect ratios are inlined — covering print, social media, video/cinema, and AI-generated formats. Zero dependency on ComfyUI-nhknodes or any other external pack.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `aspect_ratio` | COMBO | `1:1 (Square)` | Aspect ratio used to compute width and height |
| **IN** | `megapixels` | FLOAT | `1.0` | Target total megapixels. `1.0` MP ≈ 1024×1024 for a square aspect ratio |
| **IN** | `multiple` | INT | `8` | Nearest multiple to round the computed width/height to. `8` = VAE-safe (recommended) |
| **IN** | `width_override` | INT | `0` | Override width in pixels. `0` = use the value computed from `aspect_ratio` + `megapixels` |
| **IN** | `height_override` | INT | `0` | Override height in pixels. `0` = use the value computed from `aspect_ratio` + `megapixels` |
| **IN** | `batch_size` | INT | `1` | Number of latents in the batch |
| **IN** | `use_image_size` | BOOLEAN | `false` | When `true`, width and height are read from the connected image |
| **IN** | `image` | IMAGE | *(optional)* | Source image — required when `use_image_size = true` |
| **OUT** | `width` | INT | — | Final output width |
| **OUT** | `height` | INT | — | Final output height |
| **OUT** | `batch` | INT | — | Batch size passthrough |
| **OUT** | `latent` | LATENT | — | Empty latent at `width × height × batch` |

<details>
<summary>Aspect ratio presets (17 total)</summary>

| Ratio | Family | Use case |
|---|---|---|
| `1:1` | Square | Instagram, profile pictures |
| `4:5` | Portrait Social | Instagram portrait, Pinterest |
| `5:7` | Portrait Print | Standard print portrait |
| `2:3` | Portrait Photo | 35mm photo, portrait photography |
| `3:4` | Portrait Standard | Tablet, classic portrait |
| `5:12` | Portrait Tall | Mobile stories, tall banners |
| `9:16` | Portrait Widescreen | Stories, Reels, TikTok |
| `9:21` | Portrait Ultrawide | Ultra-tall mobile content |
| `1:2` | Portrait Panoramic | Ultra-tall format |
| `5:4` | Landscape Social | Facebook landscape |
| `7:5` | Landscape Print | Standard print landscape |
| `3:2` | Photo | 35mm landscape, DSLR default |
| `4:3` | Standard | Classic monitor, MFT cameras |
| `12:5` | Landscape Wide | Ultra-wide landscape |
| `16:9` | Widescreen | HD video, YouTube, monitors |
| `21:9` | Ultrawide | Cinematic ultrawide |
| `2:1` | Panoramic | Panoramic photography |

</details>

---

### 📐 Size Picker Full
> Category: `YSNodes/utility`

Extends **Size Picker** with an optional VAE input and two latent outputs instead of one.
Use `latent_empty` for txt2img pipelines and `latent_from_image` for img2img / inpainting pipelines.
All inputs are identical to Size Picker.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `aspect_ratio` | COMBO | `1:1 (Square)` | Aspect ratio used to compute width and height |
| **IN** | `megapixels` | FLOAT | `1.0` | Target total megapixels. `1.0` MP ≈ 1024×1024 for a square aspect ratio |
| **IN** | `multiple` | INT | `8` | Nearest multiple to round the computed width/height to. `8` = VAE-safe (recommended) |
| **IN** | `width_override` | INT | `0` | Override width in pixels. `0` = use the value computed from `aspect_ratio` + `megapixels` |
| **IN** | `height_override` | INT | `0` | Override height in pixels. `0` = use the value computed from `aspect_ratio` + `megapixels` |
| **IN** | `batch_size` | INT | `1` | Number of latents in the batch |
| **IN** | `use_image_size` | BOOLEAN | `false` | When `true`, width and height are read from the connected image |
| **IN** | `image` | IMAGE | *(optional)* | Source image — for size detection and VAE encoding |
| **IN** | `vae` | VAE | *(optional)* | VAE model for encoding. When not connected, `latent_from_image` returns an empty latent |
| **OUT** | `width` | INT | — | Final output width |
| **OUT** | `height` | INT | — | Final output height |
| **OUT** | `batch` | INT | — | Batch size passthrough |
| **OUT** | `latent_empty` | LATENT | — | Always an empty latent at `(width, height, batch)` — use for txt2img |
| **OUT** | `latent_from_image` | LATENT | — | VAE-encoded image resized to `(width, height)` — use for img2img / inpaint. Falls back to empty latent when VAE is not connected |

---

### 🔗 Reference Conditioning
> Category: `YSNodes/conditioning`

Attaches a reference image's latent to a conditioning tensor, for models that support reference-image conditioning (e.g. Flux Kontext-style workflows).

The image (and optional mask) can be scaled to a target megapixel count before encoding, controlled by `enable_megapixels`. When disabled, the image and mask pass through unchanged.

Internally this node mirrors the core ComfyUI `VAEEncode` + `ReferenceLatent` pipeline — the encoded latent is appended to the conditioning's `"reference_latents"` list.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `conditioning` | CONDITIONING | — | Conditioning to attach the reference latent to. Works for positive or negative conditioning |
| **IN** | `pixels` | IMAGE | — | Reference image to encode and attach as a latent reference |
| **IN** | `vae` | VAE | — | VAE model used to encode the reference image. Required |
| **IN** | `upscale_method` | COMBO | `nearest-exact` | Interpolation method used when scaling |
| **IN** | `megapixels` | FLOAT | `1.0` | Target resolution in megapixels when `enable_megapixels=True`. The image is scaled so `width × height ≈ megapixels × 1,048,576`, preserving aspect ratio |
| **IN** | `multiple` | INT | `8` | Nearest multiple to round the scaled width/height to. `8` = VAE-safe (recommended) |
| **IN** | `enable_megapixels` | BOOLEAN | `true` | `true` → scale pixels and mask to target megapixels before encoding. `false` → use pixels and mask unchanged |
| **IN** | `mask` | MASK | *(optional)* | Optional mask. Scaled with the same method and target MP as pixels. Returns zeros when not connected |
| **OUT** | `conditioning` | CONDITIONING | — | Conditioning with the reference latent attached |
| **OUT** | `image` | IMAGE | — | The (optionally scaled) reference image |
| **OUT** | `mask` | MASK | — | The (optionally scaled) mask, or zeros if no mask was connected |
| **OUT** | `latent` | LATENT | — | VAE-encoded latent of the (optionally scaled) reference image |

---

### 🔁 Redraw Size Calculator
> Category: `YSNodes/image`

Scales an image (and optionally a mask) so its total pixel count falls within a defined megapixel range, preserving the original aspect ratio. The target MP is clamped between a minimum and maximum preset — both of which can be overridden with a custom value. Outputs both the resized and original versions of the image and mask, along with all four dimension values.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `image` | IMAGE | — | Input image to resize |
| **IN** | `upscale_method` | COMBO | `lanczos` | Interpolation method: `lanczos`, `bicubic`, `bilinear`, `nearest-exact`, `area` |
| **IN** | `min_preset` | COMBO | `0.25 MP (512×512 · SD1.5)` | Preset minimum MP — images with fewer pixels are upscaled to this |
| **IN** | `min_megapixels_override` | FLOAT | `0.0` | Custom minimum MP. `0` = use preset |
| **IN** | `max_preset` | COMBO | `1.50 MP (1024×1536 · Flux)` | Preset maximum MP — images with more pixels are downscaled to this |
| **IN** | `max_megapixels_override` | FLOAT | `0.0` | Custom maximum MP. `0` = use preset |
| **IN** | `mask` | MASK | *(optional)* | Mask resized with the same method and target MP as the image |
| **OUT** | `image_resized` | IMAGE | — | Image scaled to the target MP |
| **OUT** | `mask_resized` | MASK | — | Mask scaled to match `image_resized`. Returns zeros when no mask is connected |
| **OUT** | `image_original` | IMAGE | — | Original image passthrough |
| **OUT** | `mask_original` | MASK | — | Original mask passthrough |
| **OUT** | `width_resized` | INT | — | Width of the resized image |
| **OUT** | `height_resized` | INT | — | Height of the resized image |
| **OUT** | `width_original` | INT | — | Width of the original image |
| **OUT** | `height_original` | INT | — | Height of the original image |
| **OUT** | `target_megapixels` | FLOAT | — | The actual MP value used for scaling — useful for debugging or chaining |

<details>
<summary>MP presets (14 total)</summary>

| Family | MP | Example resolution |
|---|---|---|
| SD1.5 | 0.25 | 512×512 |
| SD1.5 | 0.31 | 640×512 |
| SD1.5 | 0.38 | 512×768 |
| SD1.5 | 0.56 | 768×768 |
| SDXL | 0.77 | 896×896 |
| SDXL | 0.94 | 1536×640 |
| SDXL | 0.98 | 1216×832 |
| SDXL | 1.00 | 1024×1024 |
| Flux | 1.50 | 1024×1536 |
| Flux | 1.56 | 1280×1280 |
| HD | 1.98 | 1920×1080 |
| 2×SDXL | 2.25 | 1536×1536 |
| 4K | 4.00 | 2048×2048 |
| 4K UHD | 8.29 | 3840×2160 |

</details>

---

### 🔔 Purge & Notification
> Category: `YSNodes/utility`

Combines VRAM purge and workflow completion notifications into a single end-of-workflow node.
Place it after your last Save Image or Preview Image node.

Sound playback and browser OS notifications are handled by the bundled JS file (`web/purge_notification.js`) which communicates with the Python node via ComfyUI's WebSocket. The `sound_file` must be placed in `comfyui-ysnodes/web/assets/`.

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `purge_cache` | BOOLEAN | `true` | Clear CUDA/MPS tensor cache and ComfyUI soft cache |
| **IN** | `purge_models` | BOOLEAN | `false` | Unload all models from VRAM. The next run will reload models from disk |
| **IN** | `alert_mode` | COMBO | `always` | `always` — notify on every run · `on_empty_queue` — notify only when the queue is fully empty |
| **IN** | `system_notification` | BOOLEAN | `true` | Fire a browser OS notification popup |
| **IN** | `notification_text` | STRING | `✅ Workflow complete` | Text body of the OS notification |
| **IN** | `play_sound` | BOOLEAN | `true` | Play an audio file in the browser tab |
| **IN** | `sound_volume` | FLOAT | `0.5` | Playback volume from `0.0` (mute) to `1.0` (full) |
| **IN** | `sound_file` | STRING | `notify.mp3` | Filename inside `web/assets/` (e.g. `notify.mp3`) or a full URL to an audio file |
| **IN** | `signal` | `*` | *(optional)* | Connect the output of your last node here to ensure correct execution order |
| **OUT** | `signal` | `*` | — | Passthrough of the `signal` input |

---

### 🖼 ImageScaleToTotalPixels
> Category: `YSNodes/image`

Scales an image (and optionally a mask) to a target megapixel count, then outputs both the scaled image/mask and their VAE-encoded latents. When `enable_megapixels=False` all outputs are the original inputs unchanged.

Single node replaces 7 nodes from the original subgraph (2× ImageScaleToTotalPixels, 2× VAEEncode, MaskToImage, ImageToMask, 2× Switch).

| | Name | Type | Default | Description |
|---|---|---|---|---|
| **IN** | `pixels` | IMAGE | — | Input image to scale |
| **IN** | `upscale_method` | COMBO | `lanczos` | Interpolation method used when scaling |
| **IN** | `megapixels` | FLOAT | `1.0` | Target resolution in megapixels. The image is scaled so `width × height ≈ megapixels × 1,048,576`, preserving the original aspect ratio |
| **IN** | `enable_megapixels` | BOOLEAN | `true` | `true` → scale image and mask to target megapixels. `false` → pass all inputs through unchanged |
| **IN** | `mask` | MASK | *(optional)* | Optional mask. Scaled with the same method and target MP as the image. Returns zeros when not connected |
| **IN** | `vae` | VAE | *(optional)* | Optional VAE model for encoding outputs to latent space. `latent_image` and `latent_mask` return zeros when not connected |
| **OUT** | `image` | IMAGE | — | Scaled image (or original when `enable_megapixels=false`) |
| **OUT** | `latent_image` | LATENT | — | VAE-encoded latent of the scaled image. Zeros when VAE is not connected |
| **OUT** | `mask` | MASK | — | Scaled mask (or original when `enable_megapixels=false`). Zeros when mask is not connected |
| **OUT** | `latent_mask` | LATENT | — | VAE-encoded latent of the scaled mask. Zeros when VAE or mask is not connected |

---

## FAQ

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

---

## Contributing

The pack uses **auto-discovery** — adding a new node is a 3-step process:

1. Create a new file inside `nodes/`, e.g. `nodes/my_new_node.py`
2. Define your node class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, and register it at the bottom:

```python
NODE_CLASS_MAPPINGS = {"MyNewNode": MyNewNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyNewNode": "✨ My New Node"}
```

3. Restart ComfyUI — done ✅

The `__init__.py` scans `nodes/` on startup and imports every `.py` file automatically. No other files need to be touched.

---

## 📝 Changelog

### v1.1.0

- **BREAKING:** `Size Picker` — replaced 35 fixed resolution presets with dynamic aspect-ratio + megapixel + multiple system (17 aspect ratios)
- **BREAKING:** `Size Picker Full` — same dynamic sizing as Size Picker, with `latent_empty` and `latent_from_image` outputs
- **NEW:** `Reference Conditioning` — attaches reference image latent to conditioning (Flux Kontext-style)
- **Updated:** README fully rewritten with per-node I/O tables, presets, and credits

### v1.0.0

- Initial release with 7 nodes:
  - Sigma Generator
  - Size Picker (fixed resolution presets)
  - Size Picker Full
  - Redraw Size Calculator
  - Purge & Notification
  - ImageScaleToTotalPixels
  - Reference Conditioning
- Auto-discovery system for node registration
- Browser notification JS integration

---

## Credits

Many of these nodes are extracted, merged, and improved from existing open-source ComfyUI nodes and subgraphs. This pack would not exist without the work of the original authors.

| Node | Based on | Author | Link |
|---|---|---|---|
| 📐 Size Picker | `📐 Size Picker` | [Enashka](https://github.com/Enashka) | [ComfyUI-nhknodes](https://github.com/Enashka/ComfyUI-nhknodes) |
| 🔔 Purge & Notification | `Unified Notification` | [royceschultz](https://github.com/royceschultz) | [ComfyUI-Notifications](https://github.com/royceschultz/ComfyUI-Notifications) |
| 🔔 Purge & Notification | `PurgeVRAM` | [T8mars](https://github.com/T8mars) | [comfyui-purgevram](https://github.com/T8mars/comfyui-purgevram) |
| 🔔 Purge & Notification | `LayerUtility: Purge VRAM` | [chflame163](https://github.com/chflame163) | [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) |
| 🖼 ImageScaleToTotalPixels | `ImageScaleToTotalPixels` | [comfyanonymous](https://github.com/comfyanonymous) | [ComfyUI (core)](https://github.com/comfyanonymous/ComfyUI) |
| 🔁 Redraw Size Calculator | `ImageScaleToTotalPixels` + subgraph | [comfyanonymous](https://github.com/comfyanonymous) | [ComfyUI (core)](https://github.com/comfyanonymous/ComfyUI) |
| ⚡ Sigma Generator | `KSamplerSelect` + `BasicScheduler` + `SplitSigmas` | [comfyanonymous](https://github.com/comfyanonymous) | [ComfyUI (core)](https://github.com/comfyanonymous/ComfyUI) |
| 📐 Size Picker Full | Extended from Size Picker | — | *(built on top of Size Picker)* |
| 🔗 Reference Conditioning | `VAEEncode` + `ReferenceLatent` | [comfyanonymous](https://github.com/comfyanonymous) | [ComfyUI (core)](https://github.com/comfyanonymous/ComfyUI) |

> **Note:** All nodes in this pack have been rewritten as self-contained Python classes with zero external dependencies. Input/output names, defaults, tooltips, and preset lists have been redesigned for clarity and usability.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2025-2026 Mr. Yassin NM (Mr. Yassinox)
