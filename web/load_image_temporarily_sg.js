/**
 * web/load_image_temporarily_sg.js
 * ===================================
 * Frontend for Load Image Temporarily
 *
 * Adds a "choose file to upload" button to the node that uploads the
 * selected file directly into ComfyUI's /temp folder (type=temp), instead
 * of the default /input folder core ComfyUI always uses.
 *
 * Also renders an image thumbnail on the node itself (like core Load Image),
 * by setting node.imgs and calling node.setSizeForImage() — the same
 * mechanism ComfyUI's own image-upload widget uses internally. This fires
 * in three places: after a successful upload, when picking an existing
 * file from the dropdown, and when reopening a saved workflow that already
 * has a file selected.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPE = "LoadImageTemporarily";

async function uploadToTemp(file) {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("type", "temp");
    formData.append("overwrite", "true");

    const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body: formData,
    });

    if (resp.status !== 200) {
        const text = await resp.text();
        throw new Error(`Upload failed (${resp.status}): ${text}`);
    }

    return await resp.json(); // { name, subfolder, type }
}

function annotatedName(data) {
    // Mirrors ComfyUI core's convention: append the folder type in
    // brackets so folder_paths.get_annotated_filepath() resolves it
    // correctly regardless of the node's default_dir.
    const base = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
    return `${base} [${data.type}]`;
}

function viewUrlForTempFile(name) {
    // Strip an optional " [type]" annotation suffix if present — this node
    // only ever deals with files in /temp regardless of annotation.
    const clean = name.replace(/ \[[^\]]*\]$/, "");
    const params = new URLSearchParams({
        filename: clean,
        type: "temp",
        subfolder: "",
        t: Date.now(), // cache-bust in case a same-named file was overwritten
    });
    return api.apiURL(`/view?${params.toString()}`);
}

/**
 * Render the selected file as a thumbnail on the node, growing the node
 * to a sensible size — same mechanism core ComfyUI's Load Image uses.
 */
function refreshPreview(node, name) {
    if (!name) {
        node.imgs = null;
        node.setDirtyCanvas(true, true);
        return;
    }

    const img = new Image();
    img.onload = () => {
        node.imgs = [img];
        // setSizeForImage is added by ComfyUI core to LGraphNode's prototype
        // and handles resizing exactly like Load Image / Preview Image do.
        // Optional chaining so this degrades gracefully if it's ever absent.
        node.setSizeForImage?.();
        app.graph.setDirtyCanvas(true);
    };
    img.onerror = () => {
        console.warn(`[LoadImageTemporarily] Could not load preview for: ${name}`);
    };
    img.src = viewUrlForTempFile(name);
}

app.registerExtension({
    name: "comfyui-ysnodes.LoadImageTemporarily",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_TYPE) return;

        // ── Upload button + preview on upload ───────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const imageWidget = this.widgets?.find((w) => w.name === "image");
            if (!imageWidget) return result;

            const node = this;

            // Preview when picking an existing file from the dropdown
            const origCallback = imageWidget.callback;
            imageWidget.callback = function (value, ...rest) {
                refreshPreview(node, value);
                return origCallback?.apply(this, [value, ...rest]);
            };

            this.addWidget("button", "choose file to upload", "upload", () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.style.display = "none";
                document.body.appendChild(input);

                input.addEventListener("change", async () => {
                    if (!input.files.length) {
                        document.body.removeChild(input);
                        return;
                    }
                    try {
                        const data = await uploadToTemp(input.files[0]);
                        const name = annotatedName(data);

                        if (!imageWidget.options.values.includes(name)) {
                            imageWidget.options.values.push(name);
                        }
                        imageWidget.value = name;
                        refreshPreview(node, name);
                        node.setDirtyCanvas(true, true);

                        console.log(`[LoadImageTemporarily] Uploaded to temp: ${name}`);
                    } catch (err) {
                        console.error("[LoadImageTemporarily] Upload error:", err);
                        alert(`Upload failed: ${err.message}`);
                    } finally {
                        document.body.removeChild(input);
                    }
                });

                input.click();
            });

            return result;
        };

        // ── Restore preview when reopening a saved workflow ─────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;

            const imageWidget = this.widgets?.find((w) => w.name === "image");
            if (imageWidget?.value) {
                refreshPreview(this, imageWidget.value);
            }

            return result;
        };
    },
});
