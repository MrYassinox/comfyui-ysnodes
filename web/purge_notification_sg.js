/**
 * web/purge_notification_sg.js
 * ==============================
 * Frontend for 🔔 Purge & Notification (SG)
 *
 * Listens for two WebSocket events sent by purge_notification_sg.py:
 *   "psg_playsound"    → plays audio in the browser
 *   "psg_notification" → fires a browser OS notification
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Sound playback
// ---------------------------------------------------------------------------

function resolveSoundUrl(file) {
    if (!file || file.trim() === "") {
        return "/extensions/comfyui-ysnodes/assets/notify.mp3";
    }
    const f = file.trim();

    // Full URL — use as-is
    if (f.startsWith("http://") || f.startsWith("https://")) {
        return f;
    }

    // Absolute local path — browsers cannot fetch local files; warn and fallback
    if (f.startsWith("/") || /^[A-Za-z]:\\/.test(f)) {
        console.warn(
            "[PurgeNotificationSG] Local paths cannot be played by the browser. " +
            "Place the file in comfyui-ysnodes/web/assets/ and use just the filename."
        );
        return "/extensions/comfyui-ysnodes/assets/notify.mp3";
    }

    // Bare filename → serve from this pack's web/assets/
    return `/extensions/comfyui-ysnodes/assets/${f}`;
}

function playSound(file, volume) {
    const url = resolveSoundUrl(file);
    console.log(`[PurgeNotificationSG] Playing sound: ${url} vol=${volume}`);

    const audio = new Audio(url);
    audio.volume = Math.max(0, Math.min(1, volume ?? 0.5));
    audio.play().catch((err) => {
        console.warn(`[PurgeNotificationSG] Audio play failed: ${err.message}`);
        console.warn("  → Try clicking anywhere on the ComfyUI page first.");
        console.warn("  → Browsers require a user gesture before playing audio.");
    });
}

// ---------------------------------------------------------------------------
// Browser OS notification
// ---------------------------------------------------------------------------

async function requestNotificationPermission() {
    if (!("Notification" in window)) {
        console.warn("[PurgeNotificationSG] Browser does not support notifications.");
        return false;
    }
    if (Notification.permission === "granted") return true;
    if (Notification.permission === "denied") {
        console.warn(
            "[PurgeNotificationSG] Notification permission denied by browser. " +
            "Go to browser Settings → Site Settings → Notifications → Allow this site."
        );
        return false;
    }
    // "default" — ask the user
    const result = await Notification.requestPermission();
    return result === "granted";
}

async function sendSystemNotification(text) {
    const granted = await requestNotificationPermission();
    if (!granted) return;

    try {
        const notif = new Notification("ComfyUI", {
            body: text || "✅ Workflow complete",
            icon: "/favicon.ico",
            requireInteraction: false,
        });
        // Auto-close after 8 seconds
        setTimeout(() => notif.close(), 8000);
        console.log(`[PurgeNotificationSG] OS notification fired: "${text}"`);
    } catch (err) {
        console.warn(`[PurgeNotificationSG] Notification failed: ${err.message}`);
    }
}

// ---------------------------------------------------------------------------
// Register WebSocket listeners + extension
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "comfyui-ysnodes.PurgeNotification",

    async setup() {
        console.log("[PurgeNotificationSG] Registering WebSocket listeners...");

        // ── psg_playsound ────────────────────────────────────────────────
        api.addEventListener("psg_playsound", (event) => {
            const { file, volume } = event.detail ?? {};
            console.log("[PurgeNotificationSG] Received psg_playsound:", event.detail);
            playSound(file, volume);
        });

        // ── psg_notification ─────────────────────────────────────────────
        api.addEventListener("psg_notification", (event) => {
            const { text } = event.detail ?? {};
            console.log("[PurgeNotificationSG] Received psg_notification:", event.detail);
            sendSystemNotification(text);
        });

        // ── Pre-request notification permission on page load ─────────────
        // This ensures the browser permission dialog appears before the
        // first workflow runs, not mid-generation.
        await requestNotificationPermission();

        console.log("[PurgeNotificationSG] ✅ Listeners ready.");
    },
});
