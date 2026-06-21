"""
nodes/purge_notification.py
================================
Purge & Notification

HOW NOTIFICATIONS WORK IN COMFYUI
------------------------------------
Notifications are a TWO-PART system:

  Python side (this file):
    • Performs the VRAM purge immediately.
    • Sends a WebSocket message to the browser via PromptServer.send_sync().

  JavaScript side (web/purge_notification_sg.js):
    • Registers a listener for the WebSocket message.
    • Plays the audio and/or fires the browser OS notification.

  → BOTH files are required. Without the JS file, no sound/notification fires.

Purge logic  (origin: chflame163/ComfyUI_LayerStyle → T8mars/comfyui-purgevram)
  purge_cache=True   → comfy.model_management.soft_empty_cache()
                       + torch CUDA/MPS cache clear
  purge_models=True  → comfy.model_management.unload_all_models()

Notification logic  (origin: royceschultz/ComfyUI-Notifications)
  system_notification=True → sends "psg_notification" WS event → JS fires OS alert
  play_sound=True          → sends "psg_playsound"    WS event → JS plays audio
  mode="always"            → fires every execution
  mode="on_empty_queue"    → fires only when ComfyUI queue is fully empty
"""

import gc
import torch
import comfy.model_management
from server import PromptServer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALERT_MODES = ["always", "on_empty_queue"]
DEFAULT_NOTIFICATION_TEXT = "✅ Workflow complete"
DEFAULT_SOUND_FILE = "notify.mp3"

# WebSocket event names — must match exactly what purge_notification_sg.js listens for
WS_EVENT_PLAYSOUND    = "psg_playsound"
WS_EVENT_NOTIFICATION = "psg_notification"

# ---------------------------------------------------------------------------
# Purge helpers
# ---------------------------------------------------------------------------

def _purge_cache():
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def _purge_models():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()

# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def _should_notify(mode: str) -> bool:
    if mode == "always":
        return True
    try:
        remaining = len(PromptServer.instance.prompt_queue.queue)
        return remaining == 0
    except Exception:
        return True

def _send_notification(text: str):
    try:
        PromptServer.instance.send_sync(WS_EVENT_NOTIFICATION, {"text": text})
    except Exception as e:
        print(f"[PurgeNotification] send_notification error: {e}")

def _send_playsound(file: str, volume: float):
    try:
        PromptServer.instance.send_sync(WS_EVENT_PLAYSOUND, {"file": file, "volume": volume})
    except Exception as e:
        print(f"[PurgeNotification] send_playsound error: {e}")

# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class PurgeNotification:
    """
    🔔 Purge & Notification

    Place at the END of your workflow to:
      1. Purge VRAM cache and/or unload models
      2. Play a sound in the browser
      3. Send a browser OS notification

    Requires web/purge_notification_sg.js to be present in the
    comfyui-ysnodes folder for sound/notification to work.
    """

    CATEGORY  = "YSNodes/utility"
    FUNCTION  = "run"
    OUTPUT_NODE = True

    RETURN_TYPES  = ("*",)
    RETURN_NAMES  = ("signal",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "purge_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear CUDA/MPS cache and ComfyUI soft cache.",
                }),
                "purge_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload ALL models from VRAM. Next run reloads from disk.",
                }),
                "alert_mode": (ALERT_MODES, {
                    "default": "always",
                    "tooltip": "always = notify every run. on_empty_queue = only when queue is empty.",
                }),
                "system_notification": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Send a browser OS notification popup.",
                }),
                "notification_text": ("STRING", {
                    "default": DEFAULT_NOTIFICATION_TEXT,
                    "multiline": False,
                    "tooltip": "Text shown in the browser OS notification.",
                }),
                "play_sound": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Play an audio file in the browser tab.",
                }),
                "sound_volume": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Playback volume 0.0 (mute) to 1.0 (full).",
                }),
                "sound_file": ("STRING", {
                    "default": DEFAULT_SOUND_FILE,
                    "multiline": False,
                    "tooltip": (
                        "Sound to play. Can be:\n"
                        "  • Filename in comfyui-ysnodes/web/assets/ (e.g. notify.mp3)\n"
                        "  • Full local path (e.g. C:\\Windows\\Media\\alarm.wav)\n"
                        "  • URL (e.g. https://example.com/alert.mp3)"
                    ),
                }),
            },
            "optional": {
                "signal": ("*", {
                    "tooltip": "Connect last node output here to ensure correct execution order.",
                }),
            },
        }

    def run(self, purge_cache, purge_models, alert_mode,
            system_notification, notification_text,
            play_sound, sound_volume, sound_file,
            signal=None):

        # ── Purge ───────────────────────────────────────────────────────
        if purge_cache:
            _purge_cache()
            print("[PurgeNotification] Cache purged.")

        if purge_models:
            _purge_models()
            print("[PurgeNotification] Models unloaded.")

        # ── Notifications ───────────────────────────────────────────────
        if _should_notify(alert_mode):
            if system_notification:
                _send_notification(notification_text)
                print(f"[PurgeNotification] Notification sent: '{notification_text}'")
            if play_sound:
                _send_playsound(sound_file, sound_volume)
                print(f"[PurgeNotification] Play-sound sent: '{sound_file}' vol={sound_volume}")

        return (signal,)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS        = {"PurgeNotification": PurgeNotification}
NODE_DISPLAY_NAME_MAPPINGS = {"PurgeNotification": "Purge & Notification"}
