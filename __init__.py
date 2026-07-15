"""
comfyui-ysnodes
===============
A personal pack of custom ComfyUI nodes by Mr. Yassin NM.

Version: 1.2.0

Adding a new node
-----------------
1. Create a new file inside  comfyui-ysnodes/nodes/
2. Define your node class(es) in that file
3. Add them to  NODE_CLASS_MAPPINGS  and  NODE_DISPLAY_NAME_MAPPINGS  at the
   bottom of that file (same pattern as the existing ones)
4. That's it — this __init__.py discovers them automatically.

Adding frontend JS
------------------
Place .js files in  comfyui-ysnodes/web/
They are auto-served by ComfyUI at /extensions/comfyui-ysnodes/<filename>
and registered via the WEB_DIRECTORY variable below.
"""

import os
import importlib

__version__ = "1.2.0"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# Serve files in web/ to the ComfyUI frontend
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Auto-discover every .py file inside the nodes/ sub-package
_nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")

for _filename in sorted(os.listdir(_nodes_dir)):
    if _filename.startswith("_") or not _filename.endswith(".py"):
        continue
    _module_name = _filename[:-3]  # strip .py
    _module = importlib.import_module(f".nodes.{_module_name}", package=__name__)

    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]