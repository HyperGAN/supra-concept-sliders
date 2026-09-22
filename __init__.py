"""ComfyUI entry stub. Node maps stay empty until COMFYUI.md is implemented."""

if __package__:
    from .comfy_particle import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    from comfy_particle import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
