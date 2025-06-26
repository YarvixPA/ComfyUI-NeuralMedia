import torch
import comfy.utils

class UnstitchImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unstitch": ("UNSTITCH", {"forceInput": True}),
            "image": ("IMAGE",),
            "selection": (["1", "2"], {"default": "1", "tooltip": "Choose which image to output: 1=first, 2=second"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "unstitch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"
    DESCRIPTION = """
Reverses a previous stitching operation, outputting one of the original images based on selection (1 or 2), omitting spacing.
"""

    def unstitch(self, unstitch, image, selection):
        info = unstitch
        direction = info.get("direction")
        spacing = info.get("spacing_width", 0)
        # Determine dimension and original sizes
        if direction in ["left", "right"]:
            dim = 2
            orig1 = info["shape1"][2]
        else:
            dim = 1
            orig1 = info["shape1"][1]
        size = image.shape[dim]
        orig2 = size - spacing - orig1
        # Compute slice indices
        if direction in ["right", "down"]:
            start1, end1 = 0, orig1
            start2, end2 = orig1 + spacing, orig1 + spacing + orig2
        else:
            start1, end1 = size - orig1, size
            start2, end2 = spacing, spacing + orig2
        # Slice helper
        def slice_dim(img, s, e, d):
            sl = [slice(None)] * img.ndim; sl[d] = slice(s, e)
            return img[tuple(sl)]
        first_img = slice_dim(image, start1, end1, dim)
        second_img = slice_dim(image, start2, end2, dim)
        idx = int(selection) - 1
        return ([first_img, second_img][idx],)

NODE_CLASS_MAPPINGS = {
    "UnstitchImages": UnstitchImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnstitchImages": "🖌️ Unstitch Images"
}