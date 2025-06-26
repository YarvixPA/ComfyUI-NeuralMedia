import torch

class UnstitchImagesAndMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unstitch": ("UNSTITCH", {"forceInput": True}),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "selection": (["1", "2"], {"default": "1", "tooltip": "Choose which slice to output: 1=first, 2=second"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "unstitch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"
    DESCRIPTION = """
Reverses a previous Stitch Images and Mask operation, slicing both IMAGE and MASK according to stored parameters and selection.
"""

    def unstitch(self, unstitch, image, mask, selection):
        # retrieve stitching info
        info = unstitch
        direction = info.get("direction")
        spacing = info.get("spacing_width", 0)
        shape1 = info.get("shape1")

        # determine dimension and original lengths
        if direction in ["left", "right"]:
            dim = 2
            orig1 = shape1[2]
        else:
            dim = 1
            orig1 = shape1[1]

        total = image.shape[dim]
        orig2 = total - orig1 - spacing

        # compute slice ranges
        if direction in ["right", "down"]:
            s1, e1 = 0, orig1
            s2, e2 = orig1 + spacing, orig1 + spacing + orig2
        else:
            s1, e1 = total - orig1, total
            s2, e2 = spacing, spacing + orig2

        # helper to slice a tensor along dim
        def slice_dim(x, s, e, d):
            sl = [slice(None)] * x.ndim
            sl[d] = slice(s, e)
            return x[tuple(sl)]

        img1 = slice_dim(image, s1, e1, dim)
        img2 = slice_dim(image, s2, e2, dim)
        m1   = slice_dim(mask,  s1, e1, dim)
        m2   = slice_dim(mask,  s2, e2, dim)

        idx = int(selection) - 1
        out_img = [img1, img2][idx]
        out_mask = [m1, m2][idx]
        return (out_img, out_mask)

NODE_CLASS_MAPPINGS = {
    "UnstitchImagesAndMask": UnstitchImagesAndMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnstitchImagesAndMask": "🖌️ Unstitch Images and Mask"
}
