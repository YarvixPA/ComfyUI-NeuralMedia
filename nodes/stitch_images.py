import torch
import comfy.utils

class StitchImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2}),
                "spacing_color": (["white", "black", "red", "green", "blue"], {"default": "white"}),
            },
            "optional": {"image2": ("IMAGE",)},
        }

    # Outputs: first 'unstitch', then 'IMAGE'
    RETURN_TYPES = ("UNSTITCH", "IMAGE")
    RETURN_NAMES = ("unstitch", "IMAGE")
    FUNCTION = "stitch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"
    DESCRIPTION = """
Stitches two images in the given direction with optional spacing and size/channel alignment.
Outputs an 'unstitch' dict with parameters and original shapes for later reversal, and the stitched IMAGE.
"""

    def stitch(self, image1, direction, match_image_size, spacing_width, spacing_color, image2=None):
        # Prepare info for reversing
        info = {
            "direction": direction,
            "match_image_size": match_image_size,
            "spacing_width": spacing_width,
            "spacing_color": spacing_color,
            "shape1": image1.shape,
            "shape2": image2.shape if image2 is not None else None,
        }
        # If no second image, pass through
        if image2 is None:
            return (info, image1)

        # Align batch counts
        image1, image2 = self._align_batch(image1, image2)
        # Resize or pad as needed
        if match_image_size:
            image2 = self._resize_to_match(image2, image1, direction)
        else:
            image1, image2 = self._pad_to_match(image1, image2, direction)
        # Align channel counts
        image1, image2 = self._align_channels(image1, image2)

        # Compose list for concatenation
        imgs = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            spacing = self._make_spacing(image1, image2, spacing_width, spacing_color, direction)
            imgs.insert(1, spacing)

        # Concatenate along horizontal (dim=2) or vertical (dim=1)
        dim = 2 if direction in ["left", "right"] else 1
        stitched = torch.cat(imgs, dim=dim)

        return (info, stitched)

    @staticmethod
    def _align_batch(img1, img2):
        b1, b2 = img1.shape[0], img2.shape[0]
        max_b = max(b1, b2)
        if b1 < max_b:
            img1 = torch.cat([img1, img1[-1:].repeat(max_b - b1, 1, 1, 1)])
        if b2 < max_b:
            img2 = torch.cat([img2, img2[-1:].repeat(max_b - b2, 1, 1, 1)])
        return img1, img2

    @staticmethod
    def _resize_to_match(img, ref, direction):
        h_ref, w_ref = ref.shape[1:3]
        if direction in ["left", "right"]:
            target_h, target_w = h_ref, int(h_ref * img.shape[2] / img.shape[1])
        else:
            target_w, target_h = w_ref, int(w_ref * img.shape[1] / img.shape[2])
        return comfy.utils.common_upscale(img.movedim(-1,1), target_w, target_h, "lanczos", "disabled").movedim(1,-1)

    @staticmethod
    def _pad_to_match(img1, img2, direction):
        h1, w1 = img1.shape[1:3]; h2, w2 = img2.shape[1:3]
        if direction in ["left","right"]:
            t_h = max(h1,h2)
            pads = [((t_h-h1)//2,(t_h-h1+1)//2), ((t_h-h2)//2,(t_h-h2+1)//2)]
            img1 = torch.nn.functional.pad(img1,(0,0,pads[0][0],pads[0][1]),mode='constant',value=0.0)
            img2 = torch.nn.functional.pad(img2,(0,0,pads[1][0],pads[1][1]),mode='constant',value=0.0)
        else:
            t_w = max(w1,w2)
            pads = [((t_w-w1)//2,(t_w-w1+1)//2), ((t_w-w2)//2,(t_w-w2+1)//2)]
            img1 = torch.nn.functional.pad(img1,(pads[0][0],pads[0][1],0,0),mode='constant',value=0.0)
            img2 = torch.nn.functional.pad(img2,(pads[1][0],pads[1][1],0,0),mode='constant',value=0.0)
        return img1, img2

    @staticmethod
    def _align_channels(img1, img2):
        c1, c2 = img1.shape[-1], img2.shape[-1]
        max_c = max(c1,c2)
        if c1<max_c:
            img1 = torch.cat([img1, torch.ones(*img1.shape[:-1],max_c-c1,device=img1.device)],dim=-1)
        if c2<max_c:
            img2 = torch.cat([img2, torch.ones(*img2.shape[:-1],max_c-c2,device=img2.device)],dim=-1)
        return img1, img2

    @staticmethod
    def _make_spacing(img1, img2, width, color, direction):
        w = width + (width%2)
        if direction in ["left","right"]:
            shape = (img1.shape[0], max(img1.shape[1],img2.shape[1]), w, img1.shape[-1])
        else:
            shape = (img1.shape[0], w, max(img1.shape[2],img2.shape[2]), img1.shape[-1])
        spacing = torch.zeros(shape,device=img1.device)
        cmap = {"white":1.0,"black":0.0,"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1)}
        val = cmap[color]
        if isinstance(val,tuple):
            for i,c in enumerate(val): spacing[...,i]=c
        else:
            spacing[...,:min(3,spacing.shape[-1])]=val
        if spacing.shape[-1]==4: spacing[...,3]=1.0
        return spacing

NODE_CLASS_MAPPINGS = {
    "StitchImages": StitchImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StitchImages": "🖌️ Stitch Images"
}