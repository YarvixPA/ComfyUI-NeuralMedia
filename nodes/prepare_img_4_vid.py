from nodes import MAX_RESOLUTION
import comfy.utils
import torch

class Prepimg2Vid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":              ("IMAGE",),
                "resolution":         (["480p (SD)", "720p (HD)", "1080p (Full HD)", "1440p (Quad HD)", "2160p (4K)"], { "default": "720p (HD)" }),
                "aspect_ratio":       (["16:9 (Horizontal)", "3:2 (Horizontal)", "1:1 (Square)", "2:3 (Vertical)", "9:16 (Vertical)"], { "default": "16:9 (Horizontal)" }),
                "horizontal_offset":  ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "vertical_offset":    ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Utils/Video"

    _H_MAP = {
        "480p (SD)":       480,
        "720p (HD)":       720,
        "1080p (Full HD)": 1080,
        "1440p (Quad HD)": 1440,
        "2160p (4K)":      2160,
    }

    _AR_MAP = {
        "16:9 (Horizontal)": (16, 9),
        "3:2 (Horizontal)":  (3,  2),
        "1:1 (Square)":      (1,  1),
        "2:3 (Vertical)":    (2,  3),
        "9:16 (Vertical)":   (9, 16),
    }

    def execute(self, image, resolution, aspect_ratio, horizontal_offset, vertical_offset):
        # 1) Upscale input to a base height of 1080 for consistent cropping
        base_h = 1080
        _, oh, ow, _ = image.shape
        base_w = round(ow * (base_h / oh))

        t = image.permute(0, 3, 1, 2)
        t = comfy.utils.lanczos(t, base_w, base_h)
        img = t.permute(0, 2, 3, 1)

        # 2) Determine crop region for desired aspect ratio
        ar_w, ar_h = self._AR_MAP[aspect_ratio]
        ratio = ar_w / ar_h

        region_h = base_h
        region_w = round(region_h * ratio)
        if region_w > base_w:
            region_w = base_w
            region_h = round(region_w / ratio)

        # 3) Center + apply offsets (clamped)
        cx = (base_w - region_w) // 2
        cy = (base_h - region_h) // 2
        x0 = max(0, min(cx + horizontal_offset, base_w - region_w))
        y0 = max(0, min(cy + vertical_offset, base_h - region_h))

        # 4) Crop strictly inside the frame
        cropped = img[:, y0 : y0 + region_h, x0 : x0 + region_w, :]

        # 5) Final resize to target resolution height & aspect ratio
        out_h = self._H_MAP[resolution]
        out_w = round(out_h * ratio)

        t2 = cropped.permute(0, 3, 1, 2)
        t2 = comfy.utils.lanczos(t2, out_w, out_h)
        out = t2.permute(0, 2, 3, 1)

        return (torch.clamp(out, 0.0, 1.0), out_w, out_h)

NODE_CLASS_MAPPINGS = {
    "Prepimg2Vid": Prepimg2Vid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prepimg2Vid": "🖌️ Prepare img2vid"
}
