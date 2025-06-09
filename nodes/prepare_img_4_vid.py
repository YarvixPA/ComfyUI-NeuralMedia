from nodes import MAX_RESOLUTION
import comfy.utils
import torch

class Prepimg2Vid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":             ("IMAGE",),
                "resolution":        (["480p (SD)", "720p (HD)", "1080p (Full HD)", "1440p (Quad HD)", "2160p (4K)"], {"default": "720p (HD)"}),
                "aspect_ratio":      (["16:9 (Horizontal)", "3:2 (Horizontal)", "1:1 (Square)", "2:3 (Vertical)", "9:16 (Vertical)"], {"default": "16:9 (Horizontal)"}),
                "horizontal_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "vertical_offset":   ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
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
        # Convert to CHW for processing
        tensor = image.permute(0, 3, 1, 2)
        _, _, orig_h, orig_w = tensor.shape

        # Upscale to base height for consistent cropping
        base_h = 1080
        base_w = int(orig_w * (base_h / orig_h))
        upscaled = comfy.utils.lanczos(tensor, base_w, base_h)

        # Determine crop region for desired AR
        ar_w, ar_h = self._AR_MAP[aspect_ratio]
        target_ratio = ar_w / ar_h

        # Compute maximal crop dimensions that fit
        crop_h = base_h
        crop_w = int(crop_h * target_ratio)
        if crop_w > base_w:
            crop_w = base_w
            crop_h = int(crop_w / target_ratio)

        # Center and apply offsets (clamped within bounds)
        cen_x = (base_w - crop_w) // 2
        cen_y = (base_h - crop_h) // 2
        x0 = min(max(cen_x + horizontal_offset, 0), base_w - crop_w)
        y0 = min(max(cen_y + vertical_offset, 0), base_h - crop_h)

        # Crop and convert back to HWC
        cropped = upscaled[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        cropped = cropped.permute(0, 2, 3, 1)

        # Final resize to target resolution
        out_h = self._H_MAP[resolution]
        out_w = int(out_h * target_ratio)
        # Ensure even dimensions
        out_w -= out_w % 2
        out_h -= out_h % 2

        # Resize
        final_tensor = cropped.permute(0, 3, 1, 2)
        resized = comfy.utils.lanczos(final_tensor, out_w, out_h)
        output = resized.permute(0, 2, 3, 1)

        # Clamp and return exact dimensions
        return (torch.clamp(output, 0.0, 1.0), out_w, out_h)

NODE_CLASS_MAPPINGS = {
    "Prepimg2Vid": Prepimg2Vid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prepimg2Vid": "🖌️ Prepare img2vid"
}
