import torch
import torch.nn.functional as F
import comfy.utils

class StitchImagesAndMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_size": ("BOOLEAN", {"default": True}),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "spacing_color": (["white", "black", "red", "green", "blue"], {"default": "white"}),
            },
            "optional": {
                "mask1": ("MASK",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
            },
        }

    RETURN_TYPES = ("UNSTITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("unstitch", "IMAGE", "MASK")
    FUNCTION = "stitch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"
    DESCRIPTION = (
        "Stitches two images and their corresponding masks with optional spacing. "
        "If mask1 or mask2 is not provided, a full-black mask of matching size is created. "
        "spacing_color applies only to the image spacing (mask spacing remains black)."
    )

    def stitch(self, image1, direction, match_size, spacing_width, spacing_color, image2=None, mask1=None, mask2=None):
        # Prepare info for unstitch
        info = {
            "direction": direction,
            "match_size": match_size,
            "spacing_width": spacing_width,
            "spacing_color": spacing_color,
            "shape1": image1.shape,
            "shape2": image2.shape if image2 is not None else None,
        }
        B, H1, W1, C = image1.shape
        # Default masks: full black (zeros)
        if mask1 is None:
            mask1 = torch.zeros((B, H1, W1), device=image1.device)
        # If no second image, return just the first
        if image2 is None:
            return (info, image1, mask1)
        if mask2 is None:
            mask2 = torch.zeros_like(mask1)

        # Align batches
        image1, image2 = self._align_batch(image1, image2)
        mask1, mask2 = self._align_batch_mask(mask1, mask2)

        # Resize or pad both images and masks
        if match_size:
            image2 = self._resize_to_match(image2, image1, direction)
            mask1 = self._resize_mask(mask1, (image1.shape[1], image1.shape[2]))
            mask2 = self._resize_mask(mask2, (image2.shape[1], image2.shape[2]))
        else:
            image1, image2 = self._pad_to_match(image1, image2, direction)
            mask1 = self._pad_mask(mask1, (image1.shape[1], image1.shape[2]), direction)
            mask2 = self._pad_mask(mask2, (image2.shape[1], image2.shape[2]), direction)

        # Align image channels only
        image1, image2 = self._align_channels(image1, image2)

        # Prepare concat lists
        imgs = [image1, image2] if direction in ["right", "down"] else [image2, image1]
        masks_ch = [mask1.unsqueeze(-1), mask2.unsqueeze(-1)] if direction in ["right", "down"] else [mask2.unsqueeze(-1), mask1.unsqueeze(-1)]

        # Add spacing if needed
        if spacing_width > 0:
            color_map = {
                "white": 1.0,
                "black": 0.0,
                "red": (1.0, 0.0, 0.0),
                "green": (0.0, 1.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
            }
            val = color_map.get(spacing_color, 0.0)
            if direction in ["left", "right"]:
                H_out = max(image1.shape[1], image2.shape[1])
                img_space = torch.zeros((B, H_out, spacing_width, C), device=image1.device)
                mask_space = torch.zeros((B, H_out, spacing_width, 1), device=mask1.device)
            else:
                W_out = max(image1.shape[2], image2.shape[2])
                img_space = torch.zeros((B, spacing_width, W_out, C), device=image1.device)
                mask_space = torch.zeros((B, spacing_width, W_out, 1), device=mask1.device)
            if isinstance(val, tuple):
                for i, cval in enumerate(val):
                    if i < img_space.shape[-1]:
                        img_space[..., i] = cval
            else:
                img_space[..., :min(3, img_space.shape[-1])] = val
            imgs.insert(1, img_space)
            masks_ch.insert(1, mask_space)

        dim = 2 if direction in ["left", "right"] else 1
        stitched_image = torch.cat(imgs, dim=dim)
        stitched_mask = torch.cat(masks_ch, dim=dim).squeeze(-1)

        return (info, stitched_image, stitched_mask)

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
    def _align_batch_mask(m1, m2):
        b1, b2 = m1.shape[0], m2.shape[0]
        max_b = max(b1, b2)
        if b1 < max_b:
            m1 = torch.cat([m1, m1[-1:].repeat(max_b - b1, 1, 1)], dim=0)
        if b2 < max_b:
            m2 = torch.cat([m2, m2[-1:].repeat(max_b - b2, 1, 1)], dim=0)
        return m1, m2

    @staticmethod
    def _resize_to_match(img, ref, direction):
        h_ref, w_ref = ref.shape[1:3]
        if direction in ["left", "right"]:
            target_h, target_w = h_ref, int(h_ref * img.shape[2] / img.shape[1])
        else:
            target_w, target_h = w_ref, int(w_ref * img.shape[1] / img.shape[2])
        return comfy.utils.common_upscale(
            img.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
        ).movedim(1, -1)

    @staticmethod
    def _resize_mask(mask, target_hw):
        H, W = target_hw
        m = mask.unsqueeze(1).float()
        return F.interpolate(m, size=(H, W), mode='nearest').squeeze(1)

    @staticmethod
    def _pad_to_match(img1, img2, direction):
        h1, w1 = img1.shape[1:3]
        h2, w2 = img2.shape[1:3]
        if direction in ["left", "right"]:
            t_h = max(h1, h2)
            pad1 = ((t_h - h1) // 2, t_h - h1 - (t_h - h1) // 2)
            pad2 = ((t_h - h2) // 2, t_h - h2 - (t_h - h2) // 2)
            img1 = F.pad(img1, (0, 0, pad1[0], pad1[1]), mode='constant', value=0.0)
            img2 = F.pad(img2, (0, 0, pad2[0], pad2[1]), mode='constant', value=0.0)
        else:
            t_w = max(w1, w2)
            pad1 = ((t_w - w1) // 2, t_w - w1 - (t_w - w1) // 2)
            pad2 = ((t_w - w2) // 2, t_w - w2 - (t_w - w2) // 2)
            img1 = F.pad(img1, (pad1[0], pad1[1], 0, 0), mode='constant', value=0.0)
            img2 = F.pad(img2, (pad2[0], pad2[1], 0, 0), mode='constant', value=0.0)
        return img1, img2

    @staticmethod
    def _pad_mask(mask, target_hw, direction):
        h, w = mask.shape[1:3]
        if direction in ["left", "right"]:
            t_h = target_hw[0]
            pad = ((t_h - h) // 2, t_h - h - (t_h - h) // 2)
            return F.pad(mask.unsqueeze(1), (0, 0, pad[0], pad[1]), mode='constant', value=0.0).squeeze(1)
        else:
            t_w = target_hw[1]
            pad = ((t_w - w) // 2, t_w - w - (t_w - w) // 2)
            return F.pad(mask.unsqueeze(1), (pad[0], pad[1], 0, 0), mode='constant', value=0.0).squeeze(1)

    @staticmethod
    def _align_channels(img1, img2):
        c1, c2 = img1.shape[-1], img2.shape[-1]
        max_c = max(c1, c2)
        if c1 < max_c:
            pad = torch.ones(*img1.shape[:-1], max_c - c1, device=img1.device)
            img1 = torch.cat([img1, pad], dim=-1)
        if c2 < max_c:
            pad = torch.ones(*img2.shape[:-1], max_c - c2, device=img2.device)
            img2 = torch.cat([img2, pad], dim=-1)
        return img1, img2

NODE_CLASS_MAPPINGS = {
    "StitchImagesAndMask": StitchImagesAndMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StitchImagesAndMask": "🖌️ Stitch Images and Mask"
}