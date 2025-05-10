import torch
import comfy.utils

class BatchImagesNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"

    def batch(self, image1, image2=None, image3=None):
        target_h, target_w = image1.shape[1], image1.shape[2]

        def resize_if_needed(img):
            if img.shape[1:] != (target_h, target_w):
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1),
                    target_w, target_h,
                    "bilinear", "center"
                ).movedim(1, -1)
            return img

        images = [image1]
        if image2 is not None:
            images.append(resize_if_needed(image2))
        if image3 is not None:
            images.append(resize_if_needed(image3))

        out = torch.cat(images, dim=0)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "BatchImagesNode": BatchImagesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImagesNode": "🖌️ Batch Images"
}
