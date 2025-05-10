import torch
import comfy.utils

class BatchImagesNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "ComfyUI-NeuralMedia/Image"

    def batch(self, image1, image2, image3):
        # Asegurarnos de que todas las imágenes tengan las mismas dimensiones
        target_h, target_w = image1.shape[1], image1.shape[2]

        def resize_if_needed(img):
            if img.shape[1:] != (target_h, target_w):
                # Upscale al tamaño de image1
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1),
                    target_w, target_h,
                    "bilinear", "center"
                ).movedim(1, -1)
            return img

        img2 = resize_if_needed(image2)
        img3 = resize_if_needed(image3)

        # Concatenar los tres batches de imágenes
        out = torch.cat((image1, img2, img3), dim=0)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "BatchImagesNode": BatchImagesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImagesNode": "🖌️ Batch Images"
}