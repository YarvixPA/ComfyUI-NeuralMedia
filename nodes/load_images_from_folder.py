import os
from PIL import Image, ImageOps
import numpy as np
import torch

class LoadImagesFromFolderNode: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "max_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "reload": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "load_images"
    CATEGORY = "ComfyUI-NeuralMedia"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'reload' in kwargs and kwargs['reload']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, max_images: int = 0, skip_images: int = 0, reload=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # Start at skip_images index
        dir_files = dir_files[skip_images:]

        images = []
        limit_images = max_images > 0
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= max_images:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)
            image_count += 1

        return images,

NODE_CLASS_MAPPINGS = {
    "LoadImagesFromFolderNode": LoadImagesFromFolderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromFolderNode": "🖌️ Load Images from Folder"
}
