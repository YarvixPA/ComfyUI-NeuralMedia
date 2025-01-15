from __future__ import annotations
import os
import json
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

class SaveImageNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The images to save."}),
                "filename_prefix": (IO.STRING, {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "file_format": (["PNG", "JPG", "webP"], {"default": "PNG", "tooltip": "The format to save the image."})
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI-NeuralMedia"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", file_format="PNG", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = list()

        file_format = file_format.lower()
        if file_format not in ["png", "jpg", "webp"]:
            raise ValueError("Unsupported format. Please choose from PNG, JPG, or webP.")

        for (batch_number, image) in enumerate(images):
            img_array = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            metadata = None
            if not args.disable_metadata and file_format == "png":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key in extra_pnginfo:
                        metadata.add_text(key, json.dumps(extra_pnginfo[key]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{file_format}"
            file_path = os.path.join(full_output_folder, file)

            if file_format == "png":
                img.save(file_path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(file_path, format=file_format.upper())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "SaveImageNode": SaveImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNode": "🖌️ Save Image"
}
