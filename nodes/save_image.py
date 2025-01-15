from __future__ import annotations
import os
import json
import numpy as np
from PIL import Image, PngImagePlugin

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
                "file_format": (["PNG", "JPG", "WebP"], {"default": "PNG", "tooltip": "The format to save the image."})
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
        results = []

        # Validate file format
        file_format_lower = file_format.lower()
        valid_formats = {"png", "jpg", "webp"}
        if file_format_lower not in valid_formats:
            raise ValueError(f"Unsupported format. Please choose from {', '.join(valid_formats)}.")

        for batch_number, image in enumerate(images):
            img_array = np.clip((255. * image.cpu().numpy()), 0, 255).astype(np.uint8)
            mode = "RGBA" if img_array.shape[-1] == 4 else "RGB"

            # Adjust format if necessary
            if mode == "RGBA" and file_format_lower == "jpg":
                file_format_lower = "png"

            # Create PIL image
            img = Image.fromarray(img_array, mode=mode)

            # Add metadata for PNG
            metadata = None
            if not args.disable_metadata and file_format_lower == "png":
                metadata = PngImagePlugin.PngInfo()
                if prompt:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            # Prepare file path and name
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{file_format_lower}"
            file_path = os.path.join(full_output_folder, file)

            # Save image
            save_kwargs = {"pnginfo": metadata, "compress_level": self.compress_level} if file_format_lower == "png" else {}
            img.save(file_path, format=file_format_lower.upper(), **save_kwargs)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "original_format": file_format,
                "final_format": file_format_lower
            })
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "SaveImageNode": SaveImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNode": "🖌️ Save Image"
}
