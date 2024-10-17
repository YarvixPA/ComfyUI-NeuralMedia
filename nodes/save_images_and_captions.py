import os
import numpy as np
from PIL import Image

class SaveCaptionsImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "captions": ("STRING", {"forceInput": True, "multiline": True}),
                "images": ("IMAGE", {"multiline": True}),
                "save_path": ("STRING", {"multiline": False}),
                "prefix_file": ("STRING", {"multiline": False}),
                "delimiter_file": ("STRING", {"default": "_"}),
                "caption_extension": ("STRING", {"default": "txt"}),
                "image_extension": (['png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'bmp'],),
                "image_quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
                "overwrite": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_images_captions"
    CATEGORY = "ComfyUI-NeuralMedia"

    def save_images_captions(self, captions, images, save_path='', prefix_file="", delimiter_file='_', caption_extension='txt', image_extension='png', image_quality=100, lossless_webp="false", overwrite=False):

        os.makedirs(save_path, exist_ok=True)  # Simplified directory creation
        
        if isinstance(captions, str):
            captions = [captions]
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]

        existing_files = set(os.listdir(save_path))

        def get_next_number(prefix, extension):
            matching_files = [f for f in existing_files if f.startswith(prefix) and f.endswith(extension)]
            numbers = [int(f.replace(prefix, '').replace(extension, '').replace(delimiter_file, ''))
                       for f in matching_files if f.replace(prefix, '').replace(extension, '').replace(delimiter_file, '').isdigit()]
            return max(numbers, default=0) + 1

        for idx, (caption, image) in enumerate(zip(captions, images), 1):
            if not caption.strip():
                print(f"Warning: Caption is empty for index {idx}. Skipping...")
                continue

            current_number = get_next_number(f"{prefix_file}{delimiter_file}", f".{image_extension}")
            base_filename = f"{prefix_file}{delimiter_file}{current_number}"
            img_path = os.path.join(save_path, f"{base_filename}.{image_extension}")
            text_path = os.path.join(save_path, f"{base_filename}.{caption_extension}")

            img_array = 255. * (image.cpu().numpy() if hasattr(image, 'cpu') else image)
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            try:
                self.save_image(img, img_path, image_extension, image_quality, lossless_webp)
                print(f"Image file saved to: {img_path}")
            except Exception as e:
                print(f"Failed to save image at {img_path}: {e}")

            try:
                self.writeTextFile(text_path, caption)
            except Exception as e:
                print(f"Unable to save text file `{text_path}`: {e}")

        return {"ui": {"message": f"Saved {len(captions)} text and image pairs."}}

    def save_image(self, img, img_path, image_extension, image_quality, lossless_webp):
        """Helper method to save images based on extension and quality."""
        options = {"quality": image_quality, "optimize": True}
        if image_extension in ["jpg", "jpeg"]:
            img.save(img_path, **options)
        elif image_extension == 'webp':
            img.save(img_path, **options, lossless=(lossless_webp == "true"))
        else:
            img.save(img_path)

    def writeTextFile(self, file, content):
        """Helper method to save the caption text."""
        try:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Text file saved to: {file}")
        except OSError as e:
            print(f"Unable to save text file `{file}`: {e}")

NODE_CLASS_MAPPINGS = {
    "SaveCaptionsImages": SaveCaptionsImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveCaptionsImages": "🖌️ Save Captions & Images"
}
