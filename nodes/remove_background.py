import os
from transformers import AutoModelForImageSegmentation, AutoConfig
from huggingface_hub import hf_hub_download
import torch
from safetensors.torch import load_file as safetensors_load_file
from torchvision import transforms
import numpy as np
from PIL import Image

# Optimize precision for matrix multiplications
torch.set_float32_matmul_precision("high")

MODEL_CONFIGS = {
    "BiRefNet": {"resize_dims": (1024, 1024), "repo_id": "ZhengPeng7/BiRefNet"},
    "BiRefNet_lite": {"resize_dims": (1024, 1024), "repo_id": "ZhengPeng7/BiRefNet_lite"},
    "BiRefNet_lite-2K": {"resize_dims": (1440, 2560), "repo_id": "ZhengPeng7/BiRefNet_lite-2K"},
    "RMBG 2.0": {"resize_dims": (1024, 1024), "repo_id": "briaai/RMBG-2.0"}
}

def get_transform(resize_dims):
    return transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def select_device(device):
    return "cuda" if device == 'auto' and torch.cuda.is_available() else device

def initialize_model(repo_id, model_name, resize_dims, update_model):
    target_dir = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "RemoveBackground", model_name)
    os.makedirs(target_dir, exist_ok=True)

    # Files to check/download
    files_to_download = ["model.safetensors", "config.json", "BiRefNet_config.py", "birefnet.py"]
    model_found = True

    for file_name in files_to_download:
        file_path = os.path.join(target_dir, file_name)
        if not os.path.exists(file_path):
            model_found = False
            hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=target_dir)

    if model_found:
        print(f"🖌️ Remove Background: {model_name} detected.")
    elif update_model:
        print(f"🖌️ Remove Background: Model '{model_name}' updated.")
    else:
        print(f"🖌️ Remove Background: Model '{model_name}' downloaded.")

    # Load model and state_dict
    model_file = os.path.join(target_dir, "model.safetensors")
    config = AutoConfig.from_pretrained(target_dir, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
    model.load_state_dict(safetensors_load_file(model_file, device="cpu"))

    return model

def convert_tensor_to_pil(image_tensor):
    return Image.fromarray((image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))

def convert_pil_to_tensor(image_pil):
    return torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)

class RemoveBackgroundNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "RemoveBackground_model": (["BiRefNet", "BiRefNet_lite", "BiRefNet_lite-2K", "RMBG 2.0 (NO COMMERCIAL USE)"], {"default": "BiRefNet"}),
                "background_color": ([
                    "transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", 
                    "violet", "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", 
                    "tan", "steelblue", "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"
                ], {"default": "transparency"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "update_model": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "ComfyUI-NeuralMedia"

    def background_remove(self, image, RemoveBackground_model, device, background_color, update_model):
        # Map input to internal configuration
        model_map = {
            "RMBG 2.0 (NO COMMERCIAL USE)": "RMBG 2.0",
            "BiRefNet": "BiRefNet",
            "BiRefNet_lite": "BiRefNet_lite",
            "BiRefNet_lite-2K": "BiRefNet_lite-2K",
        }
        model_name = model_map[RemoveBackground_model]
        model_config = MODEL_CONFIGS[model_name]

        model = initialize_model(model_config["repo_id"], model_name, model_config["resize_dims"], update_model)

        device = select_device(device)
        model.to(device).eval()

        if device == "cuda" and torch.cuda.is_available() and model_name != "RMBG 2.0":
            model.half()

        print(f"🖌️ Remove Background: Model '{model_name}' loaded on {device}.")

        processed_images, processed_masks = [], []
        transform = get_transform(model_config["resize_dims"])

        for img in image:
            original_img = convert_tensor_to_pil(img)
            transformed_tensor = transform(original_img.resize(model_config["resize_dims"])).unsqueeze(0).to(device)
            if device == "cuda" and model_name != "RMBG 2.0":
                transformed_tensor = transformed_tensor.half()

            with torch.no_grad():
                result = model(transformed_tensor)[-1].sigmoid().cpu()
                result = (result - result.min()) / (result.max() - result.min())

            mask_img = Image.fromarray((result.squeeze() * 255).numpy().astype(np.uint8))

            if mask_img.size != original_img.size:
                mask_img = mask_img.resize(original_img.size, Image.BILINEAR)

            mode = "RGBA" if background_color == 'transparency' else "RGB"
            color = (0, 0, 0, 0) if background_color == 'transparency' else background_color
            background_img = Image.new(mode, mask_img.size, color)
            background_img.paste(original_img, mask=mask_img)

            processed_images.append(convert_pil_to_tensor(background_img))
            processed_masks.append(convert_pil_to_tensor(mask_img))

        return torch.cat(processed_images), torch.cat(processed_masks)

NODE_CLASS_MAPPINGS = {
    "RemoveBackgroundNode": RemoveBackgroundNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackgroundNode": "🖌️ Remove Background"
}
