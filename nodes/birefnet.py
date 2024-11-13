import os
import subprocess
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")  # Set high precision for matrix multiplications

# Model configuration with dimensions and repository URLs
MODEL_CONFIGS = {
    "BiRefNet": {"resize_dims": (1024, 1024), "repo_url": "https://huggingface.co/ZhengPeng7/BiRefNet"},
    "BiRefNet_lite": {"resize_dims": (1024, 1024), "repo_url": "https://huggingface.co/ZhengPeng7/BiRefNet_lite"},
    "BiRefNet_lite-2K": {"resize_dims": (1440, 2560), "repo_url": "https://huggingface.co/ZhengPeng7/BiRefNet_lite-2K"},
    "RMBG_2.0": {"resize_dims": (1024, 1024), "repo_url": "https://huggingface.co/briaai/RMBG-2.0"}
}

def get_transform(model_name):
    """Define image transformation based on model-specific dimensions."""
    dims = MODEL_CONFIGS[model_name]["resize_dims"]
    return transforms.Compose([
        transforms.Resize(dims),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def select_device(device):
    """Select device based on availability and user preference."""
    return "cuda" if device == 'auto' and torch.cuda.is_available() else device

def manage_model_files(model_name, update_model):
    """Download or update model files as necessary."""
    model_info = MODEL_CONFIGS[model_name]
    target_path = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "BiRefNet", model_name)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if not os.path.exists(os.path.join(target_path, ".git")):
        print(f"🖌️ Downloading model '{model_name}'...")
        subprocess.run(["git", "clone", model_info["repo_url"], target_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🖌️ Model '{model_name}' downloaded successfully.")
    elif update_model:
        print(f"🖌️ Updating model '{model_name}'...")
        subprocess.run(["git", "-C", target_path, "pull"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🖌️ Model '{model_name}' updated successfully.")

def convert_tensor_to_pil(image_tensor):
    """Convert a PyTorch tensor to a PIL image."""
    return Image.fromarray((image_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8))

def convert_pil_to_tensor(image_pil):
    """Convert a PIL image to a PyTorch tensor."""
    return torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)

class BiRefNetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "BiRefNet_model": (["BiRefNet", "BiRefNet_lite", "BiRefNet_lite-2K", "RMBG 2.0 (no commercial use)"], {"default": "BiRefNet"}),
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
    CATEGORY = "ComfyUI-NeuralMedia/BiRefNet"
  
    def background_remove(self, image, BiRefNet_model, device, background_color, update_model):
        # Map user-friendly names to model identifiers
        model_map = {
            "RMBG 2.0 (no commercial use)": "RMBG_2.0",
            "BiRefNet": "BiRefNet",
            "BiRefNet_lite": "BiRefNet_lite",
            "BiRefNet_lite-2K": "BiRefNet_lite-2K"
        }
        model_name = model_map[BiRefNet_model]

        # Handle model file management
        manage_model_files(model_name, update_model)

        # Load the model and move it to the specified device
        model_path = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "BiRefNet", model_name)
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True, revision="main")
        
        device = select_device(device)
        model.to(device)
        
        # Only use half-precision for compatible models on CUDA
        if device == "cuda" and torch.cuda.is_available() and model_name != "RMBG_2.0":
            model.half()
        
        print(f"🖌️ Model '{model_name}' loaded on {device}.")

        processed_images, processed_masks = [], []
        transform = get_transform(model_name)
        
        for img in image:
            # Apply transformations and resize to fit model input requirements
            original_img = convert_tensor_to_pil(img)
            w, h = original_img.size
            transformed_tensor = transform(original_img.resize(MODEL_CONFIGS[model_name]["resize_dims"])).unsqueeze(0).to(device)
            if device == "cuda" and model_name != "RMBG_2.0":
                transformed_tensor = transformed_tensor.half()

            # Run the model without gradient tracking
            with torch.no_grad():
                result = model(transformed_tensor)[-1].sigmoid().cpu()
                result = (result - result.min()) / (result.max() - result.min())  # Normalize result

            mask_img = Image.fromarray((result.squeeze() * 255).numpy().astype(np.uint8))

            # Ensure mask dimensions match the original image
            if mask_img.size != original_img.size:
                mask_img = mask_img.resize(original_img.size, Image.BILINEAR)

            # Create background and paste the original image using the mask
            mode = "RGBA" if background_color == 'transparency' else "RGB"
            color = (0, 0, 0, 0) if background_color == 'transparency' else background_color
            background_img = Image.new(mode, mask_img.size, color)
            background_img.paste(original_img, mask=mask_img)

            # Convert processed images back to tensors
            processed_images.append(convert_pil_to_tensor(background_img))
            processed_masks.append(convert_pil_to_tensor(mask_img))

        # Return concatenated results
        return torch.cat(processed_images), torch.cat(processed_masks)

NODE_CLASS_MAPPINGS = {
    "BiRefNetNode": BiRefNetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNetNode": "🖌️ BiRefNet"
}
