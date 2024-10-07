import os
import subprocess
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

# Define image transformations based on the model
def get_transform_image(model_name):
    if model_name == "BiRefNet_lite-2K":
        resize_dims = (1440, 2560)
    elif model_name in ["BiRefNet", "BiRefNet_lite"]:
        resize_dims = (1024, 1024)
    else:
        raise ValueError("Unrecognized model.")
    
    # Standard image transformation
    return transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# Get the correct device (auto, cuda, cpu)
def get_device_by_name(device):
    return "cuda" if device == 'auto' and torch.cuda.is_available() else device

# Clone the model repository if it doesn't exist
def clone_model_repo(model_name):
    repo_urls = {
        "BiRefNet": "https://huggingface.co/ZhengPeng7/BiRefNet",
        "BiRefNet_lite": "https://huggingface.co/ZhengPeng7/BiRefNet_lite",
        "BiRefNet_lite-2K": "https://huggingface.co/ZhengPeng7/BiRefNet_lite-2K",
    }

    repo_url = repo_urls.get(model_name)
    if not repo_url:
        raise ValueError("Invalid model selected.")

    target_path = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "BiRefNet", model_name)
    correct_path = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "BiRefNet")
    if not os.path.exists(correct_path):
        os.makedirs(correct_path, exist_ok=True)
    if not os.path.exists(target_path):
        # Print message indicating that the model is being downloaded
        print(f"🖌️ ComfyUI-NeuralMedia downloading ({model_name})")
        # Run git clone silently (suppress output)
        subprocess.run(
            ["git", "clone", repo_url, target_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        # Only show this message if the model has been downloaded (git clone performed)
        print(f"🖌️ ComfyUI-NeuralMedia ({model_name}) has been downloaded in {target_path}")

# Convert tensor to PIL image
def tensor2pil(image):
    return Image.fromarray((image.cpu().numpy().squeeze() * 255).astype(np.uint8))

# Convert PIL image to tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Resize the image based on the model
def resize_image(image, model_name):
    size_map = {
        "BiRefNet": (1024, 1024),
        "BiRefNet_lite": (1024, 1024),
        "BiRefNet_lite-2K": (1440, 2560),
    }
    
    size = size_map.get(model_name)
    if not size:
        raise ValueError("Unrecognized model.")
    
    return image.convert('RGB').resize(size, Image.BILINEAR)

class BiRefNetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "BiRefNet_model": (["--select model--", "BiRefNet", "BiRefNet_lite", "BiRefNet_lite-2K"], {"default": "--select model--"}),  
                "background_color": (["transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", "violet",
                                      "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", "tan", "steelblue",
                                      "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"],
                                      {"default": "transparency"}),  
                "device": (["auto", "cuda", "cpu"], {"default": "auto"})  
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "ComfyUI-NeuralMedia/BiRefNet"
  
    def background_remove(self, image, BiRefNet_model, device, background_color):
        if BiRefNet_model == "select model":
            raise ValueError("Please select a valid model.")

        # Clone the model repository if necessary
        clone_model_repo(BiRefNet_model)

        # Load the selected model
        model_path = os.path.join("ComfyUI", "models", "ComfyUI-NeuralMedia", "BiRefNet", BiRefNet_model)
        birefnet = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        device = get_device_by_name(device)
        
        # Print that the model has been loaded on the selected device
        print(f"🖌️ ComfyUI-NeuralMedia ({BiRefNet_model}) has been loaded on {device}")
        birefnet.to(device)

        # Transformations and image processing
        processed_images, processed_masks = [], []
        transform_image = get_transform_image(BiRefNet_model)
        
        for img in image:
            orig_image = tensor2pil(img)
            w, h = orig_image.size
            im_tensor = transform_image(resize_image(orig_image, BiRefNet_model)).unsqueeze(0).to(device)

            with torch.no_grad():
                result = birefnet(im_tensor)[-1].sigmoid().cpu()

            result = torch.squeeze(F.interpolate(result, size=(h, w)))
            result = (result - result.min()) / (result.max() - result.min())
            
            # Convert result to PIL image
            pil_im = Image.fromarray((result * 255).numpy().astype(np.uint8).squeeze())

            # Create background image according to selected background color
            mode = "RGBA" if background_color == 'transparency' else "RGB"
            color = (0, 0, 0, 0) if background_color == 'transparency' else background_color
            new_im = Image.new(mode, pil_im.size, color)
            new_im.paste(orig_image, mask=pil_im)

            processed_images.append(pil2tensor(new_im))
            processed_masks.append(pil2tensor(pil_im))

        return torch.cat(processed_images), torch.cat(processed_masks)

NODE_CLASS_MAPPINGS = {
    "BiRefNetNode": BiRefNetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNetNode": "🖌️ BiRefNet"
}