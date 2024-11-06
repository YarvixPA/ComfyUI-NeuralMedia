import torch
import cv2
import numpy as np

# Function to convert a tensor [B, H, W, C] to a NumPy array [H, W, C]
def tensor_to_numpy(tensor):
    array = tensor.squeeze(0).cpu().numpy()  # Remove the batch dimension
    array = np.clip(array, 0, 1) * 255  # Ensure values are in the correct range
    return array.astype(np.uint8)  # Convert to 8-bit

# Function to convert a NumPy array [H, W, C] to a tensor [B, H, W, C]
def numpy_to_tensor(array):
    array = array.astype(np.float32) / 255.0  # Normalize back to the range [0, 1]
    tensor = torch.from_numpy(array).unsqueeze(0)  # Reintroduce the batch dimension
    return tensor

# Function to expand the image
def expand_image(image, top, bottom, left, right):
    h, w, c = image.shape
    new_h = h + top + bottom
    new_w = w + left + right
    expanded_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    expanded_image[top:top+h, left:left+w] = image
    return expanded_image

# Generate the mask for the expanded areas
def generate_mask(image, top, bottom, left, right):
    h, w, _ = image.shape
    new_h = h + top + bottom
    new_w = w + left + right
    mask = np.ones((new_h, new_w), dtype=np.float32)
    mask[top:top+h, left:left+w] = 0
    return mask

# Apply blur to the mask
def blur_mask(mask, blur_radius):
    ksize = max(1, int(blur_radius) * 2 + 1)  # Kernel size must be odd
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)

# Apply cropping to the image and mask
def apply_crop_masking(image, mask, padding, resize_size):
    y_indices, x_indices = np.where(mask >= 0.5)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, mask  # No valid area
    top, bottom = max(0, y_indices.min() - padding), min(image.shape[0], y_indices.max() + padding)
    left, right = max(0, x_indices.min() - padding), min(image.shape[1], x_indices.max() + padding)
    height, width = bottom - top, right - left
    if height > width:
        diff = height - width
        left, right = max(0, left - diff // 2), min(image.shape[1], right + diff - diff // 2)
    elif width > height:
        diff = width - height
        top, bottom = max(0, top - diff // 2), min(image.shape[0], bottom + diff - diff // 2)
    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    if resize_size > 0:
        # Resize maintaining aspect ratio
        original_height, original_width = cropped_image.shape[:2]
        aspect_ratio = original_width / original_height
        if original_height > original_width:
            new_height = resize_size
            new_width = int(resize_size * aspect_ratio)
        else:
            new_width = resize_size
            new_height = int(resize_size / aspect_ratio)
        cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        cropped_mask = cv2.resize(cropped_mask, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return cropped_image, cropped_mask

# Outpainting node class
class OutpaintNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_blur": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0}),
                "top": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "left": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "crop_masking": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "padding_crop_masking": ("INT", {"default": 32, "min": 0, "max": 100}),
                "resize_crop_masking": (["none", "1024", "2048", "4096"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("WELD_DATA", "IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Inpaint and Outpaint"
    
    def execute(self, image, mask_blur, top, bottom, left, right, crop_masking, padding_crop_masking, resize_crop_masking):
        # 1. Convert the image to NumPy [H, W, C]
        np_image = tensor_to_numpy(image)
        
        # 2. Expand the image
        expanded_image = expand_image(np_image, top, bottom, left, right)
        
        # 3. Generate the mask
        mask = generate_mask(np_image, top, bottom, left, right)
        
        # 4. Blur the mask
        blurred_mask = blur_mask(mask, mask_blur)
        
        # 5. Crop the image and mask if crop_masking is enabled
        weld_data = {
            "original_image": expanded_image,
            "top": top,
            "bottom": top + np_image.shape[0],
            "left": left,
            "right": left + np_image.shape[1],
            "expanded_h": expanded_image.shape[0],
            "expanded_w": expanded_image.shape[1]
        }
        
        if crop_masking:
            resize_value = 0 if resize_crop_masking == "none" else int(resize_crop_masking)
            expanded_image, blurred_mask = apply_crop_masking(expanded_image, blurred_mask, padding_crop_masking, resize_value)
            y_indices, x_indices = np.where(blurred_mask >= 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                weld_data.update({
                    "top": y_indices.min(),
                    "bottom": y_indices.max(),
                    "left": x_indices.min(),
                    "right": x_indices.max()
                })
        
        # 6. Apply inpainting using Navier-Stokes
        inpainted_image = cv2.inpaint(expanded_image, (blurred_mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
        
        # 7. Convert back to tensor
        final_image_tensor = numpy_to_tensor(inpainted_image)
        final_mask_tensor = torch.from_numpy(blurred_mask).unsqueeze(0)
        
        # 8. Return the image, mask, and weld_data
        return (weld_data, final_image_tensor, final_mask_tensor)

NODE_CLASS_MAPPINGS = {
    "OutpaintNode": OutpaintNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OutpaintNode": "🖌️ Outpaint"
}
