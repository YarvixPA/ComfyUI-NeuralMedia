import torch
import cv2
import numpy as np

# Helper function to convert a tensor [B, H, W, C] or [B, H, W] to NumPy array [H, W, C] or [H, W]
def tensor_to_numpy(tensor):
    """
    Converts a tensor to a NumPy array and ensures that the values are in the range [0, 1].
    """
    array = tensor.squeeze(0).cpu().numpy()
    return np.clip(array, 0, 1).astype(np.float32)

# Helper function to convert a NumPy array [H, W, C] or [H, W] to a tensor [B, H, W, C] or [B, H, W]
def numpy_to_tensor(array):
    """
    Converts a NumPy array to a tensor ensuring that the values are in the range [0, 1].
    """
    return torch.from_numpy(np.clip(array, 0, 1).astype(np.float32)).unsqueeze(0)

# Function to apply blur to the mask
def apply_blur(mask, blur_radius):
    """
    Applies Gaussian blur to the mask.
    """
    ksize = max(1, int(blur_radius) * 2 + 1)  # Kernel size must be odd
    return np.clip(cv2.GaussianBlur(mask, (ksize, ksize), 0), 0, 1)

# Function to handle the mask mode
def apply_mask_mode(mask, mode):
    """
    Applies the mask mode, inverting it if necessary.
    """
    return 1.0 - mask if mode == "inpaint not masked" else mask

# Function to expand the mask (dilation)
def apply_mask_expand(mask, expand_pixels):
    """
    Expands the mask by dilating the masked area by 'expand_pixels'.
    """
    if expand_pixels > 0:
        kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return np.clip(mask, 0, 1)

# Function to progressively/interpolatively apply the masked content
def apply_mask_content(image, mask, content):
    """
    Applies the mask content, such as 'latent nothing', 'fill', or 'original'.
    """
    if content == "latent nothing":
        fill_color = np.full_like(image, 0.5)
        return (1 - mask[..., None]) * image + mask[..., None] * fill_color
    elif content == "fill":
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        if mask_uint8.ndim == 3:
            mask_uint8 = mask_uint8[:, :, 0]
        inpainted = cv2.inpaint(image_uint8, mask_uint8, 3, cv2.INPAINT_NS)
        inpainted_float = inpainted.astype(np.float32) / 255.0
        # Apply Gaussian blur
        blurred_inpainted = cv2.GaussianBlur(inpainted_float, (5, 5), 0)
        return (1 - mask[..., None]) * image + mask[..., None] * blurred_inpainted
    return image  # For 'original' or default content

# Function to apply cropping to the image and mask if 'crop_masking' is True
def apply_crop_masking(image, mask, padding, resize_size):
    """
    Crops the image and mask while maintaining a 1:1 aspect ratio.
    """
    y_indices, x_indices = np.where(mask >= 0.5)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, mask, (0, 0, image.shape[1], image.shape[0])  # No valid area, return full image and mask
    
    top, bottom = max(0, y_indices.min() - padding), min(image.shape[0], y_indices.max() + padding)
    left, right = max(0, x_indices.min() - padding), min(image.shape[1], x_indices.max() + padding)
    
    # Ensure a 1:1 aspect ratio by adjusting the cropping box
    height, width = bottom - top, right - left
    if height > width:
        diff = height - width
        left = max(0, left - diff // 2)
        right = min(image.shape[1], right + (diff - diff // 2))
    elif width > height:
        diff = width - height
        top = max(0, top - diff // 2)
        bottom = min(image.shape[0], bottom + (diff - diff // 2))
    
    # Perform the crop
    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    
    # Store the cropping coordinates
    crop_coords = (top, bottom, left, right)
    
    # Resize with Lanczos if necessary, maintaining aspect ratio
    if resize_size > 0:
        cropped_image = cv2.resize(cropped_image, (resize_size, resize_size), interpolation=cv2.INTER_LANCZOS4)
        cropped_mask = cv2.resize(cropped_mask, (resize_size, resize_size), interpolation=cv2.INTER_LANCZOS4)
    
    return cropped_image, cropped_mask, crop_coords

# Inpainting node class
class InpaintNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_mode": (["inpaint masked", "inpaint not masked"], {"default": "inpaint masked"}),
                "mask_expand": ("INT", {"default": 0, "min": 0, "max": 100}),
                "mask_blur": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0}),
                "mask_content": (["original", "fill", "latent nothing"], {"default": "original"}),
                "crop_masking": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "padding_crop_masking": ("INT", {"default": 32, "min": 0, "max": 100}),
                "resize_crop_masking": (["none", "1024", "2048", "4096"], {"default": "none"}),
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("WELD_DATA", "IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Inpaint and Outpaint"

    def execute(self, image, mask, crop_masking, mask_mode, mask_expand, mask_blur, mask_content, padding_crop_masking, resize_crop_masking):
        np_image = tensor_to_numpy(image)
        np_mask = tensor_to_numpy(mask)
        if np_mask.ndim == 3:
            np_mask = np_mask[:, :, 0]
        np_mask = np.clip(np_mask, 0, 1)
        np_mask = apply_mask_mode(np_mask, mask_mode)
        np_mask = apply_mask_expand(np_mask, mask_expand)
        blurred_mask = apply_blur(np_mask, mask_blur)
        result_image = apply_mask_content(np_image, blurred_mask, mask_content)
        
        weld_data = {
            "original_image": np_image,
            "top": 0, "bottom": np_image.shape[0],
            "left": 0, "right": np_image.shape[1]
        }

        if crop_masking:
            resize_value = 0 if resize_crop_masking == "none" else int(resize_crop_masking)
            result_image, blurred_mask, crop_coords = apply_crop_masking(result_image, blurred_mask, padding_crop_masking, resize_value)
            top, bottom, left, right = crop_coords
            weld_data.update({
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right
            })
        
        final_image_tensor = numpy_to_tensor(result_image)
        final_mask_tensor = numpy_to_tensor(blurred_mask)
        
        return (weld_data, final_image_tensor, final_mask_tensor)

NODE_CLASS_MAPPINGS = {
    "InpaintNode": InpaintNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintNode": "🖌️ Inpaint"
}
