import torch
import numpy as np
import cv2

# Function to convert a tensor [B, H, W, C] or [B, H, W] to NumPy array [H, W, C] or [H, W]
def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() > 1 and tensor.shape[0] == 1:
            array = tensor.squeeze(0).cpu().numpy()  # Remove the batch dimension if it's equal to 1
        else:
            array = tensor.cpu().numpy()
    else:
        array = tensor  # If already a NumPy array, return as is
    return np.clip(array, 0, 1).astype(np.float32)

# Function to convert a NumPy array [H, W, C] or [H, W] to a tensor [B, H, W, C] or [B, H, W]
def numpy_to_tensor(array):
    return torch.from_numpy(np.clip(array, 0, 1).astype(np.float32)).unsqueeze(0)  # Add the batch dimension back

# Function to weld the cropped image back into its original position
def weld_image(original_image, cropped_image, weld_data):
    """
    Places the cropped image back into its original position on the original image.
    """
    top, bottom, left, right = weld_data['top'], weld_data['bottom'], weld_data['left'], weld_data['right']
    
    # Ensure the original image has the correct dimensions
    welded_image = original_image.copy()
    welded_image[top:bottom, left:right] = cropped_image
    return welded_image

# Weld Inpaint Node class
class WeldInpaintNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weld_data": ("WELD_DATA",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Inpaint and Outpaint"
    
    def execute(self, weld_data, image):
        # Check if weld_data is None or crop_masking is not enabled
        if weld_data is None or not weld_data.get('crop_masking_enabled', False):
            raise RuntimeError("🖌️ ComfyUI-NeuralMedia - Weld Inpaint is only usable when crop masking is enabled")
        
        # Convert the tensors to NumPy arrays
        np_cropped_image = tensor_to_numpy(image)
        np_original_image = tensor_to_numpy(weld_data['original_image'])
        
        # Extract the weld data information
        top, bottom, left, right = weld_data['top'], weld_data['bottom'], weld_data['left'], weld_data['right']
        
        # Resize the cropped image to match the original dimensions if necessary
        cropped_h, cropped_w = np_cropped_image.shape[:2]
        target_h, target_w = bottom - top, right - left
        
        if (cropped_h, cropped_w) != (target_h, target_w):
            np_cropped_image = cv2.resize(np_cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Weld the cropped image back into the original image
        welded_image = weld_image(np_original_image, np_cropped_image, weld_data)
        
        # Convert the result back to tensor
        final_image_tensor = numpy_to_tensor(welded_image)
        
        return (final_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "WeldInpaintNode": WeldInpaintNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeldInpaintNode": "🖌️ Weld Inpaint"
}
