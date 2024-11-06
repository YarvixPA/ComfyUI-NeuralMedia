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
    # Resize the cropped image to fit the target area
    target_h, target_w = bottom - top, right - left
    cropped_h, cropped_w = cropped_image.shape[:2]
    if (cropped_h, cropped_w) != (target_h, target_w):
        cropped_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
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
                "mode": (["inpaint", "outpaint"], {"default": "inpaint"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Inpaint and Outpaint"
    
    def execute(self, weld_data, image, mode):
        # Convert the tensors to NumPy arrays
        np_cropped_image = tensor_to_numpy(image)
        np_original_image = tensor_to_numpy(weld_data['original_image'])
        
        # Extract the weld data information
        top, bottom, left, right = weld_data.get('top', 0), weld_data.get('bottom', np_original_image.shape[0]), weld_data.get('left', 0), weld_data.get('right', np_original_image.shape[1])
        
        if mode == "inpaint":
            # Resize the cropped image to match the original dimensions if necessary
            cropped_h, cropped_w = np_cropped_image.shape[:2]
            target_h, target_w = bottom - top, right - left
            
            if (cropped_h, cropped_w) != (target_h, target_w):
                np_cropped_image = cv2.resize(np_cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Weld the cropped image back into the original image
            welded_image = weld_image(np_original_image, np_cropped_image, weld_data)
        
        elif mode == "outpaint":
            # For outpaint, expand the original image and place the cropped image accordingly
            expanded_h = weld_data.get('expanded_h', np_original_image.shape[0])
            expanded_w = weld_data.get('expanded_w', np_original_image.shape[1])
            expanded_image = np.zeros((expanded_h, expanded_w, np_original_image.shape[2]), dtype=np_original_image.dtype)
            
            # Place the original image into the expanded image (before cropping adjustments)
            expanded_image[:np_original_image.shape[0], :np_original_image.shape[1]] = np_original_image
            
            # Resize the cropped image to match the extended region if necessary
            target_h, target_w = bottom - top, right - left
            cropped_h, cropped_w = np_cropped_image.shape[:2]
            
            if (cropped_h, cropped_w) != (target_h, target_w):
                np_cropped_image = cv2.resize(np_cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Place the cropped image onto the extended area only (replace extended area with incoming image)
            expanded_image[top:bottom, left:right] = np_cropped_image
            welded_image = expanded_image
        else:
            raise ValueError("Invalid mode. Must be 'inpaint' or 'outpaint'.")
        
        # Convert the result back to tensor
        final_image_tensor = numpy_to_tensor(welded_image)
        
        return (final_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "WeldInpaintNode": WeldInpaintNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeldInpaintNode": "🖌️ Weld Inpaint"
}
