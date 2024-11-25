import comfy.clip_vision
import comfy.sd
import folder_paths
import torch

class CLIPVisionStyleApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_model": (folder_paths.get_filename_list("clip_vision"), {"tooltip": "Name of the CLIP Vision model to load."}),
                "style_model": (folder_paths.get_filename_list("style_models"), {"tooltip": "The style model to load and apply."}),
                "conditioning": ("CONDITIONING", {"tooltip": "Existing conditioning to enhance with the style model."}),
                "image": ("IMAGE", {"tooltip": "The image to encode using the CLIP Vision model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001, "tooltip": "Influence of the style model on the conditioning."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_and_apply"
    CATEGORY = "ComfyUI-NeuralMedia"
    DESCRIPTION = "Loads a CLIP Vision model and a style model, encodes an image, and applies the style model to the conditioning."

    def load_and_apply(self, clip_model, style_model, conditioning, image, strength=1.0):
        # Load and encode with CLIP Vision model
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_model)
        clip_vision_output = comfy.clip_vision.load(clip_path).encode_image(image)

        # Load style model
        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model_instance = comfy.sd.load_style_model(style_model_path)

        # Apply style model to conditioning
        style_conditioning = (strength * style_model_instance.get_cond(clip_vision_output)
                              .flatten(start_dim=0, end_dim=1).unsqueeze(dim=0))
        return ([torch.cat((t[0], style_conditioning), dim=1), t[1].copy()] for t in conditioning),

# Mappings for the node
NODE_CLASS_MAPPINGS = {
    "CLIPVisionStyleApply": CLIPVisionStyleApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPVisionStyleApply": "🖌️ CLIP Vision and Style model Apply"
}
