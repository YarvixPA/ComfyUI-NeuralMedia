from __future__ import annotations
import torch
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

class StyleModelApplyAdvanced(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "The existing conditioning to be modified."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "The CLIP Vision model provided by another node."}),
                "style_model": ("STYLE_MODEL", {"tooltip": "The style model provided by another node."}),
                "crop": (["center", "none"], {"tooltip": "Whether to crop the image to its center."}),
                "strength_type": (["multiply", "attn_bias"], {"tooltip": "The method to apply strength to the style model."}),
            },
            "optional": {
                "switch_1": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "image_1": ("IMAGE",),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "switch_2": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "image_2": ("IMAGE",),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "switch_3": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "image_3": ("IMAGE",),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "process"
    CATEGORY = "ComfyUI-NeuralMedia/Style Model"
    DESCRIPTION = "Encodes images with CLIP Vision and applies a style model to modify conditioning for multiple inputs."

    def process(self, conditioning, clip_vision, style_model, crop="center", strength_type="multiply", 
                switch_1=False, image_1=None, strength_1=1.0,
                switch_2=False, image_2=None, strength_2=1.0,
                switch_3=False, image_3=None, strength_3=1.0):

        throw_exception_if_processing_interrupted()

        crop_image = crop == "center"
        style_conditioning = None

        # Process each image if the corresponding switch is enabled
        if switch_1 and image_1 is not None:
            clip_encoded_1 = clip_vision.encode_image(image_1, crop=crop_image)
            style_cond_1 = style_model.get_cond(clip_encoded_1).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            if strength_type == "multiply":
                style_cond_1 *= strength_1
            style_conditioning = style_cond_1 if style_conditioning is None else torch.cat((style_conditioning, style_cond_1), dim=1)

        if switch_2 and image_2 is not None:
            clip_encoded_2 = clip_vision.encode_image(image_2, crop=crop_image)
            style_cond_2 = style_model.get_cond(clip_encoded_2).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            if strength_type == "multiply":
                style_cond_2 *= strength_2
            style_conditioning = style_cond_2 if style_conditioning is None else torch.cat((style_conditioning, style_cond_2), dim=1)

        if switch_3 and image_3 is not None:
            clip_encoded_3 = clip_vision.encode_image(image_3, crop=crop_image)
            style_cond_3 = style_model.get_cond(clip_encoded_3).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            if strength_type == "multiply":
                style_cond_3 *= strength_3
            style_conditioning = style_cond_3 if style_conditioning is None else torch.cat((style_conditioning, style_cond_3), dim=1)

        # If no images are enabled, return the original conditioning
        if style_conditioning is None:
            return (conditioning,)

        updated_conditioning = []
        for cond in conditioning:
            combined_cond = torch.cat((cond[0], style_conditioning), dim=1)
            updated_conditioning.append([combined_cond, cond[1].copy()])

        return (updated_conditioning,)

NODE_CLASS_MAPPINGS = {
    "StyleModelApplyAdvanced": StyleModelApplyAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplyAdvanced": "🖌️ Style Model Apply (Advanced)"
}
