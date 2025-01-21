from __future__ import annotations
import torch
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

class StyleModelApplySimple(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "clip_vision": ("CLIP_VISION", {"tooltip": "The CLIP Vision model used for encoding the image."}),
                "style_model": ("STYLE_MODEL", {"tooltip": "The style model to be applied."}),
                "image": ("IMAGE", {"tooltip": "The image to encode and apply style."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "The intensity of the style application."}),
                "strength_type": (["multiply", "attn_bias"], {"default": "multiply", "tooltip": "Method to apply the style."}),
                "crop": (["center", "none"], {"default": "center", "tooltip": "Crop method for the CLIP Vision encoder."}),
                "conditioning": ("CONDITIONING", {"tooltip": "The initial conditioning to modify with the style."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "process"
    CATEGORY = "ComfyUI-NeuralMedia/Style Model"
    DESCRIPTION = "Encodes an image with CLIP Vision and applies a style model to modify the conditioning."

    def process(self, clip_vision, style_model, image, strength, strength_type, crop, conditioning):
        throw_exception_if_processing_interrupted()

        # Encode image using CLIP Vision
        crop_image = True if crop == "center" else False
        clip_vision_output = clip_vision.encode_image(image, crop=crop_image)

        # Extract style model conditioning
        style_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            style_cond *= strength

        # Modify the conditioning
        modified_conditioning = []
        for cond in conditioning:
            text_cond, keys = cond
            keys = keys.copy()

            if strength_type == "attn_bias" and strength != 1.0:
                attn_bias = torch.log(torch.tensor([strength]))
                n_txt = text_cond.shape[1]
                n_style = style_cond.shape[1]

                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((text_cond.shape[0], n_txt + n_style, n_txt + n_style), dtype=torch.float16)
                new_mask = torch.zeros((text_cond.shape[0], n_txt + n_style, n_txt + n_style), dtype=torch.float16)
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt:] = attn_bias
                new_mask[:, n_txt:, :n_txt] = attn_bias

                keys["attention_mask"] = new_mask.to(text_cond.device)

            modified_conditioning.append([torch.cat((text_cond, style_cond), dim=1), keys])

        return (modified_conditioning,)

NODE_CLASS_MAPPINGS = {
    "StyleModelApplySimple": StyleModelApplySimple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplySimple": "🖌️ Style Model Apply (Simple)"
}
