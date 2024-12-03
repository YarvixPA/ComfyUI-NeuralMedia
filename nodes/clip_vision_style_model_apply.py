import comfy.clip_vision
import comfy.sd
import torch

class CLIPVisionAndStyleApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "The existing conditioning to be modified."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "The CLIP Vision model provided by another node."}),
                "style_model": ("STYLE_MODEL", {"tooltip": "The style model provided by another node."}),
                "image": ("IMAGE", {"tooltip": "The image to be encoded by the CLIP Vision model."}),
                "crop": (["center", "none"], {"tooltip": "Whether to crop the image to its center."}),
                "strength_type": (["multiply"], {"tooltip": "The method to apply strength to the style model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "The influence of the style model on the conditioning."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "ComfyUI-NeuralMedia"
    DESCRIPTION = "Applies a CLIP Vision model and a style model to modify conditioning."

    def apply(self, conditioning, clip_vision, style_model, image, crop="center", strength_type="multiply", strength=1.0):
        crop_image = crop == "center"

        clip_encoded = clip_vision.encode_image(image, crop=crop_image)

        style_conditioning = style_model.get_cond(clip_encoded).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            style_conditioning *= strength

        updated_conditioning = []
        for cond in conditioning:
            combined_cond = torch.cat((cond[0], style_conditioning), dim=1)
            updated_conditioning.append([combined_cond, cond[1].copy()])

        return (updated_conditioning,)

NODE_CLASS_MAPPINGS = {
    "CLIPVisionAndStyleApply": CLIPVisionAndStyleApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPVisionAndStyleApply": "🖌️ CLIP Vision & Style Apply"
}
