MAX_RESOLUTION = 16384  # Define MAX_RESOLUTION with an appropriate value

class SDXLTextEncoderSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 4096, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 4096, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 4096, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 4096, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP", ),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ComfyUI-NeuralMedia"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text):
        tokens_g = clip.tokenize(text)
        tokens_l = clip.tokenize(text)
        
        if len(tokens_l["l"]) != len(tokens_g["g"]):
            empty = clip.tokenize("")
            while len(tokens_l["l"]) < len(tokens_g["g"]):
                tokens_l["l"] += empty["l"]
            while len(tokens_l["l"]) > len(tokens_g["g"]):
                tokens_g["g"] += empty["g"]
                
        tokens = {"g": tokens_g["g"], "l": tokens_l["l"]}
        
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDXLTextEncoderSimpleNode": SDXLTextEncoderSimpleNode
}

NODE_CLASS_MAPPINGS = {
    "SDXLTextEncoderSimpleNode": SDXLTextEncoderSimpleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLTextEncoderSimpleNode": "🖌️ SDXL Text Encoder Simple"
}
