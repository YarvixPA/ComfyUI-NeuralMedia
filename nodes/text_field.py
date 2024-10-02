class TextFieldNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_multiline"
    CATEGORY = "ComfyUI-NeuralMedia"

    def text_multiline(self, text):
        return text,

NODE_CLASS_MAPPINGS = {
    "TextFieldNode": TextFieldNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFieldNode": "🖌️ Text Field"
}
