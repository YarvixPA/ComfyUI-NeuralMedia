class FrameCalculator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("INT", {"default": 30}),
                "duration_seconds": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("frame_rate", "num_frames")
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Utils/Video"

    def execute(self, fps=30, duration_seconds=0):
        fps = int(fps)
        duration_seconds = int(duration_seconds)

        frame_rate = fps
        num_frames = fps * duration_seconds
        return (frame_rate, num_frames)


NODE_CLASS_MAPPINGS = {
    "FrameCalculator": FrameCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameCalculator": "🖌️ Frame Calculator"
}