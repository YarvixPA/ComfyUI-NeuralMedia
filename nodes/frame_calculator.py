class FrameCalculator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30.0}),
                "duration_seconds": ("INT",   {"default": 0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("frame_rate", "num_frames")
    FUNCTION = "execute"
    CATEGORY = "ComfyUI-NeuralMedia/Utils/Video"

    def execute(self, fps=30.0, duration_seconds=0):
        fps = float(fps)
        duration_seconds = int(duration_seconds)

        frame_rate = fps
        num_frames = int(round(frame_rate * duration_seconds))
        return (frame_rate, num_frames)

NODE_CLASS_MAPPINGS = {
    "FrameCalculator": FrameCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameCalculator": "🖌️ Frame Calculator"
}
