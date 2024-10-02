import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
from nodes import ControlNetApplyAdvanced

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# This applies the ControlNet stack.
class MultiControlnetApply:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_positive": ("CONDITIONING", ),
                             "base_negative": ("CONDITIONING",),
                             "switch": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                             "multicontrolnet_stack": ("MULTICONTROLNET_STACK",),
                            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("base_positive", "base_negative", "show_help",)
    FUNCTION = "apply_controlnet_stack"
    CATEGORY = "ComfyUI-NeuralMedia/ControlNets"

    def apply_controlnet_stack(self, base_positive, base_negative, switch, multicontrolnet_stack=None):

        if not switch:
            return (base_positive, base_negative, "ControlNet stack is off.",)

        if multicontrolnet_stack is not None:
            for controlnet_tuple in multicontrolnet_stack:
                if len(controlnet_tuple) == 5:
                    # For nodes like MultiControlnet
                    controlnet_name, image, strength, start_percent, end_percent = controlnet_tuple
                    controlnet_type = None  # No controlnet type in this case
                else:
                    # For nodes like MultiControlnetUnion
                    controlnet_name, image, strength, start_percent, end_percent, controlnet_type = controlnet_tuple

                # Load controlnet model if needed
                if isinstance(controlnet_name, str):
                    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
                    controlnet = comfy.controlnet.load_controlnet(controlnet_path)
                else:
                    controlnet = controlnet_name

                # Apply controlnet type if available
                if controlnet_type is not None and controlnet_type >= 0:
                    controlnet.set_extra_arg("control_type", [controlnet_type])
                else:
                    controlnet.set_extra_arg("control_type", [])

                # Apply controlnet to the conditioning
                controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(base_positive, base_negative,
                                                                                     controlnet, image, strength,
                                                                                     start_percent, end_percent)

                base_positive, base_negative = controlnet_conditioning[0], controlnet_conditioning[1]

        return (base_positive, base_negative, "ControlNet stack applied successfully.")

NODE_CLASS_MAPPINGS = {
    "MultiControlnetApply": MultiControlnetApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiControlnetApply": "🖌️ Multi-Controlnet Apply"
}
