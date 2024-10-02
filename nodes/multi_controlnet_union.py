import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
from nodes import ControlNetApplyAdvanced
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# This node applies a single ControlNet model to multiple inputs.
class MultiControlnetUnion:

    controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "add_more_switches": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),  # Flag to add more switches to the last ControlNet
                "controlnet": (cls.controlnets,),  # ControlNet selection input
            },
            "optional": {
                "multicontrolnet_stack": ("MULTICONTROLNET_STACK",),  # Stack of previous ControlNets
                "switch_1": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_type_1": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_1": ("IMAGE",),
                "switch_2": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_type_2": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_2": ("IMAGE",),
                "switch_3": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_type_3": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "controlnet_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_3": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("MULTICONTROLNET_STACK",)
    RETURN_NAMES = ("multicontrolnet_stack",)
    FUNCTION = "controlnet_stacker"
    CATEGORY = "ComfyUI-NeuralMedia/ControlNets"

    def controlnet_stacker(self, add_more_switches, controlnet, switch_1, controlnet_type_1, controlnet_strength_1, start_percent_1, end_percent_1,
                           switch_2, controlnet_type_2, controlnet_strength_2, start_percent_2, end_percent_2,
                           switch_3, controlnet_type_3, controlnet_strength_3, start_percent_3, end_percent_3,
                           image_1=None, image_2=None, image_3=None, multicontrolnet_stack=None):

        # If nothing has been configured in this node, pass the previous stack without modifications
        if not any([switch_1, switch_2, switch_3]) and all([image_1 is None, image_2 is None, image_3 is None]) and multicontrolnet_stack:
            return multicontrolnet_stack,

        # If 'add_more_switches' is activated, use the last model from the previous stack
        if add_more_switches and multicontrolnet_stack:
            controlnet_model = multicontrolnet_stack[-1][0]  # Use the last model from the stack
        elif controlnet != "None":
            # Load a new ControlNet if 'add_more_switches' is set to No
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
            controlnet_model = comfy.controlnet.load_controlnet(controlnet_path)
        else:
            print("You must select a ControlNet++ model or use a stack")
            return multicontrolnet_stack or []

        # If 'add_more_switches' is activated, controlnet_list must continue accumulating
        controlnet_list = multicontrolnet_stack[:] if multicontrolnet_stack else []

        # Function to add a controlnet if the switch is "On" and the image is not None
        def add_controlnet(switch, image, controlnet_type, controlnet_strength, start_percent, end_percent):
            if switch and image is not None:
                controlnet_type_number = UNION_CONTROLNET_TYPES.get(controlnet_type, -1)
                controlnet_list.append((controlnet_model, image, controlnet_strength, start_percent, end_percent, controlnet_type_number))

        # Add the controlnets according to the activated switches
        add_controlnet(switch_1, image_1, controlnet_type_1, controlnet_strength_1, start_percent_1, end_percent_1)
        add_controlnet(switch_2, image_2, controlnet_type_2, controlnet_strength_2, start_percent_2, end_percent_2)
        add_controlnet(switch_3, image_3, controlnet_type_3, controlnet_strength_3, start_percent_3, end_percent_3)

        return controlnet_list,

NODE_CLASS_MAPPINGS = {
    "MultiControlnetUnion": MultiControlnetUnion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiControlnetUnion": "🖌️ Multi-ControlnetUnion"
}
