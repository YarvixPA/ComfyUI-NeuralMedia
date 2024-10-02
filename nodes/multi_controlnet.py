import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
from nodes import ControlNetApplyAdvanced
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# This node applies multiple ControlNets in a chained manner.
class MultiControlnet:

    controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {  # multicontrolnet_stack is now optional
                "multicontrolnet_stack": ("MULTICONTROLNET_STACK",),  # Stack of previous ControlNets (optional)
                "switch_1": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_1": (cls.controlnets,),
                "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_1": ("IMAGE",),
                #
                "switch_2": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_2": (cls.controlnets,),
                "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image_2": ("IMAGE",),
                #
                "switch_3": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "controlnet_3": (cls.controlnets,),
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

    def controlnet_stacker(self, multicontrolnet_stack=None, switch_1=False, controlnet_1="None", controlnet_strength_1=1.0, start_percent_1=0.0, end_percent_1=1.0,
                           switch_2=False, controlnet_2="None", controlnet_strength_2=1.0, start_percent_2=0.0, end_percent_2=1.0,
                           switch_3=False, controlnet_3="None", controlnet_strength_3=1.0, start_percent_3=0.0, end_percent_3=1.0,
                           image_1=None, image_2=None, image_3=None):
        
        # If nothing has been configured in this node and there is no previous multicontrolnet_stack, return empty
        if not any([switch_1, switch_2, switch_3]) and all([image_1 is None, image_2 is None, image_3 is None]) and not multicontrolnet_stack:
            return [],

        # Initialize the stack with the previous multicontrolnet_stack if it exists
        controlnet_list = multicontrolnet_stack[:] if multicontrolnet_stack else []

        # Function to add a controlnet if the switch is "On" and the image is not None
        def add_controlnet(switch, controlnet, image, controlnet_strength, start_percent, end_percent):
            if switch and image is not None and controlnet != "None":
                controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
                controlnet_model = comfy.controlnet.load_controlnet(controlnet_path)
                controlnet_list.append((controlnet_model, image, controlnet_strength, start_percent, end_percent))

        # Add the controlnets for each activated switch
        add_controlnet(switch_1, controlnet_1, image_1, controlnet_strength_1, start_percent_1, end_percent_1)
        add_controlnet(switch_2, controlnet_2, image_2, controlnet_strength_2, start_percent_2, end_percent_2)
        add_controlnet(switch_3, controlnet_3, image_3, controlnet_strength_3, start_percent_3, end_percent_3)

        return controlnet_list,

NODE_CLASS_MAPPINGS = {
    "MultiControlnet": MultiControlnet
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiControlnet": "🖌️ Multi-Controlnet"
}

