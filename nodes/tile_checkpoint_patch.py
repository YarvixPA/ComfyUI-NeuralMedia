import copy
import torch
from torch.nn import Conv2d
from torch.nn import functional as F
from torch import Tensor
from typing import Optional

class TileCheckpointPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
                "copy_option": (["Make a copy", "Modify in place"],),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE")  # Removed CLIP from the outputs
    FUNCTION = "tile_checkpoint_patch"
    CATEGORY = "ComfyUI-NeuralMedia/Tile"

    def tile_checkpoint_patch(self, model, vae, tiling, copy_option):
        # Duplicate or modify in place
        if copy_option == "Modify in place":
            model_copy = model
            vae_copy = vae
        else:
            model_copy = copy.deepcopy(model)
            vae_copy = copy.deepcopy(vae)
        
        # Apply tiling to the model
        self.apply_tiling(model_copy.model, tiling)
        
        # Apply tiling to the VAE
        self.apply_tiling(vae_copy.first_stage_model, tiling)
        
        # Return the model and vae (CLIP removed)
        return (model_copy, vae_copy)

    def apply_tiling(self, model, tiling):
        if tiling == "enable":
            self.make_circular_asymm(model, True, True)
        elif tiling == "x_only":
            self.make_circular_asymm(model, True, False)
        elif tiling == "y_only":
            self.make_circular_asymm(model, False, True)
        else:
            self.make_circular_asymm(model, False, False)

    def make_circular_asymm(self, model, tileX: bool, tileY: bool):
        for layer in [
            layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
        ]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            layer._conv_forward = self.__replacementConv2DConvForward.__get__(layer, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        working = F.pad(input, self.paddingX, mode=self.padding_modeX)
        working = F.pad(working, self.paddingY, mode=self.padding_modeY)
        return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)

NODE_CLASS_MAPPINGS = {
    "TileCheckpointPatchNode": TileCheckpointPatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileCheckpointPatchNode": "🖌️ Tile Checkpoint Patch"
}
