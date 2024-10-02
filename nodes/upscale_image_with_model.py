import os
import logging
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import torch
import comfy.utils
import folder_paths

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non-commercial upscale models.")
except:
    pass

class UpscaleImageWithModel:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_image"
    CATEGORY = "ComfyUI-NeuralMedia"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (folder_paths.get_filename_list("upscale_models"),),  # Model selection
                "upscale_by": (
                    "FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01},  # Input for scaling factor
                ),
                "image": ("IMAGE",),  # Input restored image
                "tile_size": (
                    "INT", {"default": 512, "min": 128, "max": 8192, "step": 8},  # Control for tile size
                )
            }
        }

    def upscale_image(self, model_name, upscale_by, image, tile_size):
        # Load the selected model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        upscale_model = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(upscale_model, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0  # Memory estimate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        overlap = 32  # Keep the original overlap value

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile_size, tile_y=tile_size, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                
                # Adjust according to the upscale_by value
                size_diff = upscale_by / upscale_model.scale
                if size_diff != 1:
                    s = comfy.utils.common_upscale(
                        s,
                        width=round(s.shape[3] * size_diff),
                        height=round(s.shape[2] * size_diff),
                        upscale_method="lanczos",
                        crop="disabled",
                    )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile_size //= 2
                if tile_size < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "UpscaleImageWithModel": UpscaleImageWithModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleImageWithModel": "🖌️ Upscale Image with Model"
}
