import torch
import safetensors.torch
import folder_paths

class VacePatchLoader:
    """Apply a VACE .safetensors patch to a diffusion model (layers only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base diffusion model."}),
                "vace_patch_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "VACE patch (.safetensors) file."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model with VACE layers applied.",)
    FUNCTION = "apply_patch"
    CATEGORY = "ComfyUI-NeuralMedia/Model/Patch/Wan2.1"
    DESCRIPTION = "Overwrite model tensors with those in a VACE patch."

    def apply_patch(self, model, vace_patch_name):
        # Load patch
        path = folder_paths.get_full_path_or_raise("checkpoints", vace_patch_name)
        patch = safetensors.torch.load_file(path, device="cpu")
        # Clone model and find its nn.Module
        patched = model.clone()
        for attr in ("model", "diffusion_model", "inner_model"):  
            if hasattr(patched, attr):
                nn_mod = getattr(patched, attr)
                break
        else:
            raise AttributeError("Internal nn.Module not found in MODEL.")
        # Apply patch (ignore unmatched keys)
        missing, _ = nn_mod.load_state_dict(patch, strict=False)
        if missing:
            print(f"[VACE Patch] {len(patch)} tensors loaded, {len(missing)} missing.")
        else:
            print(f"[VACE Patch] Injected {len(patch)} tensors.")
        return (patched,)

NODE_CLASS_MAPPINGS = {
    "VacePatchLoader": VacePatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VacePatchLoader": "🖌️ Apply VACE Patch",
}
