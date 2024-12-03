import torch
import torch.nn.functional as F

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c

class InpaintConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "pixels": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "noise_mask": ("BOOLEAN", {"default": True, "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."})
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "ComfyUI-NeuralMedia"

    def encode(self, positive, negative, pixels, vae, mask, noise_mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5

        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}
        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                       "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)

NODE_CLASS_MAPPINGS = {
    "InpaintConditioningNode": InpaintConditioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintConditioningNode": "🖌️ Inpaint Conditioning"
}
