import numpy as np
import os
import torch
from PIL import Image
import vtracer
import folder_paths

class VTracerImageVectorizerNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 1, "max": 10}),
                "color_precision": ("INT", {"default": 6, "min": 1, "max": 10}),
                "layer_difference": ("INT", {"default": 16, "min": 1, "max": 30}),
                "corner_threshold": ("INT", {"default": 60, "min": 1, "max": 120}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 3.5, "max": 10.0}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 20}),
                "splice_threshold": ("INT", {"default": 45, "min": 1, "max": 90}),
                "path_precision": ("INT", {"default": 3, "min": 1, "max": 8}),
                "save_format": (["SVG", "PDF"], {"default": "SVG"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ComfyUI-NeuralMedia"

    def execute(self, image, colormode, hierarchical, mode, filter_speckle, color_precision, layer_difference, corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision, save_format, filename_prefix="ComfyUI"):
        # Convert PyTorch tensor to NumPy and remove batch dimension if needed
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[0] == 1:
                image = image.squeeze(0)
        
        # Convert the image to pixels and use VTracer to vectorize directly from pixels
        img = Image.fromarray(image).convert('RGBA')
        pixels = list(img.getdata())

        # Vectorize the image using VTracer and get the SVG as a string
        vectorized_image_svg = vtracer.convert_pixels_to_svg(
            pixels,
            size=img.size,
            colormode=colormode,
            hierarchical=hierarchical,
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            max_iterations=max_iterations,
            splice_threshold=splice_threshold,
            path_precision=path_precision
        )

        # Save the vector file
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = self.get_unique_save_image_path(
            filename_prefix, self.output_dir, image.shape[1], image.shape[0], save_format
        )

        if save_format == "SVG":
            svg_filename = os.path.join(full_output_folder, f"{filename_prefix}_{counter:05}.svg")
            with open(svg_filename, 'w') as svg_file:
                svg_file.write(vectorized_image_svg)
            return {"ui": {"filename": svg_filename, "subfolder": subfolder, "type": self.type}}
        
        elif save_format == "PDF":
            svg_filename = os.path.join(full_output_folder, f"{filename_prefix}_{counter:05}.svg")
            with open(svg_filename, 'w') as svg_file:
                svg_file.write(vectorized_image_svg)
            pdf_filename = svg_filename.replace('.svg', '.pdf')
            self.convert_svg_to_pdf(svg_filename, pdf_filename, image.shape[1], image.shape[0])
            os.remove(svg_filename)  # Remove the temporary SVG file
            return {"ui": {"filename": pdf_filename, "subfolder": subfolder, "type": self.type}}

    def convert_svg_to_pdf(self, svg_filename, pdf_filename, width, height):
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        drawing = svg2rlg(svg_filename)
        scale = min(width / drawing.width, height / drawing.height)
        drawing.width = width
        drawing.height = height
        drawing.scale(scale, scale)
        renderPDF.drawToFile(drawing, pdf_filename)

    def get_unique_save_image_path(self, filename_prefix, output_dir, width, height, save_format):
        counter = 0
        while True:
            full_output_folder, filename, _, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, output_dir, width, height
            )
            extension = ".svg" if save_format == "SVG" else ".pdf"
            final_filename = os.path.join(full_output_folder, f"{filename_prefix}_{counter:05}{extension}")
            if not os.path.exists(final_filename):
                break
            counter += 1
        return full_output_folder, filename, counter, subfolder, filename_prefix

NODE_CLASS_MAPPINGS = {
    "VTracerImageVectorizerNode": VTracerImageVectorizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTracerImageVectorizerNode": "🖌️ VTracer (Image vectorizer)"
}