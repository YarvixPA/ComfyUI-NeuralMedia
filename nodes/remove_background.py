import os
import sys
import math
import importlib
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from torchvision import transforms
from PIL import Image
import numpy as np

# Optimizar precisión para multiplicaciones matriciales
torch.set_float32_matmul_precision('high')

# Configuración de modelos según README
MODEL_CONFIGS = {
    'BiRefNet':         {'repo_id': 'ZhengPeng7/BiRefNet',         'size': (1024, 1024), 'dynamic': False},
    'BiRefNet_lite':    {'repo_id': 'ZhengPeng7/BiRefNet_lite',    'size': (1024, 1024), 'dynamic': False},
    'BiRefNet_lite-2K': {'repo_id': 'ZhengPeng7/BiRefNet_lite-2K', 'size': (2560, 1440), 'dynamic': False},
    'BiRefNet_dynamic': {'repo_id': 'ZhengPeng7/BiRefNet_dynamic','size': None,            'dynamic': True},
}

# Selección de dispositivo
def select_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

# Inicializar modelo vendorizado en ComfyUI/models, sin cache global
def initialize_model(repo_id, model_name, update_model=False):
    base = os.path.join('ComfyUI', 'models', 'ComfyUI-NeuralMedia', 'RemoveBackground')
    target_dir = os.path.join(base, model_name)
    os.makedirs(target_dir, exist_ok=True)
    files = ['model.safetensors', 'config.json', 'BiRefNet_config.py', 'birefnet.py']
    model_found = True
    from huggingface_hub import hf_hub_download
    for f in files:
        dst = os.path.join(target_dir, f)
        if not os.path.exists(dst):
            model_found = False
            hf_hub_download(repo_id=repo_id, filename=f, local_dir=target_dir)
    if model_found:
        print(f"🖌️ Remove Background: {model_name} detectado.")
    elif update_model:
        print(f"🖌️ Remove Background: Modelo '{model_name}' actualizado.")
    else:
        print(f"🖌️ Remove Background: Modelo '{model_name}' descargado.")
    init_py = os.path.join(target_dir, '__init__.py')
    if not os.path.exists(init_py): open(init_py, 'a').close()
    if base not in sys.path:
        sys.path.insert(0, base)
    mod = importlib.import_module(f"{model_name}.birefnet")
    BiRefNetClass = getattr(mod, 'BiRefNet')
    model = BiRefNetClass(bb_pretrained=False)
    state = safetensors_load_file(os.path.join(target_dir, 'model.safetensors'), device='cpu')
    model.load_state_dict(state)
    return model

# Conversiones

def to_pil(tensor):
    arr = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# Para imagen (RGB/RGBA) y máscara (L) en formato CHANNEL-LAST

def pil_to_tensor(pil):
    arr = np.array(pil).astype(np.float32) / 255.0
    if pil.mode == 'L':  # máscara de un canal
        tensor = torch.from_numpy(arr).unsqueeze(0)           # [1, H, W]
    else:  # RGB/RGBA
        tensor = torch.from_numpy(arr).unsqueeze(0)           # [1, H, W, C]
    return tensor

class RemoveBackgroundNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'image': ('IMAGE',),
            'model': (list(MODEL_CONFIGS.keys()), {'default':'BiRefNet'}),
            'background_color':(['transparency','white','black'],{'default':'transparency'}),
            'device':(['auto','cuda','cpu'],{'default':'auto'}),
            'update_model':('BOOLEAN',{'default':False})
        }}
    RETURN_TYPES = ('IMAGE','MASK')
    RETURN_NAMES = ('image','mask')
    FUNCTION = 'background_remove'
    CATEGORY = 'ComfyUI-NeuralMedia'

    def background_remove(self, image, model, background_color, device, update_model):
        cfg = MODEL_CONFIGS[model]
        dev = select_device(device)
        net = initialize_model(cfg['repo_id'], model, update_model).to(dev)
        net.eval()
        if dev == 'cuda': net.half()

        results_img, results_mask = [], []
        # Transform básico
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        for t in image:
            pil = to_pil(t)
            ow, oh = pil.size
            # Preprocesado y padding
            if not cfg['dynamic']:
                tw, th = cfg['size']
                scale = min(tw/ow, th/oh)
                nw, nh = int(ow*scale), int(oh*scale)
                resized = pil.resize((nw, nh), Image.BILINEAR)
                inp = transform(resized).unsqueeze(0).to(dev)
                pad_w = tw - nw
                pad_h = th - nh
                inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
            else:
                inp = transform(pil).unsqueeze(0).to(dev)
                pad_w = math.ceil(ow/32)*32 - ow
                pad_h = math.ceil(oh/32)*32 - oh
                inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
            if dev=='cuda': inp = inp.half()

            # Inferencia
            with torch.no_grad():
                out = net(inp)[-1].sigmoid().cpu()
                out = (out - out.min())/(out.max()-out.min())
            # Postprocesado máscara
            if not cfg['dynamic']:
                m = out[:, :, :nh, :nw]
                m = F.interpolate(m, size=(oh,ow), mode='bilinear', align_corners=False)
            else:
                m = out[:, :, :oh, :ow]
            mask_pil = to_pil(m)

            # Composición
            mode_fg = 'RGBA' if background_color=='transparency' else 'RGB'
            col = (0,0,0,0) if background_color=='transparency' else background_color
            bg = Image.new(mode_fg, (ow,oh), col)
            bg.paste(pil, mask=mask_pil)

            # Agregar tensores channel-last
            results_img.append(pil_to_tensor(bg))
            results_mask.append(pil_to_tensor(mask_pil))

        # Concatenar en batch axis
        return torch.cat(results_img, dim=0), torch.cat(results_mask, dim=0)

NODE_CLASS_MAPPINGS = {
    "RemoveBackgroundNode": RemoveBackgroundNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackgroundNode": "🖌️ Remove Background"
}
