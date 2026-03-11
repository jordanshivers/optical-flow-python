# Vendored from https://github.com/princeton-vl/WAFT (BSD-3-Clause)

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from ..depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingFeature(nn.Module):
    def __init__(self, encoder='vits', pretrained=True):
        super().__init__()
        self.model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.encoder = encoder
        depth_anything = DepthAnythingV2(**self.model_configs[encoder])
        if pretrained:
            import os
            from pathlib import Path
            # Check local path first, then auto-download from HuggingFace
            ckpt_path = os.path.join('depth-anything-ckpts', f'depth_anything_v2_{encoder}.pth')
            if os.path.exists(ckpt_path):
                depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            else:
                # Auto-download from HuggingFace
                cache_dir = Path(os.environ.get('OPTICAL_FLOW_CACHE_DIR',
                                                 Path.home() / '.cache' / 'optical_flow' / 'models'))
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_path = cache_dir / f'depth_anything_v2_{encoder}.pth'
                if not cached_path.exists():
                    _HF_URLS = {
                        'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
                        'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
                        'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
                    }
                    if encoder in _HF_URLS:
                        import urllib.request
                        print(f"Downloading DepthAnythingV2 ({encoder}) weights...")
                        urllib.request.urlretrieve(_HF_URLS[encoder], str(cached_path))
                        print(f"Saved to {cached_path}")
                if cached_path.exists():
                    depth_anything.load_state_dict(torch.load(str(cached_path), map_location='cpu'))
        self.depth_anything = depth_anything

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(
            x, self.depth_anything.intermediate_layer_idx[self.encoder], return_class_token=True)
        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        out, path_1, path_2, path_3, path_4 = self.depth_anything.depth_head.forward(
            features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1': path_1, 'path_2': path_2, 'path_3': path_3, 'path_4': path_4, 'features': features}
