"""SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow.

Wang, Y., Lipson, L. & Deng, J. "SEA-RAFT: Simple, Efficient, Accurate
RAFT for Optical Flow." European Conference on Computer Vision (ECCV), 2024.

Source: https://github.com/princeton-vl/SEA-RAFT
License: BSD-3-Clause
"""
import numpy as np
from optical_flow.methods.deep._base import DeepFlowBase


class SEARAFTFlow(DeepFlowBase):
    """SEA-RAFT optical flow wrapper.

    Args:
        model_name: Pretrained model variant (default 'sea-raft-things').
        device: Torch device ('cuda', 'cpu', 'mps', or None for auto-detect).
        iters: Number of recurrent update iterations (default 12).
    """

    def __init__(self, model_name='sea-raft-things', device=None, iters=12):
        super().__init__(model_name=model_name, device=device)
        self.iters = iters

    def _load_model(self):
        import torch
        import argparse
        from optical_flow.methods.deep._model_cache import ensure_model

        model_path = ensure_model('sea-raft', self.model_name)
        device = self._get_device()

        from optical_flow.methods.deep._vendor.sea_raft.raft import RAFT as SEARAFTNet

        # Config matching MemorySlices/Tartan-C-T-TSKH-spring540x960-M (spring-M)
        args = argparse.Namespace(
            pretrain='resnet34',
            initial_dim=64,
            block_dims=[64, 128, 256],
            num_blocks=2,
            radius=4,
            dim=128,
            iters=self.iters,
            use_var=True,
            var_min=0,
            var_max=10,
            dropout=0,
            mixed_precision=False,
            corr_levels=4,
        )
        model = SEARAFTNet(args)

        if str(model_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path), device=str(device))
        else:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Handle DataParallel state dicts
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        self._model = model
        self._device = device

    def _inference(self, im1, im2):
        import torch

        self._ensure_model()
        device = self._device

        # Convert from float64 [0,255] to uint8 then to float32 tensor [0,255]
        im1_uint8 = np.clip(im1, 0, 255).astype(np.uint8)
        im2_uint8 = np.clip(im2, 0, 255).astype(np.uint8)

        # Ensure 3-channel
        if im1_uint8.ndim == 2:
            im1_uint8 = np.stack([im1_uint8] * 3, axis=-1)
            im2_uint8 = np.stack([im2_uint8] * 3, axis=-1)

        # (H, W, 3) -> (1, 3, H, W) float32
        # SEA-RAFT forward() handles normalization and padding internally
        t1 = torch.from_numpy(im1_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)
        t2 = torch.from_numpy(im2_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            result = self._model(t1, t2, iters=self.iters, test_mode=True)
            flow_up = result['flow'][-1]

        # Convert to numpy (H, W, 2)
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        return flow.astype(np.float64)
