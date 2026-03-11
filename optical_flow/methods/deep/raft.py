"""RAFT: Recurrent All Pairs Field Transforms for Optical Flow.

Teed, Z. & Deng, J. "RAFT: Recurrent All Pairs Field Transforms for
Optical Flow." European Conference on Computer Vision (ECCV), 2020.

Source: https://github.com/princeton-vl/RAFT
License: BSD-3-Clause
"""
import numpy as np
from optical_flow.methods.deep._base import DeepFlowBase


class RAFTFlow(DeepFlowBase):
    """RAFT optical flow wrapper.

    Args:
        model_name: Pretrained model variant. One of 'raft-things' (default),
            'raft-sintel', 'raft-kitti', 'raft-small'.
        device: Torch device ('cuda', 'cpu', 'mps', or None for auto-detect).
        iters: Number of recurrent update iterations (default 20).
    """

    def __init__(self, model_name='raft-things', device=None, iters=20):
        super().__init__(model_name=model_name, device=device)
        self.iters = iters
        self._small = 'small' in (model_name or '')

    def _load_model(self):
        import torch
        from optical_flow.methods.deep._model_cache import ensure_model

        model_path = ensure_model('raft', self.model_name)
        device = self._get_device()

        if self._small:
            from optical_flow.methods.deep._vendor.raft.raft import RAFT as RAFTNet
            # Small model uses different architecture params
            import argparse
            args = argparse.Namespace(
                small=True, mixed_precision=False, alternate_corr=False,
                dropout=0.0,
            )
            model = RAFTNet(args)
        else:
            from optical_flow.methods.deep._vendor.raft.raft import RAFT as RAFTNet
            import argparse
            args = argparse.Namespace(
                small=False, mixed_precision=False, alternate_corr=False,
                dropout=0.0,
            )
            model = RAFTNet(args)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Handle DataParallel state dicts (keys prefixed with 'module.')
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
        t1 = torch.from_numpy(im1_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)
        t2 = torch.from_numpy(im2_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # Pad to multiple of 8
        H, W = t1.shape[2], t1.shape[3]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            t1 = torch.nn.functional.pad(t1, [0, pad_w, 0, pad_h], mode='replicate')
            t2 = torch.nn.functional.pad(t2, [0, pad_w, 0, pad_h], mode='replicate')

        with torch.no_grad():
            _, flow_up = self._model(t1, t2, iters=self.iters, test_mode=True)

        # Crop padding and convert to numpy (H, W, 2)
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()[:H, :W, :]
        return flow.astype(np.float64)
