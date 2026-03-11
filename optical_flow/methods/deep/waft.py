"""WAFT: Warping-Alone Field Transforms for Optical Flow.

Wang, Y. & Deng, J. "WAFT: Warping-Alone Field Transforms for Optical Flow."
International Conference on Learning Representations (ICLR), 2026.

Source: https://github.com/princeton-vl/WAFT
License: BSD-3-Clause
"""
import numpy as np
from optical_flow.methods.deep._base import DeepFlowBase


class WAFTFlow(DeepFlowBase):
    """WAFT optical flow wrapper.

    WAFT replaces cost volumes with high-resolution warping, achieving
    state-of-the-art results on Spring, Sintel, and KITTI benchmarks
    with up to 4.1x speedup over comparable methods.

    Requires: PyTorch >= 2.0, timm, gdown, and optionally xformers.
    DepthAnythingV2 weights are auto-downloaded on first use.

    Available model_name variants:
        'waft-things'       General-purpose (TartanAir+Chairs+Things). Default.
        'waft-sintel'       Fine-tuned on Sintel (best for natural scenes).
        'waft-kitti'        Fine-tuned on KITTI (best for driving/outdoor).
        'waft-spring-540p'  Fine-tuned on Spring at 540p.
        'waft-spring-1080p' Fine-tuned on Spring at 1080p (best for HD video).

    Args:
        model_name: Pretrained model variant (default 'waft-things').
        device: Torch device ('cuda', 'cpu', 'mps', or None for auto-detect).
        iters: Number of refinement iterations (default 12).
        dav2_backbone: DepthAnythingV2 backbone ('vits', 'vitb', 'vitl').
            Default 'vits'.
        network_backbone: WAFT network backbone ('vitt', 'vits', 'vitb', 'vitl').
            Default 'vits'.
    """

    def __init__(self, model_name='waft-things', device=None, iters=12,
                 dav2_backbone='vits', network_backbone='vits'):
        super().__init__(model_name=model_name, device=device)
        self.iters = iters
        self.dav2_backbone = dav2_backbone
        self.network_backbone = network_backbone

    def _load_model(self):
        import torch
        import argparse
        from optical_flow.methods.deep._model_cache import ensure_model
        from optical_flow.methods.deep._vendor.waft.vitwarp_v8 import ViTWarpV8
        from optical_flow.methods.deep._vendor.waft.inference_tools import InferenceWrapper
        from optical_flow.methods.deep._vendor.waft.utils.utils import load_ckpt

        model_path = ensure_model('waft', self.model_name)
        device = self._get_device()

        args = argparse.Namespace(
            algorithm='vitwarp',
            dav2_backbone=self.dav2_backbone,
            network_backbone=self.network_backbone,
            iters=self.iters,
            image_size=[540, 960],
            var_max=10.0,
            var_min=-10.0,
        )

        model = ViTWarpV8(args)
        load_ckpt(model, model_path)
        model = model.to(device)
        model.eval()

        self._model = InferenceWrapper(model, scale=0.0, train_size=args.image_size,
                                       pad_to_train_size=False, tiling=False)
        self._device = device

    def _inference(self, im1, im2):
        import torch

        self._ensure_model()
        device = self._device

        # Convert from float64 [0,255] to uint8 then float32 tensor
        im1_uint8 = np.clip(im1, 0, 255).astype(np.uint8)
        im2_uint8 = np.clip(im2, 0, 255).astype(np.uint8)

        # Ensure 3-channel
        if im1_uint8.ndim == 2:
            im1_uint8 = np.stack([im1_uint8] * 3, axis=-1)
            im2_uint8 = np.stack([im2_uint8] * 3, axis=-1)

        # (H, W, 3) -> (1, 3, H, W) float32, range [0, 255]
        t1 = torch.from_numpy(im1_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)
        t2 = torch.from_numpy(im2_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)

        H, W = t1.shape[2], t1.shape[3]

        with torch.no_grad():
            output = self._model.calc_flow(t1, t2, iters=self.iters)
            flow_up = output['flow'][-1]

        # Convert to numpy (H, W, 2)
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()[:H, :W, :]
        return flow.astype(np.float64)
