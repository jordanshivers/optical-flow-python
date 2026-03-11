"""Base class for deep learning optical flow methods.

Unlike BaseOpticalFlow (which assumes pyramids, IRLS, penalty functions),
this class provides just the minimal interface needed by estimate_flow().
"""
from abc import ABC, abstractmethod
import numpy as np


class DeepFlowBase(ABC):
    """Lightweight base for deep learning flow methods.

    Subclasses implement _load_model() and _inference() for their
    specific architecture. Models are loaded lazily on first use.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device  # 'cuda', 'cpu', 'mps', or None for auto-detect
        self._model = None

        # These attributes exist so estimate_flow() doesn't crash
        # when it checks them, but they are no-ops for DL methods.
        self.images = None
        self.color_images = None
        self.display = False

        # Set by _estimate_flow_deep() before compute_flow()
        self._im1 = None
        self._im2 = None

    def parse_input_parameter(self, params):
        """Accept parameter overrides from estimate_flow(params=...)."""
        if isinstance(params, dict):
            for key, val in params.items():
                if hasattr(self, key):
                    setattr(self, key, val)

    def compute_flow(self, init=None):
        """Compute flow from the stored image pair.

        Returns:
            uv: (H, W, 2) flow field.
        """
        return self._inference(self._im1, self._im2)

    @abstractmethod
    def _inference(self, im1, im2):
        """Run model inference on an image pair.

        Args:
            im1: (H, W, 3) or (H, W) float64 image, range [0, 255].
            im2: Same format as im1.

        Returns:
            uv: (H, W, 2) float64 flow field.
        """
        pass

    @abstractmethod
    def _load_model(self):
        """Load pretrained model weights. Called lazily on first inference."""
        pass

    def _ensure_model(self):
        """Lazy model loading."""
        if self._model is None:
            self._load_model()

    def _get_device(self):
        """Auto-detect best available device."""
        import torch
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
