"""Tests for Black-Anandan optical flow."""
import numpy as np
import pytest
from optical_flow.methods.ba import BAOpticalFlow
from optical_flow.robust.robust_function import RobustFunction


class TestBlackAnandan:
    """Test Black-Anandan optical flow estimation."""

    def test_zero_flow_identical_frames(self):
        """Identical frames should produce near-zero flow."""
        H, W = 32, 32
        img = np.random.rand(H, W) * 255
        ba = BAOpticalFlow()
        ba.images = np.stack([img, img], axis=2)
        ba.pyramid_levels = 1
        ba.gnc_iters = 1
        ba.max_iters = 3
        uv = ba.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)
        np.testing.assert_allclose(uv, 0, atol=0.1)

    def test_custom_penalties(self):
        """BA should accept custom penalty configurations."""
        ba = BAOpticalFlow()
        ba.rho_spatial_u = [RobustFunction('charbonnier', 1e-3),
                           RobustFunction('charbonnier', 1e-3)]
        ba.rho_spatial_v = [RobustFunction('charbonnier', 1e-3),
                           RobustFunction('charbonnier', 1e-3)]
        ba.rho_data = RobustFunction('charbonnier', 1e-3)

        H, W = 24, 24
        img = np.random.rand(H, W) * 255
        ba.images = np.stack([img, img], axis=2)
        ba.pyramid_levels = 1
        ba.gnc_iters = 1
        ba.max_iters = 2
        uv = ba.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)

    def test_output_shape(self, rubberwhale_gray):
        """Output flow should match image dimensions."""
        gray1, gray2 = rubberwhale_gray
        ba = BAOpticalFlow()
        ba.images = np.stack([gray1, gray2], axis=2)
        ba.pyramid_levels = 2
        ba.gnc_iters = 1
        ba.max_iters = 2
        H, W = gray1.shape
        uv = ba.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)
