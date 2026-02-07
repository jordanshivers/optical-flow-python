"""Tests for Horn-Schunck optical flow."""
import numpy as np
import pytest
from optical_flow.methods.hs import HSOpticalFlow


class TestHornSchunck:
    """Test Horn-Schunck optical flow estimation."""

    def test_zero_flow_identical_frames(self):
        """Identical frames should produce near-zero flow."""
        H, W = 32, 32
        img = np.random.rand(H, W) * 255
        hs = HSOpticalFlow()
        hs.images = np.stack([img, img], axis=2)
        hs.lambda_ = 80
        hs.lambda_q = 80
        hs.pyramid_levels = 1
        hs.max_iters = 3
        uv = hs.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)
        np.testing.assert_allclose(uv, 0, atol=0.1)

    def test_output_shape(self, rubberwhale_gray):
        """Output flow should match image dimensions."""
        gray1, gray2 = rubberwhale_gray
        hs = HSOpticalFlow()
        hs.images = np.stack([gray1, gray2], axis=2)
        hs.lambda_ = 80
        hs.lambda_q = 80
        hs.pyramid_levels = 2
        hs.max_iters = 2
        H, W = gray1.shape
        uv = hs.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)

    @pytest.mark.slow
    def test_rubberwhale_accuracy(self, rubberwhale_gray, rubberwhale_gt):
        """HS on RubberWhale should achieve reasonable AAE."""
        from optical_flow.evaluation.metrics import flow_angular_error
        gray1, gray2 = rubberwhale_gray
        tu, tv = rubberwhale_gt

        hs = HSOpticalFlow()
        hs.images = np.stack([gray1, gray2], axis=2)
        hs.lambda_ = 80
        hs.lambda_q = 80
        hs.max_iters = 5
        H, W = gray1.shape
        uv = hs.compute_flow(np.zeros((H, W, 2)))

        aae, _, aepe = flow_angular_error(tu, tv, uv[:,:,0], uv[:,:,1])
        assert aae < 20.0  # Generous bound for basic HS
