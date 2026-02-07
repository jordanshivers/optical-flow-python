"""Tests for spatiotemporal derivative computation."""
import numpy as np
from optical_flow.utils.derivatives import partial_deriv


class TestPartialDeriv:
    """Test spatiotemporal derivatives."""

    def test_zero_flow_no_motion(self):
        """With identical frames and zero flow, temporal derivative should be zero."""
        H, W = 32, 32
        img = np.random.rand(H, W)
        images = np.stack([img, img], axis=2)
        uv = np.zeros((H, W, 2))
        It, Ix, Iy = partial_deriv(images, uv)
        np.testing.assert_allclose(It, 0, atol=1e-10)

    def test_output_shape(self):
        """Derivatives should match image shape."""
        H, W = 20, 30
        images = np.random.rand(H, W, 2)
        uv = np.zeros((H, W, 2))
        It, Ix, Iy = partial_deriv(images, uv)
        assert It.shape == (H, W)
        assert Ix.shape == (H, W)
        assert Iy.shape == (H, W)

    def test_spatial_deriv_on_gradient(self):
        """Spatial derivative of a linear ramp should be roughly constant."""
        H, W = 64, 64
        x_ramp = np.tile(np.arange(W, dtype=float), (H, 1))
        images = np.stack([x_ramp, x_ramp], axis=2)
        uv = np.zeros((H, W, 2))
        It, Ix, Iy = partial_deriv(images, uv)
        # Interior Ix should be approximately 1.0
        center = Ix[10:-10, 10:-10]
        np.testing.assert_allclose(center, 1.0, atol=0.1)

    def test_bilinear_interpolation(self):
        """Test with bi-linear interpolation method."""
        H, W = 20, 20
        images = np.random.rand(H, W, 2)
        uv = np.zeros((H, W, 2))
        It, Ix, Iy = partial_deriv(images, uv, interp_method='bi-linear')
        assert It.shape == (H, W)
