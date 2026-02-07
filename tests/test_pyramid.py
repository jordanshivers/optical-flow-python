"""Tests for image pyramid and flow resampling."""
import numpy as np
from optical_flow.utils.pyramid import compute_image_pyramid
from optical_flow.utils.warping import resample_flow
from optical_flow.utils.image_processing import fspecial_gaussian


class TestImagePyramid:
    """Test Gaussian image pyramid."""

    def test_pyramid_levels(self):
        """Pyramid should have the correct number of levels."""
        img = np.random.rand(128, 128)
        f = fspecial_gaussian(5, 1.0)
        pyr = compute_image_pyramid(img, f, n_levels=4, ratio=0.5)
        assert len(pyr) == 4

    def test_pyramid_decreasing_size(self):
        """Each level should be smaller than the previous."""
        img = np.random.rand(128, 128)
        f = fspecial_gaussian(5, 1.0)
        pyr = compute_image_pyramid(img, f, n_levels=4, ratio=0.5)
        for i in range(1, len(pyr)):
            assert pyr[i].shape[0] <= pyr[i-1].shape[0]
            assert pyr[i].shape[1] <= pyr[i-1].shape[1]

    def test_finest_matches_input(self):
        """Level 0 should be a copy of the input."""
        img = np.random.rand(64, 64)
        f = fspecial_gaussian(5, 1.0)
        pyr = compute_image_pyramid(img, f, n_levels=3, ratio=0.5)
        np.testing.assert_array_equal(pyr[0], img)

    def test_color_pyramid(self):
        """Pyramid should work with color images."""
        img = np.random.rand(64, 64, 3)
        f = fspecial_gaussian(5, 1.0)
        pyr = compute_image_pyramid(img, f, n_levels=3, ratio=0.5)
        assert pyr[1].ndim == 3
        assert pyr[1].shape[2] == 3

    def test_single_level(self):
        """Single level pyramid should just return the input."""
        img = np.random.rand(32, 32)
        f = fspecial_gaussian(5, 1.0)
        pyr = compute_image_pyramid(img, f, n_levels=1, ratio=0.5)
        assert len(pyr) == 1
        np.testing.assert_array_equal(pyr[0], img)


class TestResampleFlow:
    """Test flow field resampling."""

    def test_same_size_is_copy(self):
        """Resampling to same size returns copy."""
        uv = np.random.randn(10, 15, 2)
        result = resample_flow(uv, (10, 15))
        np.testing.assert_array_equal(result, uv)
        assert result is not uv

    def test_upscale_doubles_magnitude(self):
        """Upscaling 2x should double flow magnitudes."""
        uv = np.ones((10, 10, 2))
        result = resample_flow(uv, (20, 20))
        assert result.shape == (20, 20, 2)
        np.testing.assert_allclose(result[:, :, 0], 2.0, atol=0.2)
        np.testing.assert_allclose(result[:, :, 1], 2.0, atol=0.2)

    def test_output_shape(self):
        """Output should match target size."""
        uv = np.random.randn(30, 40, 2)
        result = resample_flow(uv, (60, 80))
        assert result.shape == (60, 80, 2)
