"""Tests for Classic+NL and high-level interface."""
import numpy as np
import pytest
from optical_flow.methods.config import load_of_method


class TestLoadOfMethod:
    """Test the method factory function."""

    def test_load_hs(self):
        ope = load_of_method('hs')
        assert ope is not None
        assert hasattr(ope, 'compute_flow')

    def test_load_ba(self):
        ope = load_of_method('ba')
        assert ope is not None

    def test_load_classic_nl(self):
        ope = load_of_method('classic+nl')
        assert ope is not None

    def test_load_classic_nl_fast(self):
        ope = load_of_method('classic+nl-fast')
        assert ope is not None
        assert ope.max_iters == 3
        assert ope.gnc_iters == 2

    def test_load_classic_c(self):
        ope = load_of_method('classic-c')
        assert ope is not None
        assert ope.texture is True

    def test_load_classic_pp(self):
        ope = load_of_method('classic++')
        assert ope is not None
        assert ope.interpolation_method == 'bi-cubic'

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            load_of_method('nonexistent_method')


class TestEstimateFlow:
    """Test high-level estimate_flow interface."""

    def test_hs_brightness_on_synthetic(self, synthetic_pair):
        """HS-brightness should produce flow on synthetic pair."""
        from optical_flow import estimate_flow
        im1, im2 = synthetic_pair
        uv = estimate_flow(im1, im2, method='hs-brightness',
                          params={'max_iters': 3, 'pyramid_levels': 2})
        assert uv.shape == (im1.shape[0], im1.shape[1], 2)

    def test_color_input(self, rubberwhale_images):
        """estimate_flow should handle color images."""
        from optical_flow import estimate_flow
        im1, im2 = rubberwhale_images
        # Use small region for speed
        im1_small = im1[:32, :32]
        im2_small = im2[:32, :32]
        uv = estimate_flow(im1_small, im2_small, method='hs-brightness',
                          params={'max_iters': 2, 'pyramid_levels': 1})
        assert uv.shape == (32, 32, 2)
