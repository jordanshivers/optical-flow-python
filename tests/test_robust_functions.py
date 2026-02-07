"""Tests for robust penalty functions."""
import numpy as np
import pytest
from optical_flow.robust.penalties import (
    quadratic, lorentzian, charbonnier, generalized_charbonnier,
    geman_mcclure, huber, tukey, gaussian, tdist, tdist_unnorm,
)
from optical_flow.robust.robust_function import RobustFunction


class TestPenaltyFunctions:
    """Test individual penalty function implementations."""

    def test_quadratic_value(self):
        x = np.array([0.0, 1.0, -2.0])
        y = quadratic(x, 1.0, 0)
        np.testing.assert_allclose(y, [0.0, 1.0, 4.0])

    def test_quadratic_deriv(self):
        x = np.array([0.0, 1.0, -2.0])
        y = quadratic(x, 1.0, 1)
        np.testing.assert_allclose(y, [0.0, 2.0, -4.0])

    def test_quadratic_weight(self):
        x = np.array([0.0, 1.0, -2.0])
        y = quadratic(x, 1.0, 2)
        np.testing.assert_allclose(y, [2.0, 2.0, 2.0])

    def test_quadratic_sigma(self):
        x = np.array([1.0])
        y = quadratic(x, 2.0, 0)
        np.testing.assert_allclose(y, [0.25])

    def test_lorentzian_at_zero(self):
        y = lorentzian(np.array([0.0]), 1.0, 0)
        np.testing.assert_allclose(y, [0.0])

    def test_lorentzian_symmetric(self):
        x = np.array([1.0])
        y_pos = lorentzian(x, 0.5, 0)
        y_neg = lorentzian(-x, 0.5, 0)
        np.testing.assert_allclose(y_pos, y_neg)

    def test_lorentzian_deriv_at_zero(self):
        y = lorentzian(np.array([0.0]), 1.0, 1)
        np.testing.assert_allclose(y, [0.0])

    def test_charbonnier_at_zero(self):
        y = charbonnier(np.array([0.0]), 0.01, 0)
        assert y[0] > 0  # Non-zero at origin

    def test_charbonnier_weight_positive(self):
        x = np.linspace(-5, 5, 100)
        w = charbonnier(x, 0.01, 2)
        assert np.all(w > 0)

    def test_generalized_charbonnier_reduces_to_quadratic(self):
        """With a=1, gen_charbonnier(sigma, 1) ~ sigma^2 + x^2."""
        x = np.array([0.0, 1.0, 2.0])
        sig = 0.1
        y = generalized_charbonnier(x, [sig, 1.0], 0)
        expected = sig**2 + x**2
        np.testing.assert_allclose(y, expected)

    def test_geman_mcclure_bounded(self):
        """Geman-McClure saturates at 1.0."""
        x = np.array([1e6])
        y = geman_mcclure(x, 1.0, 0)
        np.testing.assert_allclose(y, [1.0], atol=1e-6)

    def test_geman_mcclure_at_zero(self):
        y = geman_mcclure(np.array([0.0]), 1.0, 0)
        np.testing.assert_allclose(y, [0.0])

    def test_huber_quadratic_region(self):
        """Huber is quadratic for small x."""
        x = np.array([0.1])
        sigma = 1.0
        y = huber(x, sigma, 0)
        np.testing.assert_allclose(y, [0.01])

    def test_huber_linear_region(self):
        """Huber is linear for large x."""
        sigma = 0.5
        sig2 = sigma**2
        x = np.array([2.0])  # |x| > sig^2 = 0.25
        y = huber(x, sigma, 0)
        expected = 2.0 * sig2 * np.abs(x) - sig2**2
        np.testing.assert_allclose(y, expected)

    def test_tukey_at_zero(self):
        y = tukey(np.array([0.0]), 1.0, 0)
        np.testing.assert_allclose(y, [0.0])

    def test_tukey_saturates(self):
        """Tukey saturates at 1/3 for |x| > sigma."""
        y = tukey(np.array([100.0]), 1.0, 0)
        np.testing.assert_allclose(y, [1.0/3.0])

    def test_tukey_zero_weight_outside(self):
        """Tukey weight is zero outside sigma."""
        w = tukey(np.array([100.0]), 1.0, 2)
        np.testing.assert_allclose(w, [0.0])

    def test_gaussian_nll(self):
        """Gaussian penalty is negative log-likelihood."""
        x = np.array([0.0])
        sigma = 1.0
        y = gaussian(x, sigma, 0)
        expected = 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(y, [expected])

    def test_tdist_deriv_symmetry(self):
        x = np.array([1.0])
        d_pos = tdist(x, [5.0, 1.0], 1)
        d_neg = tdist(-x, [5.0, 1.0], 1)
        np.testing.assert_allclose(d_pos, -d_neg)

    def test_tdist_weight_positive(self):
        x = np.linspace(-5, 5, 100)
        w = tdist(x, [5.0, 1.0], 2)
        assert np.all(w > 0)

    def test_tdist_unnorm_no_constant(self):
        """tdist_unnorm is same as tdist except value has no constant."""
        x = np.array([0.0])
        y_unnorm = tdist_unnorm(x, [5.0, 1.0], 0)
        np.testing.assert_allclose(y_unnorm, [0.0])

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            quadratic(np.array([1.0]), 1.0, 99)


class TestRobustFunction:
    """Test the RobustFunction wrapper class."""

    def test_creation(self):
        rf = RobustFunction('charbonnier', 0.01)
        assert rf.method == 'charbonnier'
        np.testing.assert_allclose(rf.sigma, [0.01])

    def test_evaluate(self):
        rf = RobustFunction('quadratic', 1.0)
        x = np.array([2.0])
        np.testing.assert_allclose(rf.evaluate(x), [4.0])

    def test_deriv(self):
        rf = RobustFunction('quadratic', 1.0)
        x = np.array([2.0])
        np.testing.assert_allclose(rf.deriv(x), [4.0])

    def test_deriv_over_x(self):
        rf = RobustFunction('quadratic', 1.0)
        x = np.array([2.0])
        np.testing.assert_allclose(rf.deriv_over_x(x), [2.0])

    def test_generalized_charbonnier_two_params(self):
        rf = RobustFunction('generalized_charbonnier', 0.01, 0.45)
        np.testing.assert_allclose(rf.sigma, [0.01, 0.45])

    def test_tdist_two_params(self):
        rf = RobustFunction('tdist', 5.0, 1.0)
        np.testing.assert_allclose(rf.sigma, [5.0, 1.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty"):
            RobustFunction('invalid_method', 1.0)

    def test_repr(self):
        rf = RobustFunction('lorentzian', 0.5)
        assert 'lorentzian' in repr(rf)

    def test_param_property(self):
        rf = RobustFunction('charbonnier', 0.01)
        np.testing.assert_allclose(rf.param, [0.01])

    def test_numerical_derivative(self):
        """Verify analytical derivative matches numerical derivative."""
        rf = RobustFunction('lorentzian', 0.5)
        x = np.array([0.3])
        eps = 1e-7
        numerical = (rf.evaluate(x + eps) - rf.evaluate(x - eps)) / (2 * eps)
        analytical = rf.deriv(x)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)
