"""Robust penalty functions for optical flow estimation.

Provides both pure-NumPy and Numba-accelerated versions.
The Numba versions are used automatically when available.
"""
import numpy as np
from scipy.special import gammaln


# Try to import Numba-accelerated versions
try:
    from flow_fast._accel import penalties_nb as _nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def quadratic(x, sigma, d_type):
    """Quadratic penalty: rho(x) = x^2 / sigma^2."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]

    if _HAS_NUMBA and x.ndim == 1:
        if d_type == 0:
            return _nb.quadratic_evaluate(x, sig)
        elif d_type == 1:
            return _nb.quadratic_deriv(x, sig)
        elif d_type == 2:
            return _nb.quadratic_deriv_over_x(x, sig)

    sig2 = sig ** 2
    if d_type == 0:
        return x ** 2 / sig2
    elif d_type == 1:
        return 2.0 * x / sig2
    elif d_type == 2:
        return np.full_like(x, 2.0 / sig2)
    raise ValueError(f"Unknown d_type: {d_type}")


def lorentzian(x, sigma, d_type):
    """Lorentzian penalty: rho(x) = log(1 + x^2 / (2*sigma^2))."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]

    if _HAS_NUMBA and x.ndim == 1:
        if d_type == 0:
            return _nb.lorentzian_evaluate(x, sig)
        elif d_type == 1:
            return _nb.lorentzian_deriv(x, sig)
        elif d_type == 2:
            return _nb.lorentzian_deriv_over_x(x, sig)

    sig2 = sig ** 2
    if d_type == 0:
        return np.log(1.0 + x ** 2 / (2.0 * sig2))
    elif d_type == 1:
        return 2.0 * x / (2.0 * sig2 + x ** 2)
    elif d_type == 2:
        return 2.0 / (2.0 * sig2 + x ** 2)
    raise ValueError(f"Unknown d_type: {d_type}")


def charbonnier(x, sigma, d_type):
    """Charbonnier penalty (generalized L1)."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]

    if _HAS_NUMBA and x.ndim == 1:
        if d_type == 0:
            return _nb.charbonnier_evaluate(x, sig)
        elif d_type == 1:
            return _nb.charbonnier_deriv(x, sig)
        elif d_type == 2:
            return _nb.charbonnier_deriv_over_x(x, sig)

    sig2 = sig ** 2
    rho = 1.0 + (x / sig2) ** 2
    sqrt_rho = np.sqrt(rho)

    if d_type == 0:
        return sig2 * sqrt_rho
    elif d_type == 1:
        return x / (sig2 * sqrt_rho)
    elif d_type == 2:
        return 1.0 / (sig2 * sqrt_rho)
    raise ValueError(f"Unknown d_type: {d_type}")


def generalized_charbonnier(x, sigma, d_type):
    """Generalized Charbonnier penalty: rho(x) = (sig^2 + x^2)^a."""
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    sig = p[0]
    a = p[1]

    if _HAS_NUMBA and x.ndim == 1:
        if d_type == 0:
            return _nb.generalized_charbonnier_evaluate(x, sig, a)
        elif d_type == 1:
            return _nb.generalized_charbonnier_deriv(x, sig, a)
        elif d_type == 2:
            return _nb.generalized_charbonnier_deriv_over_x(x, sig, a)

    sig2 = sig ** 2
    base = sig2 + x ** 2

    if d_type == 0:
        return base ** a
    elif d_type == 1:
        return 2.0 * a * x * base ** (a - 1.0)
    elif d_type == 2:
        return 2.0 * a * base ** (a - 1.0)
    raise ValueError(f"Unknown d_type: {d_type}")


def geman_mcclure(x, sigma, d_type):
    """Geman-McClure penalty: rho(x) = x^2 / (sigma^2 + x^2)."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2
    denom = sig2 + x ** 2

    if d_type == 0:
        return x ** 2 / denom
    elif d_type == 1:
        return 2.0 * sig2 * x / (denom ** 2)
    elif d_type == 2:
        return 2.0 * sig2 / (denom ** 2)
    raise ValueError(f"Unknown d_type: {d_type}")


def huber(x, sigma, d_type):
    """Huber penalty (piecewise quadratic/linear)."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2
    absx = np.abs(x)
    mask = absx <= sig2

    if d_type == 0:
        return np.where(mask, x ** 2, 2.0 * sig2 * absx - sig2 ** 2)
    elif d_type == 1:
        return np.where(mask, 2.0 * x, 2.0 * sig2 * np.sign(x))
    elif d_type == 2:
        return np.where(mask, np.full_like(x, 2.0),
                        2.0 * sig2 / np.maximum(absx, 1e-30))
    raise ValueError(f"Unknown d_type: {d_type}")


def tukey(x, sigma, d_type):
    """Tukey biweight penalty."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2
    absx = np.abs(x)
    mask = absx <= sig
    ratio_sq = (x ** 2) / sig2
    one_minus = 1.0 - ratio_sq

    if d_type == 0:
        return np.where(mask, (1.0 / 3.0) * (1.0 - one_minus ** 3), 1.0 / 3.0)
    elif d_type == 1:
        return np.where(mask, 2.0 * x * (one_minus ** 2) / sig2, 0.0)
    elif d_type == 2:
        return np.where(mask, 2.0 * (one_minus ** 2) / sig2, 0.0)
    raise ValueError(f"Unknown d_type: {d_type}")


def gaussian(x, sigma, d_type):
    """Gaussian negative log-likelihood penalty."""
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    if d_type == 0:
        return 0.5 * np.log(2.0 * np.pi) + np.log(sig) + 0.5 * (x / sig) ** 2
    elif d_type == 1:
        return x / sig2
    elif d_type == 2:
        return np.full_like(x, 1.0 / sig2)
    raise ValueError(f"Unknown d_type: {d_type}")


def tdist(x, sigma, d_type):
    """Student-t distribution penalty."""
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    r = p[0]
    s = p[1]
    s2r = s ** 2 * r

    if d_type == 0:
        cnst = (gammaln(r / 2.0) - gammaln((r + 1.0) / 2.0)
                + 0.5 * np.log(r * np.pi) + np.log(s))
        return (r + 1.0) / 2.0 * np.log(1.0 + x ** 2 / s2r) + cnst
    elif d_type == 1:
        return (r + 1.0) * x / (s2r + x ** 2)
    elif d_type == 2:
        return (r + 1.0) / (s2r + x ** 2)
    raise ValueError(f"Unknown d_type: {d_type}")


def tdist_unnorm(x, sigma, d_type):
    """Unnormalized Student-t distribution penalty."""
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    r = p[0]
    s = p[1]
    s2r = s ** 2 * r

    if d_type == 0:
        return (r + 1.0) / 2.0 * np.log(1.0 + x ** 2 / s2r)
    elif d_type == 1:
        return (r + 1.0) * x / (s2r + x ** 2)
    elif d_type == 2:
        return (r + 1.0) / (s2r + x ** 2)
    raise ValueError(f"Unknown d_type: {d_type}")


def mixture(x, sigma, d_type):
    """Mixture of robust penalty functions. Not implemented."""
    raise NotImplementedError("Mixture penalty is not yet implemented.")


def spline_penalty(x, sigma, d_type):
    """Spline-based penalty function. Not implemented."""
    raise NotImplementedError("Spline penalty is not yet implemented.")
