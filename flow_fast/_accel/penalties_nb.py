"""Numba-compiled robust penalty functions for IRLS optical flow solvers.

Each penalty family (quadratic, lorentzian, charbonnier, generalized
charbonnier) provides three element-wise operations on 1-D ``float64`` arrays:

* ``evaluate(x, sig)``      -- the penalty value rho(x)
* ``deriv(x, sig)``         -- the first derivative rho'(x)
* ``deriv_over_x(x, sig)``  -- the IRLS weight rho'(x) / x

The ``deriv_over_x`` variants are the performance-critical hot path inside the
iteratively re-weighted least-squares (IRLS) inner loop and are the primary
reason for the Numba compilation.

All functions operate on contiguous 1-D ``float64`` arrays and return a new
array of the same length.

Mathematical definitions
------------------------
Quadratic:
    rho(x) = x^2 / sig^2
    rho'(x) = 2 x / sig^2
    rho'(x)/x = 2 / sig^2

Lorentzian:
    rho(x) = log(1 + x^2 / (2 sig^2))
    rho'(x) = 2 x / (2 sig^2 + x^2)
    rho'(x)/x = 2 / (2 sig^2 + x^2)

Charbonnier (matches MATLAB Sun/Roth/Black convention):
    sig2 = sig^2
    rho_base = 1 + (x / sig2)^2
    rho(x)   = sig2 * sqrt(rho_base)
    rho'(x)  = x / (sig2 * sqrt(rho_base))
    rho'(x)/x = 1 / (sig2 * sqrt(rho_base))

Generalized Charbonnier:
    rho(x)   = (sig^2 + x^2)^a
    rho'(x)  = 2 a x (sig^2 + x^2)^(a-1)
    rho'(x)/x = 2 a (sig^2 + x^2)^(a-1)
"""

import numba
import numpy as np


# ===================================================================
#  Quadratic
# ===================================================================
@numba.njit(cache=True)
def quadratic_evaluate(x, sig):
    """Quadratic penalty value: rho(x) = x^2 / sig^2.

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Penalty values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = x[i] * x[i] / sig2
    return y


@numba.njit(cache=True)
def quadratic_deriv(x, sig):
    """Quadratic penalty derivative: rho'(x) = 2 x / sig^2.

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Derivative values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = 2.0 * x[i] / sig2
    return y


@numba.njit(cache=True)
def quadratic_deriv_over_x(x, sig):
    """Quadratic IRLS weight: rho'(x)/x = 2 / sig^2 (constant).

    Parameters
    ----------
    x : float64[:]
        Input array (unused but kept for API consistency).
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Constant weight array.
    """
    n = x.shape[0]
    val = 2.0 / (sig * sig)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = val
    return y


# ===================================================================
#  Lorentzian
# ===================================================================
@numba.njit(cache=True)
def lorentzian_evaluate(x, sig):
    """Lorentzian penalty value: rho(x) = log(1 + x^2 / (2 sig^2)).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Penalty values.
    """
    n = x.shape[0]
    two_sig2 = 2.0 * sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = np.log(1.0 + x[i] * x[i] / two_sig2)
    return y


@numba.njit(cache=True)
def lorentzian_deriv(x, sig):
    """Lorentzian penalty derivative: rho'(x) = 2 x / (2 sig^2 + x^2).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Derivative values.
    """
    n = x.shape[0]
    two_sig2 = 2.0 * sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = 2.0 * x[i] / (two_sig2 + x[i] * x[i])
    return y


@numba.njit(cache=True)
def lorentzian_deriv_over_x(x, sig):
    """Lorentzian IRLS weight: rho'(x)/x = 2 / (2 sig^2 + x^2).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        IRLS weight values.
    """
    n = x.shape[0]
    two_sig2 = 2.0 * sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = 2.0 / (two_sig2 + x[i] * x[i])
    return y


# ===================================================================
#  Charbonnier  (Sun / Roth / Black MATLAB convention)
# ===================================================================
@numba.njit(cache=True)
def charbonnier_evaluate(x, sig):
    """Charbonnier penalty value: rho(x) = sig^2 * sqrt(1 + (x/sig^2)^2).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Penalty values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        ratio = x[i] / sig2
        y[i] = sig2 * np.sqrt(1.0 + ratio * ratio)
    return y


@numba.njit(cache=True)
def charbonnier_deriv(x, sig):
    """Charbonnier penalty derivative: rho'(x) = x / (sig^2 * sqrt(1 + (x/sig^2)^2)).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        Derivative values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        ratio = x[i] / sig2
        y[i] = x[i] / (sig2 * np.sqrt(1.0 + ratio * ratio))
    return y


@numba.njit(cache=True)
def charbonnier_deriv_over_x(x, sig):
    """Charbonnier IRLS weight: rho'(x)/x = 1 / (sig^2 * sqrt(1 + (x/sig^2)^2)).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.

    Returns
    -------
    y : float64[:]
        IRLS weight values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        ratio = x[i] / sig2
        y[i] = 1.0 / (sig2 * np.sqrt(1.0 + ratio * ratio))
    return y


# ===================================================================
#  Generalized Charbonnier
# ===================================================================
@numba.njit(cache=True)
def generalized_charbonnier_evaluate(x, sig, a):
    """Generalized Charbonnier penalty value: rho(x) = (sig^2 + x^2)^a.

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.
    a : float64
        Exponent (typically 0 < a < 1 for a sub-quadratic penalty).

    Returns
    -------
    y : float64[:]
        Penalty values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = (sig2 + x[i] * x[i]) ** a
    return y


@numba.njit(cache=True)
def generalized_charbonnier_deriv(x, sig, a):
    """Generalized Charbonnier derivative: rho'(x) = 2 a x (sig^2 + x^2)^(a-1).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.
    a : float64
        Exponent.

    Returns
    -------
    y : float64[:]
        Derivative values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    am1 = a - 1.0
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = 2.0 * a * x[i] * (sig2 + x[i] * x[i]) ** am1
    return y


@numba.njit(cache=True)
def generalized_charbonnier_deriv_over_x(x, sig, a):
    """Generalized Charbonnier IRLS weight: rho'(x)/x = 2 a (sig^2 + x^2)^(a-1).

    Parameters
    ----------
    x : float64[:]
        Input array.
    sig : float64
        Scale parameter.
    a : float64
        Exponent.

    Returns
    -------
    y : float64[:]
        IRLS weight values.
    """
    n = x.shape[0]
    sig2 = sig * sig
    am1 = a - 1.0
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = 2.0 * a * (sig2 + x[i] * x[i]) ** am1
    return y
