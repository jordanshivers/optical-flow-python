"""Robust penalty functions for optical flow estimation.

Reimplementation of MATLAB code from:
    Sun, Roth, Black. "Secrets of Optical Flow Estimation and Their Principles."
    CVPR 2010.

Each function takes (x, sigma, d_type) where:
    x:      numpy array of values
    sigma:  parameter(s) - scalar or array depending on function
    d_type: 0 = function value rho(x)
            1 = first derivative rho'(x)
            2 = rho'(x)/x  (IRLS weight)
"""
import numpy as np
from scipy.special import gammaln


def quadratic(x, sigma, d_type):
    """Quadratic penalty: rho(x) = x^2 / sigma^2.

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is used.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x (or scalar broadcast).
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    if d_type == 0:
        y = x ** 2 / sig2
    elif d_type == 1:
        y = 2.0 * x / sig2
    elif d_type == 2:
        y = np.full_like(x, 2.0 / sig2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def lorentzian(x, sigma, d_type):
    """Lorentzian penalty: rho(x) = log(1 + x^2 / (2*sigma^2)).

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is used.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    if d_type == 0:
        y = np.log(1.0 + x ** 2 / (2.0 * sig2))
    elif d_type == 1:
        y = 2.0 * x / (2.0 * sig2 + x ** 2)
    elif d_type == 2:
        y = 2.0 / (2.0 * sig2 + x ** 2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def charbonnier(x, sigma, d_type):
    """Charbonnier penalty (generalized L1).

    Matches MATLAB implementation exactly:
        sig2 = sigma(1)^2
        rho  = 1 + (x / sig2)^2
        type 0: sig2 * sqrt(rho)
        type 1: x / (sig2 * sqrt(rho))
        type 2: 1 / (sig2 * sqrt(rho))

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is the sigma value.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2
    rho = 1.0 + (x / sig2) ** 2
    sqrt_rho = np.sqrt(rho)

    if d_type == 0:
        y = sig2 * sqrt_rho
    elif d_type == 1:
        y = x / (sig2 * sqrt_rho)
    elif d_type == 2:
        y = 1.0 / (sig2 * sqrt_rho)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def generalized_charbonnier(x, sigma, d_type):
    """Generalized Charbonnier penalty: rho(x) = (sig^2 + x^2)^a.

    Args:
        x: numpy array of input values.
        sigma: array-like [sig, a] where sig is the scale and a is the exponent.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    sig = p[0]
    a = p[1]
    sig2 = sig ** 2
    base = sig2 + x ** 2

    if d_type == 0:
        y = base ** a
    elif d_type == 1:
        y = 2.0 * a * x * base ** (a - 1.0)
    elif d_type == 2:
        y = 2.0 * a * base ** (a - 1.0)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def geman_mcclure(x, sigma, d_type):
    """Geman-McClure penalty: rho(x) = x^2 / (sigma^2 + x^2).

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is used.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2
    denom = sig2 + x ** 2

    if d_type == 0:
        y = x ** 2 / denom
    elif d_type == 1:
        y = 2.0 * sig2 * x / (denom ** 2)
    elif d_type == 2:
        y = 2.0 * sig2 / (denom ** 2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def huber(x, sigma, d_type):
    """Huber penalty (piecewise quadratic/linear).

    MATLAB convention: sig2 = sigma^2, threshold at |x| <= sig2.
        type 0: x^2           if |x| <= sig2,  else 2*sig2*|x| - sig2^2
        type 1: 2*x           if |x| <= sig2,  else 2*sig2*sign(x)
        type 2: 2             if |x| <= sig2,  else 2*sig2/|x|

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is used.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    absx = np.abs(x)
    mask = absx <= sig2  # inlier region

    if d_type == 0:
        y = np.where(mask,
                     x ** 2,
                     2.0 * sig2 * absx - sig2 ** 2)
    elif d_type == 1:
        y = np.where(mask,
                     2.0 * x,
                     2.0 * sig2 * np.sign(x))
    elif d_type == 2:
        y = np.where(mask,
                     np.full_like(x, 2.0),
                     2.0 * sig2 / np.maximum(absx, 1e-30))
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def tukey(x, sigma, d_type):
    """Tukey biweight penalty.

    MATLAB: sig = sigma(1), threshold at |x| <= sig.
        type 0: 1/3 * (1 - (1 - x^2/sig^2)^3) if |x|<=sig, else 1/3
        type 1: 2*x*(1 - x^2/sig^2)^2 / sig^2  if |x|<=sig, else 0
        type 2: 2*(1 - x^2/sig^2)^2 / sig^2     if |x|<=sig, else 0

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is used as the threshold.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    absx = np.abs(x)
    mask = absx <= sig
    ratio_sq = (x ** 2) / sig2  # x^2 / sig^2
    one_minus = 1.0 - ratio_sq

    if d_type == 0:
        y = np.where(mask,
                     (1.0 / 3.0) * (1.0 - one_minus ** 3),
                     1.0 / 3.0)
    elif d_type == 1:
        y = np.where(mask,
                     2.0 * x * (one_minus ** 2) / sig2,
                     0.0)
    elif d_type == 2:
        y = np.where(mask,
                     2.0 * (one_minus ** 2) / sig2,
                     0.0)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def gaussian(x, sigma, d_type):
    """Gaussian negative log-likelihood penalty.

    rho(x) = 0.5*log(2*pi) + log(sigma) + 0.5*(x/sigma)^2

    Args:
        x: numpy array of input values.
        sigma: scalar or array; sigma[0] is the standard deviation.
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sig = np.atleast_1d(sigma)[0]
    sig2 = sig ** 2

    if d_type == 0:
        y = 0.5 * np.log(2.0 * np.pi) + np.log(sig) + 0.5 * (x / sig) ** 2
    elif d_type == 1:
        y = x / sig2
    elif d_type == 2:
        y = np.full_like(x, 1.0 / sig2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def tdist(x, sigma, d_type):
    """Student-t distribution penalty.

    Parameters sigma = [r, s] where r is degrees of freedom and s is scale.
    MATLAB:
        cnst = gammaln(r/2) - gammaln((r+1)/2) + 0.5*log(r*pi) + log(s)
        type 0: cnst + r*log(1 + 0.5*(x/s)^2)
               Actually MATLAB: (r+1)/2 * log(1 + x^2/(s^2*r))  + cnst
               Let me match exactly: y = (r+1)/2 * log(1 + x.^2 / (s^2*r)) + cnst
        type 1: r*x / (s^2*r + x^2)   [from MATLAB: r*x./(s^2*r+x.^2)]
               Actually: (r+1)*x / (s^2*r + x^2)
        type 2: r / (s^2*r + x^2)     [from MATLAB: r./(s^2*r+x.^2)]
               Actually: (r+1) / (s^2*r + x^2)

    Following MATLAB source exactly:
        type 1: (r+1)*x ./ (s^2*r + x.^2)
        type 2: (r+1) ./ (s^2*r + x.^2)

    Args:
        x: numpy array of input values.
        sigma: array-like [r, s].
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    r = p[0]
    s = p[1]
    s2r = s ** 2 * r

    if d_type == 0:
        cnst = (gammaln(r / 2.0) - gammaln((r + 1.0) / 2.0)
                + 0.5 * np.log(r * np.pi) + np.log(s))
        y = (r + 1.0) / 2.0 * np.log(1.0 + x ** 2 / s2r) + cnst
    elif d_type == 1:
        y = (r + 1.0) * x / (s2r + x ** 2)
    elif d_type == 2:
        y = (r + 1.0) / (s2r + x ** 2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def tdist_unnorm(x, sigma, d_type):
    """Unnormalized Student-t distribution penalty.

    Like tdist but without the normalization constant.
    type 0: (r+1)/2 * log(1 + x^2/(s^2*r))
    type 1, type 2: same as tdist.

    Args:
        x: numpy array of input values.
        sigma: array-like [r, s].
        d_type: 0 for value, 1 for derivative, 2 for derivative/x.

    Returns:
        y: numpy array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    p = np.atleast_1d(sigma)
    r = p[0]
    s = p[1]
    s2r = s ** 2 * r

    if d_type == 0:
        y = (r + 1.0) / 2.0 * np.log(1.0 + x ** 2 / s2r)
    elif d_type == 1:
        y = (r + 1.0) * x / (s2r + x ** 2)
    elif d_type == 2:
        y = (r + 1.0) / (s2r + x ** 2)
    else:
        raise ValueError(f"Unknown d_type: {d_type}")
    return y


def mixture(x, sigma, d_type):
    """Mixture of robust penalty functions.

    Not implemented - requires complex parameter structure.

    Raises:
        NotImplementedError
    """
    raise NotImplementedError(
        "Mixture penalty is not yet implemented. "
        "It requires a complex parameter structure: "
        "sigma = [[weights], [component_funcs], [component_params]]."
    )


def spline_penalty(x, sigma, d_type):
    """Spline-based penalty function.

    Not implemented.

    Raises:
        NotImplementedError
    """
    raise NotImplementedError(
        "Spline penalty is not yet implemented."
    )
