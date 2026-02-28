"""Li-Osher weighted median denoising for optical flow.

Unchanged from the original optical_flow.utils.denoising --
scipy.ndimage.median_filter is already well-optimised.

Public API
----------
denoise_LO(un, mfsz, lambda_param, n_iters)
"""

import numpy as np
from scipy.ndimage import median_filter


def denoise_LO(un, mfsz, lambda_param, n_iters=1):
    """Denoise using Li-Osher iterative median formula.

    Parameters
    ----------
    un : ndarray, shape (H, W)
        Input 2-D array.
    mfsz : array-like, int, or None
        Median filter size ``[h, w]`` or scalar.
    lambda_param : float
        Weight parameter.
    n_iters : int
        Number of iterations.

    Returns
    -------
    u : ndarray, shape (H, W)
        Denoised result.
    """
    if mfsz is None:
        return un.copy()

    if isinstance(mfsz, (list, tuple, np.ndarray)):
        fsz = (int(mfsz[0]), int(mfsz[1]))
    else:
        fsz = (int(mfsz), int(mfsz))

    u = un.copy()
    for _ in range(n_iters):
        u_tilde = u + lambda_param * (un - u)
        u = median_filter(u_tilde, size=fsz, mode='reflect')
    return u
