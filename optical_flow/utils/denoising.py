"""Li-Osher weighted median denoising for optical flow."""
import numpy as np
from scipy.ndimage import median_filter


def denoise_LO(un, mfsz, lambda_param, n_iters=1):
    """Denoise using Li-Osher iterative median formula.

    Args:
        un: Input 2D array (H, W).
        mfsz: Median filter size [h, w] or int.
        lambda_param: Weight parameter.
        n_iters: Number of iterations.

    Returns:
        u: Denoised result (H, W).
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
