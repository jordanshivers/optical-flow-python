"""Image processing utilities for optical flow estimation."""
import numpy as np
from scipy.ndimage import correlate


def scale_image(im, vlow, vhigh, ilow=None, ihigh=None):
    """Linearly rescale image values.

    Args:
        im: Input image.
        vlow, vhigh: Target value range.
        ilow, ihigh: Source value range. If None, uses im min/max.

    Returns:
        Rescaled image.
    """
    im = np.asarray(im, dtype=float)
    if ilow is None:
        ilow = im.min()
    if ihigh is None:
        ihigh = im.max()

    if ihigh == ilow:
        return np.full_like(im, (vlow + vhigh) / 2.0)

    return (im - ilow) / (ihigh - ilow) * (vhigh - vlow) + vlow


def fspecial_gaussian(size, sigma):
    """Create a Gaussian filter kernel (like MATLAB's fspecial('gaussian')).

    Args:
        size: Kernel size (int for square, or tuple).
        sigma: Standard deviation.

    Returns:
        h: Normalized Gaussian kernel.
    """
    if isinstance(size, (int, np.integer)):
        size = (int(size), int(size))

    m, n = [(s - 1) / 2.0 for s in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h /= s
    return h


def structure_texture_decomposition_rof(im, theta=1.0/8, n_iters=100, alp=0.95):
    """Structure-texture decomposition using ROF model.

    Rudin-Osher-Fatemi denoising with primal-dual algorithm.
    Matches MATLAB implementation: normalizes input to [-1,1], runs ROF
    per channel, then rescales texture to [0,255].

    Args:
        im: Input image (H, W) or (H, W, C).
        theta: Regularization parameter.
        n_iters: Number of iterations.
        alp: Weight for texture extraction.

    Returns:
        texture: Texture component, same shape as im, scaled to [0,255].
    """
    im = np.asarray(im, dtype=float)

    # Normalize to [-1, 1] (global across all channels, matching MATLAB)
    im_norm = scale_image(im, -1, 1)

    # Compute structure per channel
    if im_norm.ndim == 3:
        structure = np.zeros_like(im_norm)
        for c in range(im_norm.shape[2]):
            structure[:, :, c] = _rof_structure_2d(im_norm[:, :, c], theta, n_iters)
    else:
        structure = _rof_structure_2d(im_norm, theta, n_iters)

    # Texture = normalized_input - alp * structure, rescaled to [0, 255]
    texture = scale_image(im_norm - alp * structure, 0, 255)
    return texture


def _rof_structure_2d(im, theta, n_iters):
    """ROF denoising for a single 2D image. Returns the structure component.

    Pre-allocates scratch arrays to avoid per-iteration allocation.
    """
    H, W = im.shape

    # Initialize dual variables
    p = np.zeros((H, W, 2))

    # Gradient step size
    delta = 1.0 / (4.0 * theta)

    # Pre-allocate scratch arrays
    div_p = np.empty((H, W))
    grad_u = np.empty((H, W, 2))
    norm_p = np.empty((H, W))

    for _ in range(n_iters):
        # Compute divergence of p: backward difference with zero boundary
        div_p[:] = 0.0
        div_p[:, 1:] += p[:, 1:, 0] - p[:, :-1, 0]
        div_p[:, 0] += p[:, 0, 0]
        div_p[1:, :] += p[1:, :, 1] - p[:-1, :, 1]
        div_p[0, :] += p[0, :, 1]

        # Update image estimate: u = im + theta * div_p
        u = im + theta * div_p

        # Compute gradient of u: forward difference with zero at boundary
        grad_u[:] = 0.0
        grad_u[:, :-1, 0] = u[:, 1:] - u[:, :-1]  # I_x
        grad_u[:-1, :, 1] = u[1:, :] - u[:-1, :]  # I_y

        # Update dual variable: p += delta * grad_u
        p += delta * grad_u

        # Reproject: |p| <= 1
        np.sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2, out=norm_p)
        np.maximum(norm_p, 1.0, out=norm_p)
        p[:, :, 0] /= norm_p
        p[:, :, 1] /= norm_p

    # Final divergence and structure
    div_p[:] = 0.0
    div_p[:, 1:] += p[:, 1:, 0] - p[:, :-1, 0]
    div_p[:, 0] += p[:, 0, 0]
    div_p[1:, :] += p[1:, :, 1] - p[:-1, :, 1]
    div_p[0, :] += p[0, :, 1]

    return im + theta * div_p
