"""Occlusion detection for optical flow."""
import numpy as np
from scipy.ndimage import map_coordinates


def detect_occlusion(uv, images, sigma_d=0.3, sigma_i=20.0):
    """Detect occlusion using flow divergence and brightness constancy.

    Args:
        uv: Flow field (H, W, 2).
        images: Image pair (H, W, 2) or (H, W, C).
        sigma_d: Sigma for divergence term.
        sigma_i: Sigma for brightness constancy term.

    Returns:
        occ: Occlusion confidence (H, W) in [0,1]. Higher = less occluded.
    """
    H, W = uv.shape[:2]
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    # Flow divergence
    dudx = np.zeros_like(u)
    dudx[:, 1:] = u[:, 1:] - u[:, :-1]
    dvdy = np.zeros_like(v)
    dvdy[1:, :] = v[1:, :] - v[:-1, :]
    div = dudx + dvdy

    occ_div = np.exp(-div ** 2 / (2 * sigma_d ** 2))

    # Brightness constancy
    if images.shape[2] == 2:
        im1 = images[:, :, 0]
        im2 = images[:, :, 1]
    else:
        nc = images.shape[2] // 2
        im1 = images[:, :, :nc]
        im2 = images[:, :, nc:]

    y, x = np.mgrid[0:H, 0:W].astype(float)
    x2 = x + u
    y2 = y + v

    if im1.ndim == 2:
        warp2 = map_coordinates(im2, [y2, x2], order=1, mode='nearest')
        It = np.abs(warp2 - im1)
    else:
        It = np.zeros(im1.shape[:2])
        for c in range(im1.shape[2]):
            warp_c = map_coordinates(im2[:, :, c], [y2, x2], order=1, mode='nearest')
            It += np.abs(warp_c - im1[:, :, c])
        It /= im1.shape[2]

    occ_bc = np.exp(-It ** 2 / (2 * sigma_i ** 2))

    return occ_div * occ_bc
