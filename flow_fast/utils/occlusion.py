"""Occlusion detection for optical flow.

Accelerated version: replaces scipy.ndimage.map_coordinates with
cv2.remap for the image-warping step.

Public API
----------
detect_occlusion(uv, images, sigma_d, sigma_i)
"""

import numpy as np
import cv2


def detect_occlusion(uv, images, sigma_d=0.3, sigma_i=20.0):
    """Detect occlusion using flow divergence and brightness constancy.

    Parameters
    ----------
    uv : ndarray, shape (H, W, 2)
        Flow field.
    images : ndarray, shape (H, W, 2) or (H, W, C)
        Image pair concatenated along the last axis.
    sigma_d : float
        Sigma for divergence term.
    sigma_i : float
        Sigma for brightness constancy term.

    Returns
    -------
    occ : ndarray, shape (H, W), values in [0, 1]
        Occlusion confidence.  Higher = less occluded.
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

    # 0-based coordinate maps for cv2.remap (float32)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)

    if im1.ndim == 2:
        im2_32 = im2.astype(np.float32)
        warp2 = cv2.remap(im2_32, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
        It = np.abs(warp2 - im1)
    else:
        It = np.zeros(im1.shape[:2], dtype=np.float64)
        for c in range(im1.shape[2]):
            ch32 = im2[:, :, c].astype(np.float32)
            warp_c = cv2.remap(ch32, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
            It += np.abs(warp_c - im1[:, :, c])
        It /= im1.shape[2]

    occ_bc = np.exp(-It ** 2 / (2 * sigma_i ** 2))

    return occ_div * occ_bc
