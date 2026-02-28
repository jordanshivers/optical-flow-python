"""Image pyramid construction for coarse-to-fine optical flow.

Accelerated version: replaces scipy.ndimage.correlate with cv2.filter2D
and the custom _matlab_imresize_bilinear with cv2.resize(INTER_LINEAR).

Public API
----------
compute_image_pyramid(img, f, n_levels, ratio)
"""

import numpy as np
import cv2


def _matlab_round(x):
    """Round half away from zero, matching MATLAB's round() behaviour."""
    return int(np.floor(x + 0.5))


def _matlab_imresize_bilinear(img, ratio):
    """Bilinear resize matching MATLAB's imresize(A, ratio, 'bilinear',
    'Antialiasing', false).

    Uses cv2.resize with INTER_LINEAR and MATLAB-compatible coordinate
    mapping.  MATLAB's mapping is ``u = (out + 0.5) / scale - 0.5``,
    which is the same as OpenCV's default mapping when the destination
    size is computed with MATLAB's rounding convention.

    Parameters
    ----------
    img : ndarray, shape (H, W) or (H, W, C)
        Input image.
    ratio : float
        Resize ratio (< 1 for downsampling).

    Returns
    -------
    ndarray
        Resized image.
    """
    H, W = img.shape[:2]
    new_H = max(1, _matlab_round(H * ratio))
    new_W = max(1, _matlab_round(W * ratio))

    # cv2.resize takes (width, height) as dsize
    img32 = img.astype(np.float32)
    resized = cv2.resize(img32, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float64)


def compute_image_pyramid(img, f, n_levels, ratio):
    """Build a Gaussian image pyramid.

    Matches MATLAB: imfilter with 'symmetric' boundary, then imresize with
    'bilinear' and 'Antialiasing' false.

    Uses cv2.filter2D for smoothing (correlation, BORDER_REFLECT_101 matches
    MATLAB's 'symmetric' boundary) and cv2.resize for downsampling.

    Parameters
    ----------
    img : ndarray, shape (H, W) or (H, W, C)
        Input image.
    f : ndarray
        Smoothing filter kernel (2-D).
    n_levels : int
        Number of pyramid levels.
    ratio : float
        Downsampling ratio (< 1, e.g. 0.5).

    Returns
    -------
    pyramid : list of ndarray
        List of images, index 0 = finest (original), last = coarsest.
    """
    pyramid = [img.copy()]
    current = img.copy()

    f32 = f.astype(np.float32)

    for _ in range(1, n_levels):
        if current.ndim == 2:
            cur32 = current.astype(np.float32)
            smoothed = cv2.filter2D(cur32, -1, f32,
                                    borderType=cv2.BORDER_REFLECT_101)
            smoothed = smoothed.astype(np.float64)
        else:
            smoothed = np.zeros_like(current)
            for c in range(current.shape[2]):
                cur32 = current[:, :, c].astype(np.float32)
                sm = cv2.filter2D(cur32, -1, f32,
                                  borderType=cv2.BORDER_REFLECT_101)
                smoothed[:, :, c] = sm.astype(np.float64)

        current = _matlab_imresize_bilinear(smoothed, ratio)
        pyramid.append(current)

    return pyramid
