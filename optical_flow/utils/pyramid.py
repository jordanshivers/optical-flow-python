"""Image pyramid construction for coarse-to-fine optical flow."""
import numpy as np
from scipy.ndimage import correlate, map_coordinates


def _matlab_round(x):
    """Round half away from zero, matching MATLAB's round() behavior."""
    return int(np.floor(x + 0.5))


def _matlab_imresize_bilinear(img, ratio):
    """Bilinear resize matching MATLAB's imresize(A, ratio, 'bilinear', 'Antialiasing', false).

    Uses MATLAB's coordinate convention: u = (out + 0.5) / scale - 0.5
    and MATLAB's rounding for output size: floor(x + 0.5).
    """
    H, W = img.shape[:2]
    new_H = max(1, _matlab_round(H * ratio))
    new_W = max(1, _matlab_round(W * ratio))

    scale_H = new_H / H
    scale_W = new_W / W

    # MATLAB coordinate mapping
    out_rows = (np.arange(new_H) + 0.5) / scale_H - 0.5
    out_cols = (np.arange(new_W) + 0.5) / scale_W - 0.5

    # Clip to valid range (matches MATLAB boundary)
    out_rows = np.clip(out_rows, 0, H - 1)
    out_cols = np.clip(out_cols, 0, W - 1)

    coords_r, coords_c = np.meshgrid(out_rows, out_cols, indexing='ij')

    if img.ndim == 2:
        return map_coordinates(img, [coords_r, coords_c], order=1, mode='nearest')
    else:
        result = np.empty((new_H, new_W, img.shape[2]))
        for c in range(img.shape[2]):
            result[:, :, c] = map_coordinates(img[:, :, c], [coords_r, coords_c],
                                               order=1, mode='nearest')
        return result


def compute_image_pyramid(img, f, n_levels, ratio):
    """Build a Gaussian image pyramid.

    Matches MATLAB: imfilter with 'symmetric' boundary, then imresize with
    'bilinear' and 'Antialiasing' false.

    Args:
        img: Input image (H, W) or (H, W, C).
        f: Smoothing filter kernel (2D).
        n_levels: Number of pyramid levels.
        ratio: Downsampling ratio (< 1, e.g. 0.5).

    Returns:
        pyramid: List of images, index 0 = finest (original), last = coarsest.
    """
    pyramid = [img.copy()]
    current = img.copy()

    for _ in range(1, n_levels):
        if current.ndim == 2:
            smoothed = correlate(current, f, mode='reflect')
        else:
            smoothed = np.zeros_like(current)
            for c in range(current.shape[2]):
                smoothed[:, :, c] = correlate(current[:, :, c], f, mode='reflect')

        current = _matlab_imresize_bilinear(smoothed, ratio)
        pyramid.append(current)

    return pyramid
