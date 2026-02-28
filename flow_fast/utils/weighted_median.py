"""Weighted median filtering for optical flow denoising.

Accelerated version: delegates the pixel-level weighted median computation
to the Numba-compiled ``flow_fast._accel.weighted_median_nb.weighted_median_filter_numba``.
The public API wrapper handles colour-image validation, reflect padding,
and the fallback to a plain median filter when no colour guide is available.

Public API
----------
denoise_color_weighted_medfilt2(uv, color_images, occ, area_hsz, mfsz,
                                sigma_i, full_version)
"""

import numpy as np


def denoise_color_weighted_medfilt2(uv, color_images, occ, area_hsz, mfsz,
                                    sigma_i, full_version=False):
    """Colour-guided weighted median filtering for flow denoising.

    Parameters
    ----------
    uv : ndarray, shape (H, W, 2)
        Flow field.
    color_images : ndarray or None
        Colour reference (H, W, 3) for computing bilateral weights.
    occ : ndarray, shape (H, W)
        Occlusion / confidence weights.
    area_hsz : int
        Half-window size for the colour-guided filter.
    mfsz : array-like or int
        Median filter size ``[h, w]``.
    sigma_i : float
        Colour similarity bandwidth.
    full_version : bool
        Unused (kept for API compatibility with the original).

    Returns
    -------
    uv_out : ndarray, shape (H, W, 2)
        Filtered flow.
    """
    H, W = uv.shape[:2]
    uv_out = uv.copy()

    # ----- fallback: plain median when no colour guide is available ---------
    if color_images is None or color_images.size < H * W:
        from scipy.ndimage import median_filter
        sz = int(mfsz[0]) if hasattr(mfsz, '__len__') else int(mfsz)
        uv_out[:, :, 0] = median_filter(uv[:, :, 0], size=sz, mode='reflect')
        uv_out[:, :, 1] = median_filter(uv[:, :, 1], size=sz, mode='reflect')
        return uv_out

    # ----- resize colour guide if spatial dims don't match ------------------
    if color_images.shape[0] != H or color_images.shape[1] != W:
        import cv2
        # cv2.resize takes (width, height) as dsize
        if color_images.ndim == 3:
            color_images = cv2.resize(
                color_images.astype(np.float32), (W, H),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float64)
        else:
            color_images = cv2.resize(
                color_images.astype(np.float32), (W, H),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float64)

    if color_images.ndim == 2:
        color_images = color_images[:, :, np.newaxis]

    # ----- ensure 3 channels (pad with zeros if needed) ---------------------
    if color_images.shape[2] < 3:
        tmp = np.zeros((H, W, 3), dtype=np.float64)
        tmp[:, :, :color_images.shape[2]] = color_images
        color_images = tmp

    # ----- pad arrays with reflect boundary ---------------------------------
    hsz = int(area_hsz)
    pad_2d = ((hsz, hsz), (hsz, hsz))
    pad_3d = ((hsz, hsz), (hsz, hsz), (0, 0))

    u_pad = np.pad(uv[:, :, 0], pad_2d, mode='reflect').astype(np.float64)
    v_pad = np.pad(uv[:, :, 1], pad_2d, mode='reflect').astype(np.float64)
    occ_pad = np.pad(occ, pad_2d, mode='reflect').astype(np.float64)
    color_pad = np.pad(color_images, pad_3d, mode='reflect').astype(np.float64)

    # ----- output buffers ---------------------------------------------------
    out_u = np.empty((H, W), dtype=np.float64)
    out_v = np.empty((H, W), dtype=np.float64)

    # ----- Numba hot path (with pure-Python fallback) -----------------------
    try:
        from flow_fast._accel.weighted_median_nb import weighted_median_filter_numba
        weighted_median_filter_numba(
            u_pad, v_pad, color_pad, occ_pad,
            H, W, hsz, float(sigma_i),
            out_u, out_v,
        )
    except ImportError:
        _wmedfilt_python(u_pad, v_pad, color_pad, occ_pad,
                         H, W, hsz, sigma_i, out_u, out_v)

    uv_out[:, :, 0] = out_u
    uv_out[:, :, 1] = out_v
    return uv_out


# ---------------------------------------------------------------------------
# Pure-Python fallback (identical logic to the Numba kernel)
# ---------------------------------------------------------------------------

def _weighted_median_1d(w, u):
    """Compute weighted median of 1-D data."""
    idx = np.argsort(u)
    u_sorted = u[idx]
    w_sorted = w[idx]
    cumw = np.cumsum(w_sorted)
    total = cumw[-1]
    median_idx = np.searchsorted(cumw, total / 2.0)
    return u_sorted[min(median_idx, len(u_sorted) - 1)]


def _wmedfilt_python(u_pad, v_pad, color_pad, occ_pad,
                     H, W, hsz, sigma_i, out_u, out_v):
    """Vectorised weighted median filtering (pure-Python fallback).

    Operates on pre-padded arrays so that no boundary checks are needed.
    """
    inv_2sigma2 = 1.0 / (2.0 * sigma_i ** 2)

    for i in range(H):
        r0 = i
        r1 = i + 2 * hsz + 1
        for j in range(W):
            c0 = j
            c1 = j + 2 * hsz + 1

            u_patch = u_pad[r0:r1, c0:c1].ravel()
            v_patch = v_pad[r0:r1, c0:c1].ravel()
            occ_patch = occ_pad[r0:r1, c0:c1].ravel()

            center_color = color_pad[i + hsz, j + hsz, :]
            cpatch = color_pad[r0:r1, c0:c1, :].reshape(-1, color_pad.shape[2])
            cdiff = np.sum((cpatch - center_color) ** 2, axis=1)
            w_color = np.exp(-cdiff * inv_2sigma2)

            weights = w_color * occ_patch
            weights = np.maximum(weights, 1e-10)

            out_u[i, j] = _weighted_median_1d(weights, u_patch)
            out_v[i, j] = _weighted_median_1d(weights, v_patch)
