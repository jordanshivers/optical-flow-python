"""Weighted median filtering for optical flow denoising."""
import numpy as np


def weighted_median_1d(w, u):
    """Compute weighted median of 1D data.

    Args:
        w: Weights (N,).
        u: Values (N,).

    Returns:
        median: Weighted median value.
    """
    idx = np.argsort(u)
    u_sorted = u[idx]
    w_sorted = w[idx]
    cumw = np.cumsum(w_sorted)
    total = cumw[-1]
    median_idx = np.searchsorted(cumw, total / 2.0)
    return u_sorted[min(median_idx, len(u_sorted) - 1)]


def denoise_color_weighted_medfilt2(uv, color_images, occ, area_hsz, mfsz, sigma_i, full_version=False):
    """Color-guided weighted median filtering for flow denoising.

    Args:
        uv: Flow field (H, W, 2).
        color_images: Color reference (H, W, 3) for weights.
        occ: Occlusion weights (H, W).
        area_hsz: Half window size.
        mfsz: Filter size [h, w].
        sigma_i: Color similarity sigma.
        full_version: Use full version with edge weighting.

    Returns:
        uv_out: Filtered flow (H, W, 2).
    """
    H, W = uv.shape[:2]
    uv_out = uv.copy()

    if color_images is None or color_images.size < H * W:
        from scipy.ndimage import median_filter
        sz = int(mfsz[0]) if hasattr(mfsz, '__len__') else int(mfsz)
        uv_out[:, :, 0] = median_filter(uv[:, :, 0], size=sz, mode='reflect')
        uv_out[:, :, 1] = median_filter(uv[:, :, 1], size=sz, mode='reflect')
        return uv_out

    if color_images.shape[0] != H or color_images.shape[1] != W:
        from skimage.transform import resize as sk_resize
        if color_images.ndim == 3:
            color_images = sk_resize(color_images, (H, W, color_images.shape[2]),
                                    preserve_range=True, anti_aliasing=False)
        else:
            color_images = sk_resize(color_images, (H, W),
                                    preserve_range=True, anti_aliasing=False)

    if color_images.ndim == 2:
        color_images = color_images[:, :, np.newaxis]

    # Use vectorized weighted median filtering
    _wmedfilt_vectorized(uv, color_images, occ, area_hsz, sigma_i, uv_out)

    return uv_out


def _wmedfilt_vectorized(uv, color_images, occ, area_hsz, sigma_i, uv_out):
    """Vectorized weighted median filtering with pre-padded arrays.

    Uses pre-padded arrays to eliminate per-pixel boundary checks,
    and pre-computes color arrays for efficient weight computation.
    """
    H, W = uv.shape[:2]
    C = color_images.shape[2]
    hsz = area_hsz

    # Pad arrays with reflect boundary
    pad_width_2d = ((hsz, hsz), (hsz, hsz))
    pad_width_3d_2 = ((hsz, hsz), (hsz, hsz), (0, 0))

    u_pad = np.pad(uv[:, :, 0], pad_width_2d, mode='reflect')
    v_pad = np.pad(uv[:, :, 1], pad_width_2d, mode='reflect')
    occ_pad = np.pad(occ, pad_width_2d, mode='reflect')
    color_pad = np.pad(color_images, pad_width_3d_2, mode='reflect')

    patch_size = (2 * hsz + 1) ** 2
    inv_2sigma2 = 1.0 / (2.0 * sigma_i ** 2)

    for i in range(H):
        # Extract row strips for this row's patches
        r0 = i  # in padded coords: i + hsz - hsz = i
        r1 = i + 2 * hsz + 1

        for j in range(W):
            c0 = j
            c1 = j + 2 * hsz + 1

            u_patch = u_pad[r0:r1, c0:c1].ravel()
            v_patch = v_pad[r0:r1, c0:c1].ravel()
            occ_patch = occ_pad[r0:r1, c0:c1].ravel()

            # Color weight
            center_color = color_pad[i + hsz, j + hsz, :]
            cpatch = color_pad[r0:r1, c0:c1, :].reshape(-1, C)
            cdiff = np.sum((cpatch - center_color) ** 2, axis=1)
            w_color = np.exp(-cdiff * inv_2sigma2)

            weights = w_color * occ_patch
            weights = np.maximum(weights, 1e-10)

            uv_out[i, j, 0] = weighted_median_1d(weights, u_patch)
            uv_out[i, j, 1] = weighted_median_1d(weights, v_patch)
