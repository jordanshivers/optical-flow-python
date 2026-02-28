"""Spatiotemporal derivatives for optical flow estimation.

Accelerated version: replaces scipy.ndimage.correlate and
scipy.ndimage.map_coordinates with OpenCV equivalents (cv2.filter2D and
cv2.remap) for significantly faster execution.  The Hermite bi-cubic
interpolation path delegates the inner polynomial evaluation to a
Numba-compiled kernel from flow_fast._accel.bicubic_interp_nb.

Public API
----------
partial_deriv(images, uv, interp_method, deriv_filter, blend)
interp2_bicubic(Z, XI, YI, deriv_filter)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Hermite bicubic coefficient matrix (Numerical Recipes, Table 3.6.1)
# ---------------------------------------------------------------------------
_W_BICUBIC = np.array([
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
    [-3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1,  0,  0,  0,  0],
    [ 2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0, -3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1],
    [ 0,  0,  0,  0,  2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1],
    [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
    [ 9, -9,  9, -9,  6,  3, -3, -6,  6, -6, -3,  3,  4,  2,  1,  2],
    [-6,  6, -6,  6, -4, -2,  2,  4, -3,  3,  3, -3, -2, -1, -1, -2],
    [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
    [-6,  6, -6,  6, -3, -3,  3,  3, -4,  4,  2, -2, -2, -2, -1, -1],
    [ 4, -4,  4, -4,  2,  2, -2, -2,  2, -2, -2,  2,  1,  1,  1,  1]
], dtype=float)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _correlate2d(im, kernel):
    """Apply 2-D correlation using OpenCV.

    cv2.filter2D performs correlation (no kernel flip), matching the
    behaviour of scipy.ndimage.correlate.

    Parameters
    ----------
    im : ndarray, shape (H, W), float64
        Input image.
    kernel : ndarray
        2-D correlation kernel.

    Returns
    -------
    ndarray, same shape as *im*.
    """
    # Ensure float32 for OpenCV performance; convert back to float64
    im32 = im.astype(np.float32)
    k32 = kernel.astype(np.float32)
    result = cv2.filter2D(im32, -1, k32, borderType=cv2.BORDER_REFLECT_101)
    return result.astype(np.float64)


def _remap_bilinear(im, map_x, map_y):
    """Bilinear remap using OpenCV.

    Parameters
    ----------
    im : ndarray (H, W), float64
    map_x, map_y : ndarray (H, W), float32
        0-based destination-to-source coordinate maps.

    Returns
    -------
    warped : ndarray (H, W), float64
        Out-of-boundary pixels are filled with NaN.
    """
    im32 = im.astype(np.float32)
    warped = cv2.remap(im32, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=float('nan'))
    return warped.astype(np.float64)


def _remap_cubic(im, map_x, map_y):
    """Bicubic remap using OpenCV.

    Parameters
    ----------
    im : ndarray (H, W), float64
    map_x, map_y : ndarray (H, W), float32
        0-based destination-to-source coordinate maps.

    Returns
    -------
    warped : ndarray (H, W), float64
        Out-of-boundary pixels are filled with NaN.
    """
    im32 = im.astype(np.float32)
    warped = cv2.remap(im32, map_x, map_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=float('nan'))
    return warped.astype(np.float64)


# ---------------------------------------------------------------------------
# Hermite bicubic interpolation with analytical derivatives
# ---------------------------------------------------------------------------

def interp2_bicubic(Z, XI, YI, deriv_filter):
    """Hermite bicubic interpolation with analytical derivatives.

    Matches the MATLAB ``interp2_bicubic.m`` implementation exactly.
    Uses 1-based coordinates (like MATLAB) internally.

    The inner polynomial evaluation loop is delegated to the Numba-compiled
    ``eval_bicubic_polynomial`` when available, falling back to a pure-NumPy
    implementation otherwise.

    Parameters
    ----------
    Z : ndarray, shape (H, W)
        Input image.
    XI : ndarray, shape (H, W)
        Query x-coordinates (1-based, MATLAB convention).
    YI : ndarray, shape (H, W)
        Query y-coordinates (1-based, MATLAB convention).
    deriv_filter : ndarray, 1-D
        Derivative filter for computing grid derivatives.

    Returns
    -------
    ZI : ndarray, same shape as *XI*
        Interpolated values.
    ZXI : ndarray
        x-derivative at query points.
    ZYI : ndarray
        y-derivative at query points.
    """
    sy, sx = Z.shape
    input_size = XI.shape

    XI_flat = XI.ravel()
    YI_flat = YI.ravel()
    N = len(XI_flat)

    # Floor / ceil (1-based)
    fXI = np.floor(XI_flat).astype(int)
    cXI = fXI + 1
    fYI = np.floor(YI_flat).astype(int)
    cYI = fYI + 1
    oob = (fXI < 1) | (cXI > sx) | (fYI < 1) | (cYI > sy)

    # Clamp to valid 1-based range
    fXI = np.clip(fXI, 1, sx)
    cXI = np.clip(cXI, 1, sx)
    fYI = np.clip(fYI, 1, sy)
    cYI = np.clip(cYI, 1, sy)

    # Convert to 0-based for numpy indexing
    fXI0 = fXI - 1
    cXI0 = cXI - 1
    fYI0 = fYI - 1
    cYI0 = cYI - 1

    # Function values at 4 neighbours
    Z00 = Z[fYI0, fXI0]
    Z01 = Z[cYI0, fXI0]
    Z10 = Z[fYI0, cXI0]
    Z11 = Z[cYI0, cXI0]

    # Compute grid derivatives using cv2.filter2D (correlation, same as scipy)
    dfilter_x = deriv_filter.reshape(1, -1)
    dfilter_y = deriv_filter.reshape(-1, 1)
    dfilter_xy = np.outer(deriv_filter, deriv_filter)

    DX = _correlate2d(Z, dfilter_x)
    DY = _correlate2d(Z, dfilter_y)
    DXY = _correlate2d(Z, dfilter_xy)

    DX00 = DX[fYI0, fXI0];  DX01 = DX[cYI0, fXI0]
    DX10 = DX[fYI0, cXI0];  DX11 = DX[cYI0, cXI0]
    DY00 = DY[fYI0, fXI0];  DY01 = DY[cYI0, fXI0]
    DY10 = DY[fYI0, cXI0];  DY11 = DY[cYI0, cXI0]
    DXY00 = DXY[fYI0, fXI0]; DXY01 = DXY[cYI0, fXI0]
    DXY10 = DXY[fYI0, cXI0]; DXY11 = DXY[cYI0, cXI0]

    # Build V matrix (16, N)
    V = np.array([
        Z00, Z10, Z11, Z01,
        DX00, DX10, DX11, DX01,
        DY00, DY10, DY11, DY01,
        DXY00, DXY10, DXY11, DXY01
    ])  # (16, N)

    # Coefficient matrix: C = W @ V -> (16, N)
    C = _W_BICUBIC @ V

    # Fractional coordinates
    alpha_x = XI_flat - np.floor(XI_flat)
    alpha_y = YI_flat - np.floor(YI_flat)
    alpha_x[oob] = 0.0
    alpha_y[oob] = 0.0

    # Try Numba-accelerated polynomial evaluation, fall back to NumPy
    try:
        from flow_fast._accel.bicubic_interp_nb import eval_bicubic_polynomial
        ZI, ZXI, ZYI = eval_bicubic_polynomial(C, alpha_x, alpha_y, oob)
    except ImportError:
        ZI, ZXI, ZYI = _eval_bicubic_polynomial_numpy(C, alpha_x, alpha_y, oob, N)

    ZI[oob] = np.nan

    return ZI.reshape(input_size), ZXI.reshape(input_size), ZYI.reshape(input_size)


def _eval_bicubic_polynomial_numpy(C, alpha_x, alpha_y, oob, N):
    """Pure-NumPy fallback for the bicubic polynomial evaluation."""
    ZI = np.zeros(N)
    ZXI = np.zeros(N)
    ZYI = np.zeros(N)

    idx = 0
    for i in range(4):
        for j in range(4):
            c = C[idx]
            ax_i = alpha_x ** i if i > 0 else np.ones(N)
            ay_j = alpha_y ** j if j > 0 else np.ones(N)
            ZI += c * ax_i * ay_j
            if i > 0:
                ax_im1 = alpha_x ** (i - 1) if i > 1 else np.ones(N)
                ZXI += i * c * ax_im1 * ay_j
            if j > 0:
                ay_jm1 = alpha_y ** (j - 1) if j > 1 else np.ones(N)
                ZYI += j * c * ax_i * ay_jm1
            idx += 1

    return ZI, ZXI, ZYI


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def partial_deriv(images, uv, interp_method='cubic', deriv_filter=None, blend=0.5):
    """Compute spatiotemporal derivatives with warping.

    Matches the MATLAB ``partial_deriv.m`` implementation exactly.
    Uses cv2.remap instead of scipy.ndimage.map_coordinates and
    cv2.filter2D instead of scipy.ndimage.correlate for speed.

    Parameters
    ----------
    images : ndarray, shape (H, W, 2) or (H, W, C) with C even
        Image pair concatenated along the last axis.  For grayscale,
        ``images[:,:,0]`` is frame 1 and ``images[:,:,1]`` is frame 2.
        For colour, the first half of channels is frame 1 and the second
        half is frame 2.
    uv : ndarray, shape (H, W, 2)
        Current flow estimate (u, v).
    interp_method : str
        ``'cubic'``, ``'bi-cubic'``, or ``'bi-linear'``.
    deriv_filter : ndarray or None
        1-D derivative filter.  Default: 5-point central difference.
    blend : float
        Blend ratio for spatial derivatives (0.5 = average of warped and
        original).

    Returns
    -------
    It : ndarray
        Temporal derivative.
    Ix : ndarray
        Spatial derivative in x.
    Iy : ndarray
        Spatial derivative in y.
    """
    if deriv_filter is None:
        deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0

    # Split into two frames
    if images.shape[2] == 2:
        im1 = images[:, :, 0]
        im2 = images[:, :, 1]
    else:
        nc = images.shape[2] // 2
        im1 = images[:, :, :nc]
        im2 = images[:, :, nc:]

    H, W = im1.shape[:2]
    dfilter_x = deriv_filter.reshape(1, -1)
    dfilter_y = deriv_filter.reshape(-1, 1)

    # 1-based coordinate grids (MATLAB meshgrid(1:W, 1:H))
    x_grid, y_grid = np.meshgrid(np.arange(1, W + 1, dtype=float),
                                  np.arange(1, H + 1, dtype=float))
    x2 = x_grid + uv[:, :, 0]
    y2 = y_grid + uv[:, :, 1]

    # Out-of-boundary mask (1-based coords)
    B = (x2 > W) | (x2 < 1) | (y2 > H) | (y2 < 1)

    # Convert to 0-based for OpenCV remap (float32 maps)
    map_x = (x2 - 1.0).astype(np.float32)
    map_y = (y2 - 1.0).astype(np.float32)

    if interp_method == 'bi-cubic':
        # ---- Hermite bicubic with analytical derivatives ----
        if im2.ndim == 2:
            warpIm, Ix, Iy = interp2_bicubic(im2, x2, y2, deriv_filter)
            It = warpIm - im1

            nan_mask = np.isnan(warpIm)
            It[nan_mask] = 0.0

            # Blend spatial derivatives with img1 derivatives
            I1x = _correlate2d(im1, dfilter_x)
            I1y = _correlate2d(im1, dfilter_y)
            Ix = blend * Ix + (1 - blend) * I1x
            Iy = blend * Iy + (1 - blend) * I1y
            Ix[nan_mask] = 0.0
            Iy[nan_mask] = 0.0
        else:
            nc = im2.shape[2]
            warpIm = np.zeros_like(im1)
            Ix = np.zeros_like(im1)
            Iy = np.zeros_like(im1)
            for c in range(nc):
                warpIm[:, :, c], Ix[:, :, c], Iy[:, :, c] = \
                    interp2_bicubic(im2[:, :, c], x2, y2, deriv_filter)
            It = warpIm - im1

            nan_mask = np.isnan(warpIm)
            It[nan_mask] = 0.0

            for c in range(nc):
                I1x = _correlate2d(im1[:, :, c], dfilter_x)
                I1y = _correlate2d(im1[:, :, c], dfilter_y)
                Ix[:, :, c] = blend * Ix[:, :, c] + (1 - blend) * I1x
                Iy[:, :, c] = blend * Iy[:, :, c] + (1 - blend) * I1y

            Ix[nan_mask] = 0.0
            Iy[nan_mask] = 0.0

    elif interp_method in ('bi-linear', 'cubic'):
        # ---- Standard: compute derivative, then warp via cv2.remap ----
        interp_flag = cv2.INTER_LINEAR if interp_method == 'bi-linear' else cv2.INTER_CUBIC
        # Use BORDER_REPLICATE to avoid NaN at boundaries from cubic kernel
        # overshoot, then zero out truly out-of-bounds pixels via mask B.
        border_mode = cv2.BORDER_REPLICATE

        if im2.ndim == 2:
            # Warp im2
            im2_32 = im2.astype(np.float32)
            warpIm = cv2.remap(im2_32, map_x, map_y, interp_flag,
                               borderMode=border_mode).astype(np.float64)
            It = warpIm - im1
            It[B] = 0.0

            # Derivative on im2, then warp derivative
            I2x = _correlate2d(im2, dfilter_x)
            I2y = _correlate2d(im2, dfilter_y)
            Ix_w = cv2.remap(I2x.astype(np.float32), map_x, map_y, interp_flag,
                             borderMode=border_mode).astype(np.float64)
            Iy_w = cv2.remap(I2y.astype(np.float32), map_x, map_y, interp_flag,
                             borderMode=border_mode).astype(np.float64)

            # Blend with img1 derivatives
            I1x = _correlate2d(im1, dfilter_x)
            I1y = _correlate2d(im1, dfilter_y)
            Ix = blend * Ix_w + (1 - blend) * I1x
            Iy = blend * Iy_w + (1 - blend) * I1y
            Ix[B] = 0.0
            Iy[B] = 0.0
        else:
            nc = im2.shape[2]
            It = np.zeros_like(im1)
            Ix = np.zeros_like(im1)
            Iy = np.zeros_like(im1)
            B3 = np.broadcast_to(B[:, :, np.newaxis], im1.shape)

            for c in range(nc):
                warp_c = cv2.remap(im2[:, :, c].astype(np.float32),
                                   map_x, map_y, interp_flag,
                                   borderMode=border_mode).astype(np.float64)
                It[:, :, c] = warp_c - im1[:, :, c]

                I2x = _correlate2d(im2[:, :, c], dfilter_x)
                I2y = _correlate2d(im2[:, :, c], dfilter_y)
                Ix_w = cv2.remap(I2x.astype(np.float32), map_x, map_y,
                                 interp_flag,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=float('nan')).astype(np.float64)
                Iy_w = cv2.remap(I2y.astype(np.float32), map_x, map_y,
                                 interp_flag,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=float('nan')).astype(np.float64)

                I1x = _correlate2d(im1[:, :, c], dfilter_x)
                I1y = _correlate2d(im1[:, :, c], dfilter_y)
                Ix[:, :, c] = blend * Ix_w + (1 - blend) * I1x
                Iy[:, :, c] = blend * Iy_w + (1 - blend) * I1y

            It[B3] = 0.0
            Ix[B3] = 0.0
            Iy[B3] = 0.0
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")

    return It, Ix, Iy
