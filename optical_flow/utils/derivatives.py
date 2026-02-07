"""Spatiotemporal derivatives for optical flow estimation."""
import numpy as np
from scipy.ndimage import correlate, map_coordinates


# Hermite bicubic coefficient matrix (Numerical Recipes)
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


def interp2_bicubic(Z, XI, YI, deriv_filter):
    """Hermite bicubic interpolation with analytical derivatives.

    Matches the MATLAB interp2_bicubic.m implementation exactly.
    Uses 1-based coordinates (like MATLAB) internally.

    Args:
        Z: Input image (H, W).
        XI: Query x-coordinates (1-based, like MATLAB).
        YI: Query y-coordinates (1-based, like MATLAB).
        deriv_filter: 1D derivative filter for computing grid derivatives.

    Returns:
        ZI: Interpolated values.
        ZXI: x-derivative at query points.
        ZYI: y-derivative at query points.
    """
    sy, sx = Z.shape
    input_size = XI.shape

    XI_flat = XI.ravel()
    YI_flat = YI.ravel()
    N = len(XI_flat)

    # Out-of-boundary mask
    fXI = np.floor(XI_flat).astype(int)
    cXI = fXI + 1
    fYI = np.floor(YI_flat).astype(int)
    cYI = fYI + 1
    oob = (fXI < 1) | (cXI > sx) | (fYI < 1) | (cYI > sy)

    # Clamp to valid range (1-based)
    fXI = np.clip(fXI, 1, sx)
    cXI = np.clip(cXI, 1, sx)
    fYI = np.clip(fYI, 1, sy)
    cYI = np.clip(cYI, 1, sy)

    # Convert to 0-based for numpy indexing
    fXI0 = fXI - 1
    cXI0 = cXI - 1
    fYI0 = fYI - 1
    cYI0 = cYI - 1

    # Function values at 4 neighbors
    Z00 = Z[fYI0, fXI0]
    Z01 = Z[cYI0, fXI0]
    Z10 = Z[fYI0, cXI0]
    Z11 = Z[cYI0, cXI0]

    # Compute derivatives on grid using the filter
    dfilter_x = deriv_filter.reshape(1, -1)
    dfilter_y = deriv_filter.reshape(-1, 1)
    dfilter_xy = np.convolve(deriv_filter, deriv_filter).reshape(-1, 1) @ \
                 np.ones((1, 1))
    # Actually the cross-derivative filter is conv2(Dxfilter, Dyfilter, 'full')
    dfilter_xy = np.outer(deriv_filter, deriv_filter)

    DX = correlate(Z, dfilter_x, mode='reflect')
    DY = correlate(Z, dfilter_y, mode='reflect')
    DXY = correlate(Z, dfilter_xy, mode='reflect')

    DX00 = DX[fYI0, fXI0]
    DX01 = DX[cYI0, fXI0]
    DX10 = DX[fYI0, cXI0]
    DX11 = DX[cYI0, cXI0]

    DY00 = DY[fYI0, fXI0]
    DY01 = DY[cYI0, fXI0]
    DY10 = DY[fYI0, cXI0]
    DY11 = DY[cYI0, cXI0]

    DXY00 = DXY[fYI0, fXI0]
    DXY01 = DXY[cYI0, fXI0]
    DXY10 = DXY[fYI0, cXI0]
    DXY11 = DXY[cYI0, cXI0]

    # Build V matrix (16, N)
    V = np.array([
        Z00, Z10, Z11, Z01,
        DX00, DX10, DX11, DX01,
        DY00, DY10, DY11, DY01,
        DXY00, DXY10, DXY11, DXY01
    ])  # (16, N)

    # Compute coefficients: C = W @ V  -> (16, N)
    C = _W_BICUBIC @ V

    # Fractional coordinates
    alpha_x = XI_flat - np.floor(XI_flat)
    alpha_y = YI_flat - np.floor(YI_flat)

    # Zero out-of-boundary alphas
    alpha_x[oob] = 0.0
    alpha_y[oob] = 0.0

    # Evaluate polynomial and derivatives
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

    # Mark out-of-boundary as NaN
    ZI[oob] = np.nan

    return ZI.reshape(input_size), ZXI.reshape(input_size), ZYI.reshape(input_size)


def partial_deriv(images, uv, interp_method='cubic', deriv_filter=None, blend=0.5):
    """Compute spatiotemporal derivatives with warping.

    Matches the MATLAB partial_deriv.m implementation exactly.

    Args:
        images: Image pair (H, W, 2) for grayscale, or (H, W, C) with C even.
        uv: Current flow estimate (H, W, 2).
        interp_method: 'cubic', 'bi-cubic', or 'bi-linear'.
        deriv_filter: 1D derivative filter. Default: 5-point central difference.
        blend: Blend ratio for spatial derivs (0.5 = average of warped and original).

    Returns:
        It: Temporal derivative (H, W) or (H, W, C).
        Ix: Spatial derivative in x (H, W) or (H, W, C).
        Iy: Spatial derivative in y (H, W) or (H, W, C).
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

    # 1-based coordinate grids (matching MATLAB meshgrid(1:W, 1:H))
    x_grid, y_grid = np.meshgrid(np.arange(1, W + 1, dtype=float),
                                  np.arange(1, H + 1, dtype=float))
    x2 = x_grid + uv[:, :, 0]
    y2 = y_grid + uv[:, :, 1]

    # Out-of-boundary mask (1-based coords)
    B = (x2 > W) | (x2 < 1) | (y2 > H) | (y2 < 1)

    if interp_method == 'bi-cubic':
        # Hermite bicubic interpolation with analytical derivatives
        if im2.ndim == 2:
            warpIm, Ix, Iy = interp2_bicubic(im2, x2, y2, deriv_filter)
            It = warpIm - im1

            # Zero NaN pixels (out of boundary)
            nan_mask = np.isnan(warpIm)
            It[nan_mask] = 0.0

            # Blend spatial derivatives with img1 derivatives
            I1x = correlate(im1, dfilter_x, mode='reflect')
            I1y = correlate(im1, dfilter_y, mode='reflect')
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
            if nan_mask.ndim == 3:
                any_nan = np.any(nan_mask, axis=2)
            else:
                any_nan = nan_mask
            It[nan_mask] = 0.0

            # Blend with img1 derivatives per channel
            for c in range(nc):
                I1x = correlate(im1[:, :, c], dfilter_x, mode='reflect')
                I1y = correlate(im1[:, :, c], dfilter_y, mode='reflect')
                Ix[:, :, c] = blend * Ix[:, :, c] + (1 - blend) * I1x
                Iy[:, :, c] = blend * Iy[:, :, c] + (1 - blend) * I1y

            Ix[nan_mask] = 0.0
            Iy[nan_mask] = 0.0

    elif interp_method in ('bi-linear', 'cubic'):
        # Standard interpolation: compute derivative then warp (matching MATLAB)
        order = 1 if interp_method == 'bi-linear' else 3

        # Convert to 0-based for map_coordinates
        x2_0 = x2 - 1.0
        y2_0 = y2 - 1.0

        if im2.ndim == 2:
            warpIm = map_coordinates(im2, [y2_0, x2_0], order=order,
                                     mode='constant', cval=np.nan)
            It = warpIm - im1
            It[B] = 0.0

            # Compute derivative on img2, then warp the derivative
            I2x = correlate(im2, dfilter_x, mode='reflect')
            I2y = correlate(im2, dfilter_y, mode='reflect')
            Ix_w = map_coordinates(I2x, [y2_0, x2_0], order=order,
                                   mode='constant', cval=np.nan)
            Iy_w = map_coordinates(I2y, [y2_0, x2_0], order=order,
                                   mode='constant', cval=np.nan)

            # Blend with img1 derivatives
            I1x = correlate(im1, dfilter_x, mode='reflect')
            I1y = correlate(im1, dfilter_y, mode='reflect')
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
                warp_c = map_coordinates(im2[:, :, c], [y2_0, x2_0],
                                         order=order, mode='constant', cval=np.nan)
                It[:, :, c] = warp_c - im1[:, :, c]

                I2x = correlate(im2[:, :, c], dfilter_x, mode='reflect')
                I2y = correlate(im2[:, :, c], dfilter_y, mode='reflect')
                Ix_w = map_coordinates(I2x, [y2_0, x2_0], order=order,
                                       mode='constant', cval=np.nan)
                Iy_w = map_coordinates(I2y, [y2_0, x2_0], order=order,
                                       mode='constant', cval=np.nan)

                I1x = correlate(im1[:, :, c], dfilter_x, mode='reflect')
                I1y = correlate(im1[:, :, c], dfilter_y, mode='reflect')
                Ix[:, :, c] = blend * Ix_w + (1 - blend) * I1x
                Iy[:, :, c] = blend * Iy_w + (1 - blend) * I1y

            It[B3] = 0.0
            Ix[B3] = 0.0
            Iy[B3] = 0.0
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")

    return It, Ix, Iy
