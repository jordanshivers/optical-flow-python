"""
High-level interface for optical flow estimation.

Reimplements estimate_flow_interface.m from the MATLAB codebase.
"""
import numpy as np
from optical_flow.methods.config import load_of_method
from optical_flow.utils.image_processing import scale_image


def estimate_flow(im1, im2, method='classic+nl-fast', params=None):
    """Estimate optical flow between two images.

    This is the main high-level interface for optical flow estimation.
    It handles preprocessing (grayscale conversion, Lab color space for
    non-local term) and method configuration.

    Args:
        im1: First image, (H, W) grayscale or (H, W, 3) RGB, float or uint8.
        im2: Second image, same size/format as im1.
        method: Method name string. See load_of_method for options.
            Common choices:
            - 'classic+nl-fast': Fast Classic+NL (recommended)
            - 'classic+nl': Full Classic+NL
            - 'hs': Horn-Schunck with texture
            - 'ba': Black-Anandan with texture
            - 'classic-c': Classic with charbonnier
            - 'classic++': Classic++ with gen. charbonnier
        params: Optional dict of parameter overrides.

    Returns:
        uv: Estimated optical flow (H, W, 2). uv[:,:,0] = horizontal,
            uv[:,:,1] = vertical.
    """
    im1 = np.asarray(im1, dtype=float)
    im2 = np.asarray(im2, dtype=float)

    # Load configured method
    ope = load_of_method(method)

    # Apply parameter overrides
    if params is not None:
        ope.parse_input_parameter(params)

    # Handle color images
    if im1.ndim == 3 and im1.shape[2] >= 3:
        # Convert to grayscale for flow computation
        gray1 = _rgb2gray(im1)
        gray2 = _rgb2gray(im2)
        ope.images = np.stack([gray1, gray2], axis=2)
    else:
        ope.images = np.stack([im1, im2], axis=2) if im1.ndim == 2 else np.concatenate([im1, im2], axis=2)

    # Use color for weighted non-local term
    if ope.color_images is not None:
        if im1.ndim == 3 and im1.shape[2] >= 3:
            # Convert to Lab color space
            lab1 = _rgb2lab(im1)
            # Scale each channel to [0, 255]
            for j in range(lab1.shape[2]):
                lab1[:, :, j] = scale_image(lab1[:, :, j], 0, 255)
        else:
            lab1 = im1.copy()
        ope.color_images = lab1

    # Compute flow
    H, W = im1.shape[:2]
    init = np.zeros((H, W, 2))
    uv = ope.compute_flow(init)

    return uv


def _rgb2gray(im):
    """Convert RGB image to grayscale, matching MATLAB's double(rgb2gray(uint8(im))).

    MATLAB quantizes through uint8 first, so we do the same for exact matching.
    MATLAB's uint8() rounds half away from zero: floor(x + 0.5).
    """
    if im.ndim == 2:
        return im
    # Match MATLAB: uint8(im) rounds half-away-from-zero
    im_uint8 = np.clip(np.floor(im + 0.5), 0, 255).astype(np.uint8)
    gray = 0.2989 * im_uint8[:, :, 0].astype(float) + \
           0.5870 * im_uint8[:, :, 1].astype(float) + \
           0.1140 * im_uint8[:, :, 2].astype(float)
    # MATLAB uint8() rounds half-away-from-zero
    return np.floor(gray + 0.5)


def _rgb2lab(im):
    """Convert RGB image to CIE Lab color space.

    Matches the MATLAB RGB2Lab.m implementation exactly
    (BT.709 primaries, D65 white point).
    """
    im = np.asarray(im, dtype=float)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    # Normalize to [0, 1] if needed
    if R.max() > 1.0 or G.max() > 1.0 or B.max() > 1.0:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    T = 0.008856

    M, N = R.shape
    s = M * N
    RGB = np.array([R.ravel(), G.ravel(), B.ravel()])

    # RGB to XYZ (BT.709)
    MAT = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])
    XYZ = MAT @ RGB

    # Normalize for D65 white point
    X = XYZ[0] / 0.950456
    Y = XYZ[1]
    Z = XYZ[2] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    Y3 = Y ** (1.0 / 3.0)

    fX = XT * X ** (1.0 / 3.0) + (~XT) * (7.787 * X + 16.0 / 116.0)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16.0 / 116.0)
    fZ = ZT * Z ** (1.0 / 3.0) + (~ZT) * (7.787 * Z + 16.0 / 116.0)

    L = (YT * (116.0 * Y3 - 16.0) + (~YT) * (903.3 * Y)).reshape(M, N)
    a = (500.0 * (fX - fY)).reshape(M, N)
    b = (200.0 * (fY - fZ)).reshape(M, N)

    return np.stack([L, a, b], axis=2)
