"""
High-level interface for optical flow estimation.
"""
import numpy as np
from flow_fast.methods.config import load_of_method
from flow_fast.utils.image_processing import scale_image


_DEEP_METHODS = {'raft', 'sea-raft', 'waft'}


def estimate_flow(im1, im2, method='classic+nl-fast', params=None):
    """Estimate optical flow between two images.

    This is the main high-level interface for optical flow estimation.
    Same API as optical_flow.estimate_flow but uses accelerated backends.

    Args:
        im1: First image, (H, W) grayscale or (H, W, 3) RGB, float or uint8.
        im2: Second image, same size/format as im1.
        method: Method name string. See load_of_method for options.
            Also: 'lk' (Lucas-Kanade).
            Deep learning methods: 'raft', 'sea-raft', 'waft' (require PyTorch).
        params: Optional dict of parameter overrides.

    Returns:
        uv: Estimated optical flow (H, W, 2).
    """
    im1 = np.asarray(im1, dtype=float)
    im2 = np.asarray(im2, dtype=float)

    # Deep learning methods bypass variational preprocessing
    if method.lower() in _DEEP_METHODS:
        return _estimate_flow_deep(im1, im2, method, params)

    ope = load_of_method(method)

    if params is not None:
        ope.parse_input_parameter(params)

    if im1.ndim == 3 and im1.shape[2] >= 3:
        gray1 = _rgb2gray(im1)
        gray2 = _rgb2gray(im2)
        ope.images = np.stack([gray1, gray2], axis=2)
    else:
        ope.images = np.stack([im1, im2], axis=2) if im1.ndim == 2 else np.concatenate([im1, im2], axis=2)

    if ope.color_images is not None:
        if im1.ndim == 3 and im1.shape[2] >= 3:
            lab1 = _rgb2lab(im1)
            for j in range(lab1.shape[2]):
                lab1[:, :, j] = scale_image(lab1[:, :, j], 0, 255)
        else:
            lab1 = im1.copy()
        ope.color_images = lab1

    H, W = im1.shape[:2]
    init = np.zeros((H, W, 2))
    uv = ope.compute_flow(init)

    return uv


def _estimate_flow_deep(im1, im2, method, params):
    """Dispatch to deep learning flow methods."""
    from optical_flow.methods.deep import load_deep_method
    ope = load_deep_method(method)
    if params is not None:
        ope.parse_input_parameter(params)
    ope._im1 = im1
    ope._im2 = im2
    return ope.compute_flow()


def _rgb2gray(im):
    """Convert RGB image to grayscale, matching MATLAB's double(rgb2gray(uint8(im)))."""
    if im.ndim == 2:
        return im
    im_uint8 = np.clip(np.floor(im + 0.5), 0, 255).astype(np.uint8)
    gray = 0.2989 * im_uint8[:, :, 0].astype(float) + \
           0.5870 * im_uint8[:, :, 1].astype(float) + \
           0.1140 * im_uint8[:, :, 2].astype(float)
    return np.floor(gray + 0.5)


def _rgb2lab(im):
    """Convert RGB image to CIE Lab color space.

    Uses OpenCV if available, falls back to manual implementation.
    """
    try:
        import cv2
        im_uint8 = np.clip(im, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(im_uint8, cv2.COLOR_RGB2Lab).astype(np.float64)
        return lab
    except ImportError:
        pass

    # Manual implementation matching MATLAB RGB2Lab.m
    im = np.asarray(im, dtype=float)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    if R.max() > 1.0 or G.max() > 1.0 or B.max() > 1.0:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    T = 0.008856
    M, N = R.shape
    RGB = np.array([R.ravel(), G.ravel(), B.ravel()])

    MAT = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])
    XYZ = MAT @ RGB

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
