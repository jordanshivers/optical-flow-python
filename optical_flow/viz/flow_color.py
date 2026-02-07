"""Middlebury color coding for optical flow visualization."""
import numpy as np


def make_colorwheel():
    """Build the Middlebury 55-bin colorwheel.

    Returns:
        colorwheel: (55, 3) array of RGB values [0, 255].
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """Compute color image from normalized flow components.

    Args:
        u, v: Flow components (H, W), should be normalized.

    Returns:
        img: Color image (H, W, 3), uint8.
    """
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi  # [-1, 1]

    fk = (a + 1) / 2.0 * (ncols - 1)  # [0, ncols-1]
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.zeros((*u.shape, 3), dtype=np.uint8)

    for i in range(3):
        tmp = colorwheel[k0, i] / 255.0 * (1 - f) + colorwheel[k1, i] / 255.0 * f
        # Increase saturation with radius
        tmp = 1 - rad * (1 - tmp)
        # Clamp values > 1 (large flows)
        tmp[rad > 1] = tmp[rad > 1] * 0.75
        img[:, :, i] = np.floor(255 * np.clip(tmp, 0, 1)).astype(np.uint8)

    return img


def flow_to_color(flow, max_flow=None):
    """Convert flow field to RGB color image using Middlebury color coding.

    Args:
        flow: (H, W, 2) flow field.
        max_flow: Max flow magnitude for normalization. If None, auto-compute.

    Returns:
        img: (H, W, 3) uint8 RGB image.
    """
    UNKNOWN_FLOW_THRESH = 1e9

    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()

    unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)

    if max_flow is not None:
        max_rad = max_flow
    else:
        mag = np.sqrt(u[~unknown] ** 2 + v[~unknown] ** 2) if np.any(~unknown) else np.array([0])
        max_rad = mag.max() if len(mag) > 0 else 1.0

    max_rad = max(max_rad, 1e-8)
    u = u / max_rad
    v = v / max_rad

    img = compute_color(u, v)
    img[unknown] = 0

    return img
