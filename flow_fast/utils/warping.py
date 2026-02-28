"""Flow field resampling for coarse-to-fine estimation.

Accelerated version: replaces scipy.ndimage.map_coordinates with
cv2.resize for bilinear flow resampling.

Public API
----------
resample_flow(uv, target_sz, method)
"""

import numpy as np
import cv2


def resample_flow(uv, target_sz, method='bilinear'):
    """Resize flow field and scale magnitudes proportionally.

    Matches MATLAB: ``imresize(uv(:,:,1), sz, 'bilinear') * ratio``
    where ``ratio = sz(1) / size(uv, 1)`` (height ratio for both
    components).

    Uses cv2.resize instead of scipy.ndimage.map_coordinates for speed.

    Parameters
    ----------
    uv : ndarray, shape (H, W, 2)
        Flow field.
    target_sz : tuple of int
        Target size ``(H_new, W_new)``.
    method : str
        Interpolation method (only ``'bilinear'`` is supported).

    Returns
    -------
    uv_new : ndarray, shape (H_new, W_new, 2)
        Resized and magnitude-scaled flow field.
    """
    if uv.shape[0] == target_sz[0] and uv.shape[1] == target_sz[1]:
        return uv.copy()

    H, W = uv.shape[:2]
    new_H, new_W = target_sz

    # MATLAB uses height ratio for both u and v
    ratio = new_H / H

    # cv2.resize takes (width, height) as dsize
    uv_new = np.zeros((new_H, new_W, 2), dtype=np.float64)

    u32 = uv[:, :, 0].astype(np.float32)
    v32 = uv[:, :, 1].astype(np.float32)

    uv_new[:, :, 0] = cv2.resize(u32, (new_W, new_H),
                                  interpolation=cv2.INTER_LINEAR).astype(np.float64) * ratio
    uv_new[:, :, 1] = cv2.resize(v32, (new_W, new_H),
                                  interpolation=cv2.INTER_LINEAR).astype(np.float64) * ratio

    return uv_new
