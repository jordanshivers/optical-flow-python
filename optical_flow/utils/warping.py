"""Flow field resampling for coarse-to-fine estimation."""
import numpy as np
from scipy.ndimage import map_coordinates


def resample_flow(uv, target_sz, method='bilinear'):
    """Resize flow field and scale magnitudes proportionally.

    Matches MATLAB: imresize(uv(:,:,1), sz, 'bilinear') * ratio
    where ratio = sz(1)/size(uv,1) (height ratio for both components).

    Args:
        uv: Flow field (H, W, 2).
        target_sz: Target size (H_new, W_new).

    Returns:
        uv_new: Resized and scaled flow field (H_new, W_new, 2).
    """
    if uv.shape[0] == target_sz[0] and uv.shape[1] == target_sz[1]:
        return uv.copy()

    H, W = uv.shape[:2]
    new_H, new_W = target_sz

    # MATLAB uses height ratio for both u and v
    ratio = new_H / H

    # MATLAB imresize coordinate mapping: u = (out + 0.5) / scale - 0.5
    scale_H = new_H / H
    scale_W = new_W / W
    out_rows = (np.arange(new_H) + 0.5) / scale_H - 0.5
    out_cols = (np.arange(new_W) + 0.5) / scale_W - 0.5

    # Clip to valid range (matches MATLAB boundary behavior)
    out_rows = np.clip(out_rows, 0, H - 1)
    out_cols = np.clip(out_cols, 0, W - 1)

    coords_r, coords_c = np.meshgrid(out_rows, out_cols, indexing='ij')

    uv_new = np.zeros((new_H, new_W, 2))
    uv_new[:, :, 0] = map_coordinates(uv[:, :, 0], [coords_r, coords_c],
                                       order=1, mode='nearest') * ratio
    uv_new[:, :, 1] = map_coordinates(uv[:, :, 1], [coords_r, coords_c],
                                       order=1, mode='nearest') * ratio
    return uv_new
