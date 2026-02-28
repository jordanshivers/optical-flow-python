"""Optical flow evaluation metrics."""
import numpy as np


def flow_angular_error(tu, tv, u, v, border=0):
    """Compute angular error and endpoint error (Barron et al.)."""
    tu = np.asarray(tu, dtype=float)
    tv = np.asarray(tv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if border > 0:
        tu = tu[border:-border, border:-border]
        tv = tv[border:-border, border:-border]
        u = u[border:-border, border:-border]
        v = v[border:-border, border:-border]

    UNKNOWN_FLOW_THRESH = 1e9
    valid = (np.abs(tu) < UNKNOWN_FLOW_THRESH) & (np.abs(tv) < UNKNOWN_FLOW_THRESH)
    if not np.all(valid):
        tu = tu[valid]
        tv = tv[valid]
        u = u[valid]
        v = v[valid]

    n_est = 1.0 / np.sqrt(u ** 2 + v ** 2 + 1.0)
    n_gt = 1.0 / np.sqrt(tu ** 2 + tv ** 2 + 1.0)

    cos_angle = (u * tu + v * tv + 1.0) * n_est * n_gt
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    ae = np.arccos(cos_angle) * 180.0 / np.pi
    aae = np.mean(ae)
    std_ae = np.std(ae)

    epe = np.sqrt((tu - u) ** 2 + (tv - v) ** 2)
    aepe = np.mean(epe)

    return aae, std_ae, aepe
