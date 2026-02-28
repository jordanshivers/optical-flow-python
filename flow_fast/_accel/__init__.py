"""Accelerated computational backends using Numba JIT compilation."""

import numpy as np


def warmup():
    """Pre-compile all Numba JIT functions with small dummy inputs."""
    from flow_fast._accel.weighted_median_nb import weighted_median_filter_numba
    from flow_fast._accel.rof_denoise_nb import rof_structure_2d_numba
    from flow_fast._accel.penalties_nb import (
        quadratic_deriv_over_x, lorentzian_deriv_over_x,
        charbonnier_deriv_over_x, generalized_charbonnier_deriv_over_x,
    )
    from flow_fast._accel.bicubic_interp_nb import eval_bicubic_polynomial

    # Weighted median: small 4x4 test
    u = np.zeros((4, 4), dtype=np.float64)
    v = np.zeros((4, 4), dtype=np.float64)
    color = np.zeros((8, 8, 3), dtype=np.float64)
    occ = np.ones((8, 8), dtype=np.float64)
    ou = np.zeros((4, 4), dtype=np.float64)
    ov = np.zeros((4, 4), dtype=np.float64)
    weighted_median_filter_numba(u, v, color, occ, 4, 4, 2, 7.0, ou, ov)

    # ROF: small 4x4 test
    im = np.zeros((4, 4), dtype=np.float64)
    rof_structure_2d_numba(im, 0.125, 2)

    # Penalties: small vector
    x = np.zeros(4, dtype=np.float64)
    quadratic_deriv_over_x(x, 1.0)
    lorentzian_deriv_over_x(x, 1.0)
    charbonnier_deriv_over_x(x, 1e-3)
    generalized_charbonnier_deriv_over_x(x, 1e-3, 0.45)

    # Bicubic: small test
    C = np.zeros((16, 4), dtype=np.float64)
    ax = np.zeros(4, dtype=np.float64)
    ay = np.zeros(4, dtype=np.float64)
    oob = np.zeros(4, dtype=np.bool_)
    eval_bicubic_polynomial(C, ax, ay, oob)
