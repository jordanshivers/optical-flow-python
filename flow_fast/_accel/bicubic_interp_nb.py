"""Numba-compiled Hermite bicubic polynomial evaluation.

Replaces the Python ``for i in range(4): for j in range(4)`` loop in
``derivatives.py`` that evaluates the 4x4 bicubic polynomial and its
analytical x/y derivatives at N query points.

The key optimisation is the use of **Horner's method** (nested evaluation)
for the degree-3 polynomial in each variable, which reduces the 16
multiplications and 16 power calls per query point down to 12 fused
multiply-adds.  The loop over N query points is parallelised with
``numba.prange``.

Coefficient layout
------------------
The coefficient matrix ``C`` has shape ``(16, N)`` and is indexed as::

    C[4*i + j, n]   for  i in 0..3,  j in 0..3

where the polynomial is

    f(ax, ay) = sum_{i,j} C[4*i+j] * ax^i * ay^j

and its partial derivatives are

    df/dx = sum_{i>0, j} i * C[4*i+j] * ax^{i-1} * ay^j
    df/dy = sum_{i, j>0} j * C[4*i+j] * ax^i     * ay^{j-1}
"""

import numba
import numpy as np


@numba.njit(cache=True, parallel=True)
def eval_bicubic_polynomial(C, alpha_x, alpha_y, oob):
    """Evaluate bicubic polynomial and partial derivatives at N query points.

    Parameters
    ----------
    C : float64[:, :]
        Coefficient matrix, shape ``(16, N)``.  Row ordering is
        ``C[4*i + j]`` for ``i, j in 0..3``.
    alpha_x : float64[:]
        Fractional x-coordinates, shape ``(N,)``.  Values in [0, 1].
    alpha_y : float64[:]
        Fractional y-coordinates, shape ``(N,)``.  Values in [0, 1].
    oob : bool[:]
        Out-of-bounds mask, shape ``(N,)``.  ``True`` for points that
        should be set to NaN in the interpolated value.

    Returns
    -------
    ZI : float64[:]
        Interpolated values, shape ``(N,)``.  NaN where ``oob`` is True.
    ZXI : float64[:]
        Partial derivative w.r.t. x at each query point, shape ``(N,)``.
    ZYI : float64[:]
        Partial derivative w.r.t. y at each query point, shape ``(N,)``.
    """
    N = alpha_x.shape[0]
    ZI = np.empty(N, dtype=np.float64)
    ZXI = np.empty(N, dtype=np.float64)
    ZYI = np.empty(N, dtype=np.float64)

    for n in numba.prange(N):
        ax = alpha_x[n]
        ay = alpha_y[n]

        # Pre-compute powers of ax and ay
        ax2 = ax * ax
        ax3 = ax2 * ax
        ay2 = ay * ay
        ay3 = ay2 * ay

        # Store powers in local arrays for indexing convenience
        # ax_pow[k] = ax^k,  ay_pow[k] = ay^k
        ax_pow_0 = 1.0
        ax_pow_1 = ax
        ax_pow_2 = ax2
        ax_pow_3 = ax3

        ay_pow_0 = 1.0
        ay_pow_1 = ay
        ay_pow_2 = ay2
        ay_pow_3 = ay3

        # ---- Evaluate f(ax, ay) using Horner's method in ay ----
        # For each i, compute  row_i = C[4i+0] + C[4i+1]*ay + C[4i+2]*ay^2 + C[4i+3]*ay^3
        # Then ZI = row_0 + row_1*ax + row_2*ax^2 + row_3*ax^3
        #
        # Similarly for derivatives.
        #
        # Direct Horner in ay for each i-row:
        #   row_i = ((C[4i+3]*ay + C[4i+2])*ay + C[4i+1])*ay + C[4i+0]

        # Row i=0:  coefficients C[0..3, n]
        r0 = ((C[3, n] * ay + C[2, n]) * ay + C[1, n]) * ay + C[0, n]
        # Row i=1:  coefficients C[4..7, n]
        r1 = ((C[7, n] * ay + C[6, n]) * ay + C[5, n]) * ay + C[4, n]
        # Row i=2:  coefficients C[8..11, n]
        r2 = ((C[11, n] * ay + C[10, n]) * ay + C[9, n]) * ay + C[8, n]
        # Row i=3:  coefficients C[12..15, n]
        r3 = ((C[15, n] * ay + C[14, n]) * ay + C[13, n]) * ay + C[12, n]

        # f(ax, ay) via Horner in ax:  ((r3*ax + r2)*ax + r1)*ax + r0
        val = ((r3 * ax + r2) * ax + r1) * ax + r0

        # ---- df/dx = sum_{i>0} i * row_i * ax^{i-1}  ----
        # = r1 + 2*r2*ax + 3*r3*ax^2
        # Horner: (3*r3*ax + 2*r2)*ax + r1
        dval_x = (3.0 * r3 * ax + 2.0 * r2) * ax + r1

        # ---- df/dy = sum_{i, j>0} j * C[4i+j] * ax^i * ay^{j-1}  ----
        # For each i, compute:
        #   drow_i = C[4i+1] + 2*C[4i+2]*ay + 3*C[4i+3]*ay^2
        #          = (3*C[4i+3]*ay + 2*C[4i+2])*ay + C[4i+1]
        dr0 = (3.0 * C[3, n] * ay + 2.0 * C[2, n]) * ay + C[1, n]
        dr1 = (3.0 * C[7, n] * ay + 2.0 * C[6, n]) * ay + C[5, n]
        dr2 = (3.0 * C[11, n] * ay + 2.0 * C[10, n]) * ay + C[9, n]
        dr3 = (3.0 * C[15, n] * ay + 2.0 * C[14, n]) * ay + C[13, n]

        # df/dy = dr0 + dr1*ax + dr2*ax^2 + dr3*ax^3
        # Horner: ((dr3*ax + dr2)*ax + dr1)*ax + dr0
        dval_y = ((dr3 * ax + dr2) * ax + dr1) * ax + dr0

        if oob[n]:
            ZI[n] = np.nan
        else:
            ZI[n] = val
        ZXI[n] = dval_x
        ZYI[n] = dval_y

    return ZI, ZXI, ZYI
