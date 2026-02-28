"""Numba-accelerated weighted median filtering for optical flow denoising.

Replaces the pure-Python pixel loop in weighted_median.py with a JIT-compiled
implementation using Numba.  The outer row loop is parallelized with
``numba.prange``; each pixel is independent, so the workload scales linearly
with the number of available threads.

Key optimizations
-----------------
* Thread-local scratch arrays are allocated per row (not per pixel) to avoid
  repeated heap allocations inside the inner loop.
* Insertion sort is used for the ~(2*hsz+1)^2 element arrays because it is
  faster than mergesort / quicksort for N < ~256.
* The weighted median is computed in a single cumulative-sum scan over the
  sorted weights.
"""

import numba
import numpy as np


# ---------------------------------------------------------------------------
# Helper: insertion sort that simultaneously sorts *values* and rearranges
# *weights* to match.  Both arrays are modified in place over [0, n).
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _insertion_argsort(values, indices, n):
    """Sort *values[0:n]* and produce the corresponding index permutation
    in *indices[0:n]*.

    Parameters
    ----------
    values : float64[:]
        Array of values to sort.  Modified in place.
    indices : int64[:]
        On output, ``indices[k]`` is the original position of the element
        now at ``values[k]``.
    n : int
        Number of elements to consider (may be less than ``len(values)``).
    """
    for k in range(n):
        indices[k] = k
    for i in range(1, n):
        key_val = values[i]
        key_idx = indices[i]
        j = i - 1
        while j >= 0 and values[j] > key_val:
            values[j + 1] = values[j]
            indices[j + 1] = indices[j]
            j -= 1
        values[j + 1] = key_val
        indices[j + 1] = key_idx


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------
@numba.njit(parallel=True, cache=True)
def weighted_median_filter_numba(
    u_pad, v_pad, color_pad, occ_pad,
    H, W, hsz, sigma_i,
    uv_out_u, uv_out_v,
):
    """Color-guided weighted median filter (Numba hot path).

    All input arrays must already be reflect-padded by *hsz* on every spatial
    side so that boundary checks are unnecessary inside the loop.

    Parameters
    ----------
    u_pad : float64[:, :]
        Padded horizontal flow component, shape ``(H + 2*hsz, W + 2*hsz)``.
    v_pad : float64[:, :]
        Padded vertical flow component, same shape as *u_pad*.
    color_pad : float64[:, :, :]
        Padded Lab colour image, shape ``(H + 2*hsz, W + 2*hsz, 3)``.
    occ_pad : float64[:, :]
        Padded occlusion weight map, same spatial shape as *u_pad*.
    H : int
        Original (unpadded) image height.
    W : int
        Original (unpadded) image width.
    hsz : int
        Half-window size.  The full patch is ``(2*hsz + 1)^2`` pixels.
    sigma_i : float64
        Colour-similarity bandwidth (pixels with a smaller colour distance
        receive a higher weight).
    uv_out_u : float64[:, :]
        Output array for filtered horizontal flow, shape ``(H, W)``.
        Written in place.
    uv_out_v : float64[:, :]
        Output array for filtered vertical flow, shape ``(H, W)``.
        Written in place.
    """
    patch_len = (2 * hsz + 1) * (2 * hsz + 1)
    inv_2sigma2 = 1.0 / (2.0 * sigma_i * sigma_i)

    # --- parallel over rows -------------------------------------------------
    for i in numba.prange(H):
        # Per-row scratch (avoids allocation inside the inner pixel loop)
        weights = np.empty(patch_len, dtype=np.float64)
        u_vals  = np.empty(patch_len, dtype=np.float64)
        v_vals  = np.empty(patch_len, dtype=np.float64)
        sort_buf = np.empty(patch_len, dtype=np.float64)
        idx_buf  = np.empty(patch_len, dtype=np.int64)

        for j in range(W):
            # Centre pixel in padded coordinates
            ci = i + hsz
            cj = j + hsz

            center_c0 = color_pad[ci, cj, 0]
            center_c1 = color_pad[ci, cj, 1]
            center_c2 = color_pad[ci, cj, 2]

            # ---- gather patch values and compute weights -------------------
            k = 0
            for di in range(-hsz, hsz + 1):
                pi = ci + di
                for dj in range(-hsz, hsz + 1):
                    pj = cj + dj

                    # Flow values
                    u_vals[k] = u_pad[pi, pj]
                    v_vals[k] = v_pad[pi, pj]

                    # Colour distance (squared L2 in Lab space)
                    d0 = color_pad[pi, pj, 0] - center_c0
                    d1 = color_pad[pi, pj, 1] - center_c1
                    d2 = color_pad[pi, pj, 2] - center_c2
                    cdiff = d0 * d0 + d1 * d1 + d2 * d2

                    w = np.exp(-cdiff * inv_2sigma2) * occ_pad[pi, pj]
                    if w < 1e-10:
                        w = 1e-10
                    weights[k] = w
                    k += 1

            # ---- weighted median for u channel -----------------------------
            # Copy u_vals into sort_buf so we can sort in place
            for m in range(patch_len):
                sort_buf[m] = u_vals[m]
            _insertion_argsort(sort_buf, idx_buf, patch_len)

            half_total = 0.0
            for m in range(patch_len):
                half_total += weights[idx_buf[m]]
            half_total *= 0.5

            cumw = 0.0
            med_u = sort_buf[patch_len - 1]  # fallback
            for m in range(patch_len):
                cumw += weights[idx_buf[m]]
                if cumw >= half_total:
                    med_u = sort_buf[m]
                    break
            uv_out_u[i, j] = med_u

            # ---- weighted median for v channel -----------------------------
            for m in range(patch_len):
                sort_buf[m] = v_vals[m]
            _insertion_argsort(sort_buf, idx_buf, patch_len)

            cumw = 0.0
            med_v = sort_buf[patch_len - 1]
            for m in range(patch_len):
                cumw += weights[idx_buf[m]]
                if cumw >= half_total:
                    med_v = sort_buf[m]
                    break
            uv_out_v[i, j] = med_v
