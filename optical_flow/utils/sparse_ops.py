"""Sparse convolution matrix construction for optical flow."""
import numpy as np
from scipy import sparse


def convmtxn(F, sz):
    """Build N-D convolution matrix (full convolution).

    Constructs sparse matrix M such that M @ vec(x) = vec(conv2(x, F, 'full'))
    where vec() is column-major (Fortran order) vectorization.

    Args:
        F: Filter kernel (2D array).
        sz: Image size (H, W).

    Returns:
        M: Sparse CSC matrix for full convolution.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    fsz = F.shape
    out_sz = (sz[0] + fsz[0] - 1, sz[1] + fsz[1] - 1)
    n_in = sz[0] * sz[1]
    n_out = out_sz[0] * out_sz[1]

    # Find nonzero filter taps
    nz_fi, nz_fj = np.nonzero(F)
    if len(nz_fi) == 0:
        return sparse.csc_matrix((n_out, n_in))

    nz_vals = F[nz_fi, nz_fj]

    # For each nonzero filter tap, generate ALL (row, col) pairs at once
    # using vectorized index computation (Fortran/column-major ordering)
    i_arr = np.arange(sz[0])
    j_arr = np.arange(sz[1])
    # jj shape: (sz[0], sz[1]), ii shape: (sz[0], sz[1])
    ii, jj = np.meshgrid(i_arr, j_arr, indexing='ij')
    ii_flat = ii.ravel()  # column-major not needed here since meshgrid 'ij' + ravel default is C
    jj_flat = jj.ravel()
    # col indices in Fortran order: col_idx = j * sz[0] + i
    col_all = jj_flat * sz[0] + ii_flat  # shape (n_in,)

    n_in_pixels = len(col_all)
    n_taps = len(nz_fi)

    # Broadcast: for each tap t, row_idx = (jj + fj[t]) * out_sz[0] + (ii + fi[t])
    # Shape: (n_taps, n_in)
    row_all = ((jj_flat[np.newaxis, :] + nz_fj[:, np.newaxis]) * out_sz[0]
               + (ii_flat[np.newaxis, :] + nz_fi[:, np.newaxis]))

    # Tile col indices for each tap
    cols = np.tile(col_all, n_taps)
    rows = row_all.ravel()
    vals = np.repeat(nz_vals, n_in_pixels)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_out, n_in)).tocsc()


def make_convn_mat(F, sz, shape='full', pad=None):
    """Build sparse convolution matrix with shape control.

    Args:
        F: Filter kernel (2D).
        sz: Image size (H, W) tuple.
        shape: 'full', 'same', 'valid'.
        pad: Optional 'sameswap' - embeds valid-sized output into same-sized output.

    Returns:
        M: Sparse matrix.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    fsz = np.array(F.shape)
    sz = tuple(int(s) for s in sz)

    M_full = convmtxn(F, sz)
    out_full = (sz[0] + fsz[0] - 1, sz[1] + fsz[1] - 1)

    if shape == 'full':
        return M_full

    elif shape == 'same':
        offset = ((fsz - 1) // 2).astype(int)
        return _crop_convmat(M_full, out_full, sz, offset)

    elif shape == 'valid':
        valid_sz = (sz[0] - fsz[0] + 1, sz[1] - fsz[1] + 1)
        if valid_sz[0] <= 0 or valid_sz[1] <= 0:
            return sparse.csc_matrix((0, sz[0] * sz[1]))

        if pad == 'sameswap':
            # Get valid rows from full matrix (vectorized)
            vi, vj = np.meshgrid(np.arange(valid_sz[0]), np.arange(valid_sz[1]), indexing='ij')
            full_rows = ((vj + fsz[1] - 1) * out_full[0] + (vi + fsz[0] - 1)).ravel(order='F')
            M_valid = M_full[full_rows, :]
            n_valid = valid_sz[0] * valid_sz[1]
            n_same = sz[0] * sz[1]

            offset_i = (fsz[0] - 1) // 2
            offset_j = (fsz[1] - 1) // 2

            # Build embedding matrix (vectorized)
            ei, ej = np.meshgrid(np.arange(valid_sz[0]), np.arange(valid_sz[1]), indexing='ij')
            embed_rows = ((ej + offset_j) * sz[0] + (ei + offset_i)).ravel(order='F')

            E = sparse.coo_matrix(
                (np.ones(n_valid), (embed_rows, np.arange(n_valid))),
                shape=(n_same, n_valid)
            ).tocsc()

            return E @ M_valid
        else:
            # Get valid rows from full matrix (vectorized)
            vi, vj = np.meshgrid(np.arange(valid_sz[0]), np.arange(valid_sz[1]), indexing='ij')
            full_rows = ((vj + fsz[1] - 1) * out_full[0] + (vi + fsz[0] - 1)).ravel(order='F')
            return M_full[full_rows, :]

    else:
        raise ValueError(f"Unknown shape: {shape}")


def _crop_convmat(M_full, out_full_sz, target_sz, offset):
    """Crop rows of full convolution matrix to target size."""
    ti, tj = np.meshgrid(np.arange(target_sz[0]), np.arange(target_sz[1]), indexing='ij')
    rows = ((tj + offset[1]) * out_full_sz[0] + (ti + offset[0])).ravel(order='F')
    return M_full[rows, :]


def make_imfilter_mat(F, sz, boundary='replicate', shape='same'):
    """Build sparse matrix for image filtering with boundary handling.

    Equivalent to MATLAB's imfilter with 'corr' mode.

    Args:
        F: Filter kernel (2D).
        sz: Image size (H, W).
        boundary: 'replicate' (nearest), '0' (zero), 'symmetric' (reflect).
        shape: 'same'.

    Returns:
        M: Sparse filtering matrix using column-major indexing.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    fsz = np.array(F.shape)
    n_pixels = sz[0] * sz[1]
    hfsz = ((fsz - 1) // 2).astype(int)

    # Find nonzero filter taps
    nz_fi, nz_fj = np.nonzero(F)
    if len(nz_fi) == 0:
        return sparse.csc_matrix((n_pixels, n_pixels))

    nz_vals = F[nz_fi, nz_fj]

    # All pixel coordinates (Fortran ordering: iterate i fastest)
    pi, pj = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing='ij')
    pi_flat = pi.ravel(order='F')  # Fortran order to match MATLAB linear indexing
    pj_flat = pj.ravel(order='F')
    out_idx = pj_flat * sz[0] + pi_flat  # shape (n_pixels,)

    all_rows = []
    all_cols = []
    all_vals = []

    for t in range(len(nz_fi)):
        fi = nz_fi[t]
        fj = nz_fj[t]
        fval = nz_vals[t]

        # Source pixel coordinates for this filter tap
        si = pi_flat + fi - hfsz[0]
        sj = pj_flat + fj - hfsz[1]

        if boundary == '0':
            # Zero boundary: skip out-of-bounds
            valid = (si >= 0) & (si < sz[0]) & (sj >= 0) & (sj < sz[1])
            if not np.any(valid):
                continue
            si_v = si[valid]
            sj_v = sj[valid]
            src_idx = sj_v * sz[0] + si_v
            all_rows.append(out_idx[valid])
            all_cols.append(src_idx)
            all_vals.append(np.full(np.count_nonzero(valid), fval))
        elif boundary == 'replicate':
            si_c = np.clip(si, 0, sz[0] - 1)
            sj_c = np.clip(sj, 0, sz[1] - 1)
            src_idx = sj_c * sz[0] + si_c
            all_rows.append(out_idx)
            all_cols.append(src_idx)
            all_vals.append(np.full(n_pixels, fval))
        elif boundary == 'symmetric':
            # Reflect: mirror at boundaries
            si_r = si.copy()
            sj_r = sj.copy()
            # Reflect negative indices
            neg_i = si_r < 0
            si_r[neg_i] = -si_r[neg_i] - 1
            neg_j = sj_r < 0
            sj_r[neg_j] = -sj_r[neg_j] - 1
            # Reflect indices beyond size
            over_i = si_r >= sz[0]
            si_r[over_i] = 2 * sz[0] - si_r[over_i] - 1
            over_j = sj_r >= sz[1]
            sj_r[over_j] = 2 * sz[1] - sj_r[over_j] - 1
            # Final clip for safety
            si_r = np.clip(si_r, 0, sz[0] - 1)
            sj_r = np.clip(sj_r, 0, sz[1] - 1)
            src_idx = sj_r * sz[0] + si_r
            all_rows.append(out_idx)
            all_cols.append(src_idx)
            all_vals.append(np.full(n_pixels, fval))

    if len(all_rows) == 0:
        return sparse.csc_matrix((n_pixels, n_pixels))

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_pixels, n_pixels)).tocsc()
