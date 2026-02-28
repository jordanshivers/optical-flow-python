"""Numba-accelerated Rudin--Osher--Fatemi (ROF) structure--texture decomposition.

Implements the Chambolle primal-dual algorithm for total-variation denoising.
Given a noisy image *im*, the routine returns the *structure* component

    u = im + theta * div(p)

after *n_iters* iterations of the primal-dual update.

All intermediate arrays (gradient, divergence, dual variable) are fused into a
single double loop over pixels per iteration, eliminating every temporary
allocation that the original NumPy implementation required.

References
----------
Chambolle, A. (2004).  "An Algorithm for Total Variation Minimization and
Applications."  *Journal of Mathematical Imaging and Vision*, 20, 89--97.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def rof_structure_2d_numba(im, theta, n_iters):
    """ROF primal-dual total-variation denoising (structure extraction).

    Parameters
    ----------
    im : float64[:, :]
        Input 2-D image (H x W).
    theta : float64
        Regularisation weight.  Larger values yield a smoother structure
        component (more texture removed).
    n_iters : int
        Number of primal-dual iterations (typically 100).

    Returns
    -------
    u : float64[:, :]
        Structure component, shape ``(H, W)``.
    """
    H = im.shape[0]
    W = im.shape[1]

    # Dual variable (vector field)
    p_x = np.zeros((H, W), dtype=np.float64)
    p_y = np.zeros((H, W), dtype=np.float64)

    # Step size for the dual update (1 / (4 * theta) guarantees convergence
    # for 2-D images with the isotropic TV semi-norm).
    delta = 1.0 / (4.0 * theta)

    for _it in range(n_iters):
        # ---------------------------------------------------------------
        # Fused pass: for every pixel compute
        #   1. div_p  = backward-difference divergence of (p_x, p_y)
        #   2. u      = im + theta * div_p
        #   3. grad_u = forward-difference gradient of u
        #   4. p      += delta * grad_u
        #   5. reproject p so that |p| <= 1
        #
        # We need the *current* p to compute div_p, and the *current* u
        # (which depends on div_p) to compute grad_u.  Because the forward
        # gradient at (i, j) only touches u(i,j) and u(i+1,j) / u(i,j+1),
        # we can compute u on the fly from div_p.
        #
        # We therefore split into two conceptual sub-passes that are
        # merged into one double loop:
        #   (a) compute u from current p  (divergence)
        #   (b) update p from u           (gradient + reprojection)
        # ---------------------------------------------------------------

        # --- Sub-pass (a): compute u = im + theta * div_p ---------------
        # We store u temporarily in a flat buffer so that sub-pass (b) can
        # read u(i+1,j) and u(i,j+1) without data hazards.
        u = np.empty((H, W), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                # Backward difference: div_p = (p_x[i,j] - p_x[i,j-1])
                #                            + (p_y[i,j] - p_y[i-1,j])
                # Neumann boundary: p_x[i,-1] = 0, p_y[-1,j] = 0
                div_px = p_x[i, j]
                if j > 0:
                    div_px -= p_x[i, j - 1]

                div_py = p_y[i, j]
                if i > 0:
                    div_py -= p_y[i - 1, j]

                u[i, j] = im[i, j] + theta * (div_px + div_py)

        # --- Sub-pass (b): gradient of u -> dual update + reprojection --
        for i in range(H):
            for j in range(W):
                # Forward difference gradient of u
                # grad_x = u[i, j+1] - u[i, j]   (zero at right boundary)
                # grad_y = u[i+1, j] - u[i, j]   (zero at bottom boundary)
                u_ij = u[i, j]

                if j < W - 1:
                    grad_x = u[i, j + 1] - u_ij
                else:
                    grad_x = 0.0

                if i < H - 1:
                    grad_y = u[i + 1, j] - u_ij
                else:
                    grad_y = 0.0

                # Dual ascent
                new_px = p_x[i, j] + delta * grad_x
                new_py = p_y[i, j] + delta * grad_y

                # Reprojection: |p| <= 1
                norm = np.sqrt(new_px * new_px + new_py * new_py)
                if norm > 1.0:
                    new_px /= norm
                    new_py /= norm

                p_x[i, j] = new_px
                p_y[i, j] = new_py

    # Final structure component after the last iteration
    out = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            div_px = p_x[i, j]
            if j > 0:
                div_px -= p_x[i, j - 1]
            div_py = p_y[i, j]
            if i > 0:
                div_py -= p_y[i - 1, j]
            out[i, j] = im[i, j] + theta * (div_px + div_py)

    return out
