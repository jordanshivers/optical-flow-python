"""
Black-Anandan optical flow estimation with robust penalties and GNC optimization.

Black, M.J. and Anandan, P. "The robust estimation of multiple motions:
Parametric and piecewise-smooth flow fields." CVIU, 63(1), pp. 75-104, 1996.
"""
import time
import copy
import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter

from optical_flow.methods.base import BaseOpticalFlow
from optical_flow.robust.robust_function import RobustFunction
from optical_flow.utils.image_processing import (
    scale_image, structure_texture_decomposition_rof
)
from optical_flow.utils.derivatives import partial_deriv
from optical_flow.utils.sparse_ops import make_convn_mat
from optical_flow.utils.warping import resample_flow


class BAOpticalFlow(BaseOpticalFlow):
    """Black & Anandan optical flow with robust estimation and GNC."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
        self.lambda_q = 1.0
        self.gnc_iters = 3
        self.alpha = 1.0
        self.max_iters = 10
        self.max_linear = 1
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25
        self.texture = False
        self.fc = False
        self.blend = 0.5
        self.alp = 0.95
        self.auto_level = True
        self.solver = 'backslash'
        self.interpolation_method = 'cubic'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.limit_update = True
        self.display = False
        self.sor_max_iters = 10000
        self.color_images = None

        method = 'lorentzian'
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        self.rho_spatial_u = [RobustFunction(method, 0.03), RobustFunction(method, 0.03)]
        self.rho_spatial_v = [RobustFunction(method, 0.03), RobustFunction(method, 0.03)]
        self.rho_data = RobustFunction(method, 1.5)

    def compute_flow(self, init=None, gt=None):
        """Compute flow with GNC optimization and coarse-to-fine pyramid.

        Args:
            init: Initial flow (H, W, 2). Default: zeros.
            gt: Optional ground truth (H, W, 2) for evaluation during computation.

        Returns:
            uv: Estimated flow field (H, W, 2).
        """
        sz = (self.images.shape[0], self.images.shape[1])

        if init is None:
            uv = np.zeros((*sz, 2))
        else:
            uv = init.copy()

        # Preprocess images
        if self.texture:
            images = structure_texture_decomposition_rof(self.images, 1.0 / 8, 100, self.alp)
        elif self.fc:
            from optical_flow.utils.image_processing import fspecial_gaussian
            f = fspecial_gaussian(5, 1.5)
            from scipy.ndimage import correlate
            images = self.images - self.alp * correlate(self.images, f, mode='reflect')
            images = scale_image(images, 0, 255)
        else:
            images = scale_image(self.images, 0, 255)

        if self.auto_level:
            self.pyramid_levels = self._auto_pyramid_levels(images)

        # Build pyramids for GNC stage 1
        pyramid_images = self._build_pyramid(images, self.pyramid_levels, self.pyramid_spacing)

        # Build pyramids for GNC stages 2+
        gnc_pyramid_images = self._build_pyramid(images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing)

        start_time = time.time()

        # Save original alpha
        alpha_orig = self.alpha

        for ignc in range(self.gnc_iters):
            if self.display:
                print(f"GNC stage: {ignc + 1}")

            if ignc == 0:
                pyramid_levels = self.pyramid_levels
                current_pyramid = pyramid_images
            else:
                pyramid_levels = self.gnc_pyramid_levels
                current_pyramid = gnc_pyramid_images

            # Coarse to fine
            for l in range(pyramid_levels - 1, -1, -1):
                if self.display:
                    print(f"  Pyramid level: {l + 1}")

                small = copy.copy(self)
                small.images = current_pyramid[l]

                if ignc == 0:
                    small.max_linear = 1

                nsz = (current_pyramid[l].shape[0], current_pyramid[l].shape[1])
                uv = resample_flow(uv, nsz)
                uv = small.compute_flow_base(uv)

            # Update GNC alpha (linearly from 1 to 0)
            if self.gnc_iters > 1:
                new_alpha = 1 - (ignc + 1) / (self.gnc_iters - 1)
                self.alpha = min(self.alpha, new_alpha)
                self.alpha = max(0, self.alpha)

            elapsed = (time.time() - start_time) / 60
            print(f"GNC stage {ignc + 1} finished, {elapsed:.2f} minutes passed")

        # Restore alpha
        self.alpha = alpha_orig

        return uv

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level with IRLS and optional GNC blending.

        Args:
            uv: Current flow estimate (H, W, 2).

        Returns:
            uv: Updated flow estimate.
        """
        # Build quadratic relaxation version
        qua = copy.copy(self)
        qua.lambda_ = self.lambda_q

        a = self.rho_spatial_u[0].param
        b_param = self.rho_data.param
        ta = b_param[0] / a[0]

        # Quadratic spatial
        qua.rho_spatial_u = [RobustFunction('quadratic', 1) for _ in self.rho_spatial_u]
        qua.rho_spatial_v = [RobustFunction('quadratic', 1) for _ in self.rho_spatial_v]
        qua.rho_data = RobustFunction('quadratic', ta)

        for i in range(self.max_iters):
            duv = np.zeros_like(uv)

            # Compute derivatives
            It, Ix, Iy = partial_deriv(
                self.images, uv, self.interpolation_method, self.deriv_filter, self.blend
            )

            for j in range(self.max_linear):
                # GNC blending
                if self.alpha == 1:
                    A, b_vec, _, _ = qua.flow_operator(uv, duv, It, Ix, Iy)
                elif self.alpha > 0:
                    A, b_vec, _, _ = qua.flow_operator(uv, duv, It, Ix, Iy)
                    A1, b1, _, _ = self.flow_operator(uv, duv, It, Ix, Iy)
                    A = self.alpha * A + (1 - self.alpha) * A1
                    b_vec = self.alpha * b_vec + (1 - self.alpha) * b1
                elif self.alpha == 0:
                    A, b_vec, _, _ = self.flow_operator(uv, duv, It, Ix, Iy)
                else:
                    raise ValueError(f"Invalid GNC alpha: {self.alpha}")

                x = self._solve_linear_system(A, b_vec, uv.shape)

                if self.limit_update:
                    x = np.clip(x, -1, 1)

                if self.display:
                    print(f"    Iter: {i + 1} {j + 1} (delta: {np.linalg.norm(x - duv):.6f})")

                duv = x

                uv0 = uv.copy()
                uv = uv + duv

                if self.median_filter_size is not None:
                    uv[:, :, 0] = median_filter(uv[:, :, 0], size=self.median_filter_size, mode='reflect')
                    uv[:, :, 1] = median_filter(uv[:, :, 1], size=self.median_filter_size, mode='reflect')

                duv = uv - uv0
                uv = uv0

            uv = uv + duv

        return uv

    def flow_operator(self, uv, duv, It, Ix, Iy):
        """Build the linear system with filter-based spatial term and IRLS weights.

        Args:
            uv: Current flow estimate (H, W, 2).
            duv: Current flow increment (H, W, 2).
            It: Temporal derivative.
            Ix: Horizontal spatial derivative.
            Iy: Vertical spatial derivative.

        Returns:
            A: Sparse system matrix.
            b: Right-hand side vector.
            params: None.
            iterative: bool.
        """
        sz = (Ix.shape[0], Ix.shape[1])
        npixels = sz[0] * sz[1]

        # Spatial term with IRLS weights
        S = self.spatial_filters
        FU = sparse.csc_matrix((npixels, npixels))
        FV = sparse.csc_matrix((npixels, npixels))

        pp_su_all = []
        pp_sv_all = []

        for i in range(len(S)):
            FMi, FMi_csr, nnz_per_row = self._get_cached_convmat_csr(
                S[i], sz, 'valid', 'sameswap')

            # Filter the current flow + increment
            u_filt = FMi @ (uv[:, :, 0] + duv[:, :, 0]).ravel(order='F')
            v_filt = FMi @ (uv[:, :, 1] + duv[:, :, 1]).ravel(order='F')

            # IRLS weights
            pp_su = self.rho_spatial_u[i].deriv_over_x(u_filt)
            pp_sv = self.rho_spatial_v[i].deriv_over_x(v_filt)

            pp_su_all.append(pp_su)
            pp_sv_all.append(pp_sv)

            FU = FU + self._weighted_filter_product(FMi, FMi_csr, nnz_per_row, pp_su)
            FV = FV + self._weighted_filter_product(FMi, FMi_csr, nnz_per_row, pp_sv)

        # Data term
        if Ix.ndim == 3:
            # Linearize It
            It_lin = It.copy()
            for c in range(It.shape[2]):
                It_lin[:, :, c] = (It[:, :, c]
                                   + Ix[:, :, c] * duv[:, :, 0]
                                   + Iy[:, :, c] * duv[:, :, 1])
            pp_d = self.rho_data.deriv_over_x(It_lin.reshape(-1))
            pp_d = np.mean(pp_d.reshape(It_lin.shape), axis=2).ravel(order='F')
            Ix2 = np.mean(Ix**2, axis=2)
            Iy2 = np.mean(Iy**2, axis=2)
            Ixy = np.mean(Ix * Iy, axis=2)
            Itx = np.mean(It_lin * Ix, axis=2)
            Ity = np.mean(It_lin * Iy, axis=2)
        else:
            It_lin = It + Ix * duv[:, :, 0] + Iy * duv[:, :, 1]
            pp_d = self.rho_data.deriv_over_x(It_lin.ravel(order='F'))
            Ix2 = Ix**2
            Iy2 = Iy**2
            Ixy = Ix * Iy
            Itx = It_lin * Ix
            Ity = It_lin * Iy

        # Build A directly: combine data diagonals with spatial term (no intermediate M)
        lam = self.lambda_
        duu = sparse.diags(pp_d * Ix2.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FU
        dvv = sparse.diags(pp_d * Iy2.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FV
        dduv = sparse.diags(pp_d * Ixy.ravel(order='F'), 0, shape=(npixels, npixels))

        A = sparse.bmat([[duu, dduv], [dduv, dvv]])

        # Right-hand side (compute FU/FV @ u/v directly, no M needed)
        u_vec = uv[:, :, 0].ravel(order='F')
        v_vec = uv[:, :, 1].ravel(order='F')
        b_vec = np.concatenate([
            -lam * (FU @ u_vec) - pp_d * Itx.ravel(order='F'),
            -lam * (FV @ v_vec) - pp_d * Ity.ravel(order='F')
        ])

        # Check if iterative
        iterative = True
        if len(pp_su_all) > 0:
            all_uniform = all(
                (pp.max() - pp.min() < 1e-6) for pp in pp_su_all + pp_sv_all
            )
            if all_uniform and (pp_d.max() - pp_d.min() < 1e-6):
                iterative = False

        return A, b_vec, None, iterative
