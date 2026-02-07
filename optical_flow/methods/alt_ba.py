"""
Alternative Black-Anandan optical flow estimation.

Uses an auxiliary flow field (uvhat) updated via Li-Osher iterative
median formula, coupled to the main flow field.

References:
    Sun, D.; Roth, S. & Black, M.J. "Secrets of Optical Flow Estimation
    and Their Principles." CVPR 2010.
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
from optical_flow.utils.denoising import denoise_LO


class AltBAOpticalFlow(BaseOpticalFlow):
    """Alternative BA optical flow with coupling and non-local terms."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 5.0
        self.lambda_q = 5.0

        self.sor_max_iters = 10000
        self.limit_update = True
        self.display = False
        self.solver = 'backslash'
        self.warping_mode = 'backward'
        self.texture = False
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.median_filter_size = None
        self.interpolation_method = 'cubic'

        # GNC settings
        self.gnc_iters = 3
        self.alpha = 1.0
        self.max_iters = 10
        self.max_linear = 1

        # Pyramid (GNC stage 1)
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0

        # Pyramid (GNC stages 2+)
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25

        # Robust functions
        method = 'lorentzian'
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        self.rho_spatial_u = [RobustFunction(method, 0.03), RobustFunction(method, 0.03)]
        self.rho_spatial_v = [RobustFunction(method, 0.03), RobustFunction(method, 0.03)]
        self.rho_data = RobustFunction(method, 1.5)

        # Alt-BA specific
        self.seg = None
        self.mfT = 15
        self.imfsz = [7, 7]
        self.qterm = True
        self.lambda2 = 0.1         # coupling weight
        self.lambda3 = 1.0         # non-local weight
        self.weightRatio = 1.0     # lambda2/lambda3
        self.itersLO = 1           # Li-Osher iterations
        self.color_images = None
        self.replacement = True
        self.rho_couple = RobustFunction('charbonnier', 1e-3)
        self.auto_level = True

    def compute_flow(self, init=None, gt=None):
        """Compute flow with GNC optimization, coupling, and Li-Osher denoising.

        Args:
            init: Initial flow (H, W, 2). Default: zeros.
            gt: Optional ground truth (H, W, 2) for evaluation.

        Returns:
            uv: Estimated flow field (H, W, 2). Returns uvhat at the end.
        """
        sz = (self.images.shape[0], self.images.shape[1])

        if init is None:
            uv = np.zeros((*sz, 2))
        else:
            uv = init.copy()

        uvhat = uv.copy()

        # Preprocess
        if self.texture:
            images = structure_texture_decomposition_rof(self.images)
        else:
            images = scale_image(self.images, 0, 255)

        # Auto pyramid levels
        self.pyramid_levels = self._auto_pyramid_levels(images)

        # Build pyramids (GNC stage 1)
        pyramid_images = self._build_pyramid(images, self.pyramid_levels, self.pyramid_spacing)
        org_pyramid_images = self._build_pyramid(self.images, self.pyramid_levels, self.pyramid_spacing)

        # Build pyramids (GNC stages 2+)
        gnc_pyramid_images = self._build_pyramid(images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing)
        org_gnc_pyramid_images = self._build_pyramid(
            self.images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing
        )

        start_time = time.time()

        for ignc in range(self.gnc_iters):
            # Set replacement flag: True except at last GNC stage
            if ignc == self.gnc_iters - 1:
                self.replacement = False
            else:
                self.replacement = True

            if self.display:
                print(f"GNC stage: {ignc + 1}")

            if ignc == 0:
                pyramid_levels = self.pyramid_levels
            else:
                pyramid_levels = self.gnc_pyramid_levels

            # Coarse to fine
            for l in range(pyramid_levels - 1, -1, -1):
                if self.display:
                    print(f"  Pyramid level: {l + 1}")

                small = copy.copy(self)

                if ignc == 0:
                    nsz = (pyramid_images[l].shape[0], pyramid_images[l].shape[1])
                    small.images = pyramid_images[l]
                    im1 = org_pyramid_images[l]
                    if im1.ndim == 3:
                        im1 = im1[:, :, 0]
                else:
                    nsz = (gnc_pyramid_images[l].shape[0], gnc_pyramid_images[l].shape[1])
                    small.images = gnc_pyramid_images[l]
                    im1 = org_gnc_pyramid_images[l]
                    if im1.ndim == 3:
                        im1 = im1[:, :, 0]

                # Turn off coupling at the very beginning
                if l == pyramid_levels - 1 and ignc == 0:
                    small.qterm = False
                else:
                    small.qterm = True

                uv = resample_flow(uv, nsz)
                uvhat = resample_flow(uvhat, nsz)

                small.seg = small.images[:, :, 0] if small.images.ndim == 3 else small.images

                uv, uvhat = small.compute_flow_base(uv, uvhat)

            # Update GNC alpha
            if self.gnc_iters > 1:
                new_alpha = 1 - (ignc + 1) / (self.gnc_iters - 1)
                self.alpha = min(self.alpha, new_alpha)
                self.alpha = max(0, self.alpha)

            elapsed = (time.time() - start_time) / 60
            print(f"GNC stage {ignc + 1} finished, {elapsed:.2f} minutes passed")

            if gt is not None:
                from optical_flow.evaluation.metrics import flow_angular_error
                aae, stdae, aepe = flow_angular_error(
                    gt[:, :, 0], gt[:, :, 1], uv[:, :, 0], uv[:, :, 1], 0
                )
                print(f"  AAE {aae:.3f} STD {stdae:.3f} EPE {aepe:.3f}")

        # Return uvhat as final result
        uv = uvhat
        return uv

    def compute_flow_base(self, uv, uvhat):
        """Compute flow at a single pyramid level with coupling and Li-Osher denoising.

        Args:
            uv: Current flow estimate (H, W, 2).
            uvhat: Auxiliary flow estimate (H, W, 2).

        Returns:
            uv: Updated flow estimate.
            uvhat: Updated auxiliary flow estimate.
        """
        # Build quadratic relaxation version
        qua = copy.copy(self)
        qua.lambda_ = self.lambda_q

        # Quadratic spatial
        qua.rho_spatial_u = [RobustFunction('quadratic', 1) for _ in self.rho_spatial_u]
        qua.rho_spatial_v = [RobustFunction('quadratic', 1) for _ in self.rho_spatial_v]
        qua.rho_data = RobustFunction('quadratic', 1)

        # Lambda2 annealing schedule
        Lambda2s = np.logspace(np.log10(1e-4), np.log10(self.lambda2), self.max_iters)
        Lambda2s = np.append(Lambda2s, self.lambda2)
        lambda2 = Lambda2s[0]

        for i in range(self.max_iters):
            duv = np.zeros_like(uv)

            # Compute derivatives
            It, Ix, Iy = partial_deriv(
                self.images, uv, self.interpolation_method, self.deriv_filter
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

                # Add coupling term
                uv_flat = uv.ravel(order='F')
                uvhat_flat = uvhat.ravel(order='F')
                tmp = self.rho_couple.deriv_over_x(uv_flat - uvhat_flat)
                tmpA = sparse.diags(tmp, 0, shape=(A.shape[0], A.shape[1]))
                A = A + lambda2 * tmpA
                b_vec = b_vec + lambda2 * tmp * (uvhat_flat - uv_flat)

                x = self._solve_linear_system(A, b_vec, uv.shape)

                if self.limit_update:
                    x = np.clip(x, -1, 1)

                if self.display:
                    print(f"    Iter: {i + 1} {j + 1} (delta: {np.linalg.norm(x - duv):.6f})")

                duv = x

            # Update flow
            uv = uv + duv

            # Update uvhat via Li-Osher denoising
            uvhat[:, :, 0] = denoise_LO(
                uv[:, :, 0], self.median_filter_size,
                lambda2 / self.lambda3, self.itersLO
            )
            uvhat[:, :, 1] = denoise_LO(
                uv[:, :, 1], self.median_filter_size,
                lambda2 / self.lambda3, self.itersLO
            )

            # Replace uv with uvhat
            if self.replacement:
                uv = uvhat.copy()

            # Anneal lambda2
            lambda2 = Lambda2s[i + 1]

        return uv, uvhat

    def flow_operator(self, uv, duv, It, Ix, Iy):
        """Build the linear system with filter-based spatial term and IRLS weights.

        Identical structure to BA/Classic+NL flow_operator.

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

        # Spatial term
        S = self.spatial_filters
        FU = sparse.csc_matrix((npixels, npixels))
        FV = sparse.csc_matrix((npixels, npixels))

        pp_su_all = []
        pp_sv_all = []

        for i in range(len(S)):
            FMi, FMi_csr, nnz_per_row = self._get_cached_convmat_csr(
                S[i], sz, 'valid', 'sameswap')

            u_filt = FMi @ (uv[:, :, 0] + duv[:, :, 0]).ravel(order='F')
            v_filt = FMi @ (uv[:, :, 1] + duv[:, :, 1]).ravel(order='F')

            pp_su = self.rho_spatial_u[i].deriv_over_x(u_filt)
            pp_sv = self.rho_spatial_v[i].deriv_over_x(v_filt)

            pp_su_all.append(pp_su)
            pp_sv_all.append(pp_sv)

            FU = FU + self._weighted_filter_product(FMi, FMi_csr, nnz_per_row, pp_su)
            FV = FV + self._weighted_filter_product(FMi, FMi_csr, nnz_per_row, pp_sv)

        # Data term
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Linearize It
        if It.ndim == 3:
            It_lin = It.copy()
            for c in range(It.shape[2]):
                It_lin[:, :, c] = (It[:, :, c]
                                   + Ix[:, :, c] * duv[:, :, 0]
                                   + Iy[:, :, c] * duv[:, :, 1])
            pp_d = self.rho_data.deriv_over_x(It_lin.ravel())
            pp_d_2d = np.mean(pp_d.reshape(It_lin.shape), axis=2)
            Ix2 = np.mean(Ix2, axis=2)
            Iy2 = np.mean(Iy2, axis=2)
            Ixy = np.mean(Ixy, axis=2)
            Itx = np.mean(It_lin * Ix, axis=2)
            Ity = np.mean(It_lin * Iy, axis=2)
            pp_d = pp_d_2d.ravel(order='F')
        else:
            It_lin = It + Ix * duv[:, :, 0] + Iy * duv[:, :, 1]
            pp_d = self.rho_data.deriv_over_x(It_lin.ravel(order='F'))
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
