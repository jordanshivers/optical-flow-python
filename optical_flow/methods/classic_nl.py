"""
Classic+NL optical flow estimation.

Classic formulation (Black & Anandan style) with non-local term using
weighted median filtering for occlusion handling.

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
    scale_image, fspecial_gaussian, structure_texture_decomposition_rof
)
from optical_flow.utils.derivatives import partial_deriv
from optical_flow.utils.sparse_ops import make_convn_mat
from optical_flow.utils.warping import resample_flow
from optical_flow.utils.occlusion import detect_occlusion
from optical_flow.utils.weighted_median import denoise_color_weighted_medfilt2


class ClassicNLOpticalFlow(BaseOpticalFlow):
    """Classic+NL optical flow with robust estimation and non-local median term."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
        self.lambda_q = 1.0
        self.lambda2 = 0.1       # weight for coupling term (unused here, used in alt_ba)
        self.lambda3 = 1.0       # weight for non-local term

        self.sor_max_iters = 10000
        self.limit_update = True
        self.display = False
        self.solver = 'backslash'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.texture = False
        self.fc = False
        self.median_filter_size = None
        self.interpolation_method = 'bi-cubic'

        # GNC settings
        self.gnc_iters = 3
        self.alpha = 1.0
        self.max_iters = 10
        self.max_linear = 1

        # Pyramid settings (GNC stage 1)
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0

        # Pyramid settings (GNC stages 2+)
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25

        # Robust functions: generalized charbonnier
        method = 'generalized_charbonnier'
        a = 0.45
        sig = 1e-3
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        self.rho_spatial_u = [RobustFunction(method, sig, a), RobustFunction(method, sig, a)]
        self.rho_spatial_v = [RobustFunction(method, sig, a), RobustFunction(method, sig, a)]
        self.rho_data = RobustFunction(method, sig, a)

        # Segmentation / non-local settings
        self.seg = None
        self.mfT = 15
        self.imfsz = [7, 7]
        self.filter_weight = None
        self.alp = 0.95

        self.hybrid = False
        self.area_hsz = 10
        self.affine_hsz = 4
        self.sigma_i = 7
        self.color_images = None
        self.auto_level = True
        self.input_seg = None
        self.input_occ = None
        self.fullVersion = False

    def compute_flow(self, init=None, gt=None):
        """Compute flow with GNC optimization and coarse-to-fine pyramid.

        Args:
            init: Initial flow (H, W, 2). Default: zeros.
            gt: Optional ground truth (H, W, 2) for evaluation.

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

        # For segmentation / non-local term
        org_pyramid_images = self._build_pyramid(self.images, self.pyramid_levels, self.pyramid_spacing)
        org_color_pyramid_images = self._build_pyramid(
            self.color_images, self.pyramid_levels, self.pyramid_spacing
        ) if self.color_images is not None else [None] * self.pyramid_levels

        # Build pyramids for GNC stages 2+
        gnc_pyramid_images = self._build_pyramid(images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing)
        org_gnc_pyramid_images = self._build_pyramid(
            self.images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing
        )
        org_color_gnc_pyramid_images = self._build_pyramid(
            self.color_images, self.gnc_pyramid_levels, self.gnc_pyramid_spacing
        ) if self.color_images is not None else [None] * self.gnc_pyramid_levels

        start_time = time.time()

        for ignc in range(self.gnc_iters):
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
                    small.max_linear = 1
                    im1 = org_pyramid_images[l]
                    if im1.ndim == 3:
                        im1 = im1[:, :, 0]
                    small.color_images = org_color_pyramid_images[l]
                else:
                    nsz = (gnc_pyramid_images[l].shape[0], gnc_pyramid_images[l].shape[1])
                    small.images = gnc_pyramid_images[l]
                    im1 = org_gnc_pyramid_images[l]
                    if im1.ndim == 3:
                        im1 = im1[:, :, 0]
                    small.color_images = org_color_gnc_pyramid_images[l]

                uv = resample_flow(uv, nsz)

                small.seg = im1
                # Adaptively determine half window size
                small.affine_hsz = min(4, max(2, int(np.ceil(min(nsz) / 75))))

                uv = small.compute_flow_base(uv)

            # Update GNC alpha (linearly from 1 to 0)
            if self.gnc_iters > 1:
                new_alpha = 1 - (ignc + 1) / (self.gnc_iters - 1)
                self.alpha = min(self.alpha, new_alpha)
                self.alpha = max(0, self.alpha)

            elapsed = (time.time() - start_time) / 60
            msg = f"GNC stage {ignc + 1} finished, {elapsed:.2f} minutes passed"

            if gt is not None:
                from optical_flow.evaluation.metrics import flow_angular_error
                aae, stdae, aepe = flow_angular_error(
                    gt[:, :, 0], gt[:, :, 1], uv[:, :, 0], uv[:, :, 1], 0
                )
                msg += f"  AAE {aae:.3f} STD {stdae:.3f} EPE {aepe:.3f}"

            print(msg)

        return uv

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level with IRLS and GNC blending.

        Uses weighted median filtering (non-local term) if median_filter_size is set.

        Args:
            uv: Current flow estimate (H, W, 2).

        Returns:
            uv: Updated flow estimate.
        """
        # Build quadratic relaxation version
        qua = copy.copy(self)
        qua.lambda_ = self.lambda_q

        # Quadratic spatial
        qua.rho_spatial_u = []
        qua.rho_spatial_v = []
        for i in range(len(self.rho_spatial_u)):
            a = self.rho_spatial_u[i].param
            qua.rho_spatial_u.append(RobustFunction('quadratic', a[0]))
            a = self.rho_spatial_v[i].param
            qua.rho_spatial_v.append(RobustFunction('quadratic', a[0]))

        # Quadratic data
        a = self.rho_data.param
        qua.rho_data = RobustFunction('quadratic', a[0])

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

                x = self._solve_linear_system(A, b_vec, uv.shape)

                if self.limit_update:
                    x = np.clip(x, -1, 1)

                if self.display:
                    print(f"    Iter: {i + 1} {j + 1} (delta: {np.linalg.norm(x - duv):.6f})")

                duv = x

                uv0 = uv.copy()
                uv = uv + duv

                # Apply weighted median filtering (non-local term)
                if self.median_filter_size is not None:
                    occ = detect_occlusion(uv, self.images)
                    uv = denoise_color_weighted_medfilt2(
                        uv, self.color_images, occ, self.area_hsz,
                        self.median_filter_size, self.sigma_i, self.fullVersion
                    )

                duv = uv - uv0
                uv = uv0

            # Update flow
            uv = uv + duv

        return uv

    def flow_operator(self, uv, duv, It, Ix, Iy):
        """Build the linear system with filter-based spatial term and IRLS weights.

        Same structure as BA but uses the Classic+NL penalty functions.

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
            Ix2_m = np.mean(Ix2, axis=2)
            Iy2_m = np.mean(Iy2, axis=2)
            Ixy_m = np.mean(Ixy, axis=2)
            Itx_m = np.mean(It_lin * Ix, axis=2)
            Ity_m = np.mean(It_lin * Iy, axis=2)
            pp_d = pp_d_2d.ravel(order='F')
        else:
            It_lin = It + Ix * duv[:, :, 0] + Iy * duv[:, :, 1]
            pp_d = self.rho_data.deriv_over_x(It_lin.ravel(order='F'))
            Ix2_m = Ix2
            Iy2_m = Iy2
            Ixy_m = Ixy
            Itx_m = It_lin * Ix
            Ity_m = It_lin * Iy

        # Build A directly: combine data diagonals with spatial term (no intermediate M)
        lam = self.lambda_
        duu = sparse.diags(pp_d * Ix2_m.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FU
        dvv = sparse.diags(pp_d * Iy2_m.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FV
        dduv = sparse.diags(pp_d * Ixy_m.ravel(order='F'), 0, shape=(npixels, npixels))

        A = sparse.bmat([[duu, dduv], [dduv, dvv]])

        # Right-hand side (compute FU/FV @ u/v directly, no M needed)
        u_vec = uv[:, :, 0].ravel(order='F')
        v_vec = uv[:, :, 1].ravel(order='F')
        b_vec = np.concatenate([
            -lam * (FU @ u_vec) - pp_d * Itx_m.ravel(order='F'),
            -lam * (FV @ v_vec) - pp_d * Ity_m.ravel(order='F')
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
