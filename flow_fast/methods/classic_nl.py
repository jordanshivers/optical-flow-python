"""
Classic+NL optical flow estimation.

Classic formulation with non-local term using weighted median filtering.
"""
import time
import copy
import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter

from flow_fast.methods.base import BaseOpticalFlow
from flow_fast.robust.robust_function import RobustFunction
from flow_fast.utils.image_processing import (
    scale_image, fspecial_gaussian, structure_texture_decomposition_rof
)
from flow_fast.utils.derivatives import partial_deriv
from flow_fast.utils.sparse_ops import make_convn_mat
from flow_fast.utils.warping import resample_flow
from flow_fast.utils.occlusion import detect_occlusion
from flow_fast.utils.weighted_median import denoise_color_weighted_medfilt2


class ClassicNLOpticalFlow(BaseOpticalFlow):
    """Classic+NL optical flow with robust estimation and non-local median term."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
        self.lambda_q = 1.0
        self.lambda2 = 0.1
        self.lambda3 = 1.0

        self.sor_max_iters = 10000
        self.limit_update = True
        self.display = False
        self.solver = 'auto'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.texture = False
        self.fc = False
        self.median_filter_size = None
        self.interpolation_method = 'bi-cubic'

        self.gnc_iters = 3
        self.alpha = 1.0
        self.max_iters = 10
        self.max_linear = 1

        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25

        method = 'generalized_charbonnier'
        a = 0.45
        sig = 1e-3
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        self.rho_spatial_u = [RobustFunction(method, sig, a), RobustFunction(method, sig, a)]
        self.rho_spatial_v = [RobustFunction(method, sig, a), RobustFunction(method, sig, a)]
        self.rho_data = RobustFunction(method, sig, a)

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
        """Compute flow with GNC optimization and coarse-to-fine pyramid."""
        sz = (self.images.shape[0], self.images.shape[1])

        if init is None:
            uv = np.zeros((*sz, 2))
        else:
            uv = init.copy()

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

        pyramid_images = self._build_pyramid(images, self.pyramid_levels, self.pyramid_spacing)
        org_pyramid_images = self._build_pyramid(self.images, self.pyramid_levels, self.pyramid_spacing)
        org_color_pyramid_images = self._build_pyramid(
            self.color_images, self.pyramid_levels, self.pyramid_spacing
        ) if self.color_images is not None else [None] * self.pyramid_levels

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
                small.affine_hsz = min(4, max(2, int(np.ceil(min(nsz) / 75))))

                uv = small.compute_flow_base(uv)

            if self.gnc_iters > 1:
                new_alpha = 1 - (ignc + 1) / (self.gnc_iters - 1)
                self.alpha = min(self.alpha, new_alpha)
                self.alpha = max(0, self.alpha)

            elapsed = (time.time() - start_time) / 60
            msg = f"GNC stage {ignc + 1} finished, {elapsed:.2f} minutes passed"

            if gt is not None:
                from flow_fast.evaluation.metrics import flow_angular_error
                aae, stdae, aepe = flow_angular_error(
                    gt[:, :, 0], gt[:, :, 1], uv[:, :, 0], uv[:, :, 1], 0
                )
                msg += f"  AAE {aae:.3f} STD {stdae:.3f} EPE {aepe:.3f}"

            print(msg)

        return uv

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level with IRLS and weighted median."""
        qua = copy.copy(self)
        qua.lambda_ = self.lambda_q

        qua.rho_spatial_u = []
        qua.rho_spatial_v = []
        for i in range(len(self.rho_spatial_u)):
            a = self.rho_spatial_u[i].param
            qua.rho_spatial_u.append(RobustFunction('quadratic', a[0]))
            a = self.rho_spatial_v[i].param
            qua.rho_spatial_v.append(RobustFunction('quadratic', a[0]))

        a = self.rho_data.param
        qua.rho_data = RobustFunction('quadratic', a[0])

        for i in range(self.max_iters):
            duv = np.zeros_like(uv)

            It, Ix, Iy = partial_deriv(
                self.images, uv, self.interpolation_method, self.deriv_filter
            )

            for j in range(self.max_linear):
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
                    occ = detect_occlusion(uv, self.images)
                    uv = denoise_color_weighted_medfilt2(
                        uv, self.color_images, occ, self.area_hsz,
                        self.median_filter_size, self.sigma_i, self.fullVersion
                    )

                duv = uv - uv0
                uv = uv0

            uv = uv + duv

        return uv

    def flow_operator(self, uv, duv, It, Ix, Iy):
        """Build the linear system with filter-based spatial term and IRLS weights."""
        sz = (Ix.shape[0], Ix.shape[1])
        npixels = sz[0] * sz[1]

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

        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

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

        lam = self.lambda_
        duu = sparse.diags(pp_d * Ix2_m.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FU
        dvv = sparse.diags(pp_d * Iy2_m.ravel(order='F'), 0, shape=(npixels, npixels)) + lam * FV
        dduv = sparse.diags(pp_d * Ixy_m.ravel(order='F'), 0, shape=(npixels, npixels))

        A = sparse.bmat([[duu, dduv], [dduv, dvv]])

        u_vec = uv[:, :, 0].ravel(order='F')
        v_vec = uv[:, :, 1].ravel(order='F')
        b_vec = np.concatenate([
            -lam * (FU @ u_vec) - pp_d * Itx_m.ravel(order='F'),
            -lam * (FV @ v_vec) - pp_d * Ity_m.ravel(order='F')
        ])

        iterative = True
        if len(pp_su_all) > 0:
            all_uniform = all(
                (pp.max() - pp.min() < 1e-6) for pp in pp_su_all + pp_sv_all
            )
            if all_uniform and (pp_d.max() - pp_d.min() < 1e-6):
                iterative = False

        return A, b_vec, None, iterative
