"""
Horn-Schunck optical flow estimation.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter

from flow_fast.methods.base import BaseOpticalFlow
from flow_fast.robust.robust_function import RobustFunction
from flow_fast.utils.image_processing import (
    scale_image, structure_texture_decomposition_rof
)
from flow_fast.utils.derivatives import partial_deriv
from flow_fast.utils.sparse_ops import make_imfilter_mat


class HSOpticalFlow(BaseOpticalFlow):
    """Horn-Schunck optical flow with quadratic penalty and Laplacian spatial term."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 80
        self.lambda_q = 80
        self.gnc_iters = 1
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0
        self.max_warping_iters = 10
        self.solver = 'auto'
        self.interpolation_method = 'cubic'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.texture = False
        self.limit_update = True
        self.display = False
        self.sor_max_iters = 10000
        self.sigmaD2 = 1.0
        self.sigmaS2 = 1.0
        self.mf_iter = 1
        self.color_images = None

        method = 'quadratic'
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        self.rho_spatial_u = [RobustFunction(method, 1), RobustFunction(method, 1)]
        self.rho_spatial_v = [RobustFunction(method, 1), RobustFunction(method, 1)]
        self.rho_data = RobustFunction(method, 1)

    def compute_flow(self, init=None, gt=None):
        """Compute flow field using coarse-to-fine pyramid."""
        sz = (self.images.shape[0], self.images.shape[1])

        if init is None:
            uv = np.zeros((*sz, 2))
        else:
            uv = init.copy()

        if self.texture:
            images = structure_texture_decomposition_rof(self.images)
        else:
            images = scale_image(self.images, 0, 255)

        self.pyramid_levels = self._auto_pyramid_levels(images)
        pyramid_images = self._build_pyramid(images, self.pyramid_levels, self.pyramid_spacing)

        for l in range(self.pyramid_levels - 1, -1, -1):
            if self.display:
                print(f"Pyramid level: {l + 1}")

            small = self._copy_with_images(pyramid_images[l])
            nsz = (pyramid_images[l].shape[0], pyramid_images[l].shape[1])
            from flow_fast.utils.warping import resample_flow
            uv = resample_flow(uv, nsz)
            uv = small.compute_flow_base(uv)

        if self.median_filter_size is not None:
            uv[:, :, 0] = median_filter(uv[:, :, 0], size=self.median_filter_size, mode='reflect')
            uv[:, :, 1] = median_filter(uv[:, :, 1], size=self.median_filter_size, mode='reflect')

        return uv

    def _copy_with_images(self, images):
        import copy
        small = copy.copy(self)
        small.images = images
        small.pyramid_levels = 1
        return small

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level."""
        for i in range(self.max_warping_iters):
            A, b, params, iterative = self.flow_operator(uv)
            x = self._solve_linear_system(A, b, uv.shape)

            if self.display:
                print(f"  Iteration: {i + 1}  (norm: {np.linalg.norm(x):.6f})")

            if np.linalg.norm(x) < 1e-3:
                break

            if self.limit_update:
                x = np.clip(x, -1, 1)

            uv = uv + x

            if self.median_filter_size is not None:
                for _ in range(self.mf_iter):
                    uv[:, :, 0] = median_filter(uv[:, :, 0], size=self.median_filter_size, mode='reflect')
                    uv[:, :, 1] = median_filter(uv[:, :, 1], size=self.median_filter_size, mode='reflect')

        return uv

    def flow_operator(self, uv, duv=None, It=None, Ix=None, Iy=None):
        """Build the linear system A @ x = b for Horn-Schunck."""
        It_c, Ix_c, Iy_c = partial_deriv(
            self.images, uv, self.interpolation_method, self.deriv_filter
        )

        sz = (Ix_c.shape[0], Ix_c.shape[1])
        npixels = sz[0] * sz[1]

        L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
        F = make_imfilter_mat(L, sz, boundary='replicate', shape='same')

        Z = sparse.csc_matrix((npixels, npixels))
        M = sparse.bmat([[F, Z], [Z, F]])

        if Ix_c.ndim == 3:
            Ix2 = np.mean(Ix_c**2, axis=2)
            Iy2 = np.mean(Iy_c**2, axis=2)
            Ixy = np.mean(Ix_c * Iy_c, axis=2)
            Itx = np.mean(It_c * Ix_c, axis=2)
            Ity = np.mean(It_c * Iy_c, axis=2)
        else:
            Ix2 = Ix_c**2
            Iy2 = Iy_c**2
            Ixy = Ix_c * Iy_c
            Itx = It_c * Ix_c
            Ity = It_c * Iy_c

        duu = sparse.diags(Ix2.ravel(order='F'), 0, shape=(npixels, npixels))
        dvv = sparse.diags(Iy2.ravel(order='F'), 0, shape=(npixels, npixels))
        duv = sparse.diags(Ixy.ravel(order='F'), 0, shape=(npixels, npixels))

        A = sparse.bmat([[duu, duv], [duv, dvv]]) / self.sigmaD2 \
            - self.lambda_ * M / self.sigmaS2

        uv_vec = uv.ravel(order='F')
        b = (self.lambda_ * M @ uv_vec / self.sigmaS2
             - np.concatenate([Itx.ravel(order='F'), Ity.ravel(order='F')]) / self.sigmaD2)

        return A, b, None, True
