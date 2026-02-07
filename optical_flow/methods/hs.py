"""
Horn-Schunck optical flow estimation.

B.K.P. Horn and B.G. Schunck. "Determining optical flow."
Artificial Intelligence, 16:185-203, Aug. 1981.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter

from optical_flow.methods.base import BaseOpticalFlow
from optical_flow.robust.robust_function import RobustFunction
from optical_flow.utils.image_processing import (
    scale_image, structure_texture_decomposition_rof
)
from optical_flow.utils.derivatives import partial_deriv
from optical_flow.utils.sparse_ops import make_imfilter_mat


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
        self.solver = 'backslash'
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
        """Compute flow field using coarse-to-fine pyramid.

        Args:
            init: Initial flow (H, W, 2). Default: zeros.
            gt: Optional ground truth for evaluation during computation.

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
            images = structure_texture_decomposition_rof(self.images)
        else:
            images = scale_image(self.images, 0, 255)

        # Auto pyramid levels
        self.pyramid_levels = self._auto_pyramid_levels(images)

        # Build pyramid
        pyramid_images = self._build_pyramid(images, self.pyramid_levels, self.pyramid_spacing)

        # Coarse-to-fine
        for l in range(self.pyramid_levels - 1, -1, -1):
            if self.display:
                print(f"Pyramid level: {l + 1}")

            # Create a copy with current level's images
            small = self._copy_with_images(pyramid_images[l])

            # Rescale flow
            nsz = (pyramid_images[l].shape[0], pyramid_images[l].shape[1])
            from optical_flow.utils.warping import resample_flow
            uv = resample_flow(uv, nsz)

            # Compute flow at this level
            uv = small.compute_flow_base(uv)

        # Final median filter
        if self.median_filter_size is not None:
            uv[:, :, 0] = median_filter(uv[:, :, 0], size=self.median_filter_size, mode='reflect')
            uv[:, :, 1] = median_filter(uv[:, :, 1], size=self.median_filter_size, mode='reflect')

        return uv

    def _copy_with_images(self, images):
        """Create a copy of this object with different images."""
        import copy
        small = copy.copy(self)
        small.images = images
        small.pyramid_levels = 1
        return small

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level.

        Args:
            uv: Current flow estimate (H, W, 2).

        Returns:
            uv: Updated flow estimate.
        """
        for i in range(self.max_warping_iters):
            A, b, params, iterative = self.flow_operator(uv)

            x = self._solve_linear_system(A, b, uv.shape)

            if self.display:
                print(f"  Iteration: {i + 1}  (norm: {np.linalg.norm(x):.6f})")

            # Early termination
            if np.linalg.norm(x) < 1e-3:
                break

            # Limit update
            if self.limit_update:
                x = np.clip(x, -1, 1)

            uv = uv + x

            # Median filter
            if self.median_filter_size is not None:
                for _ in range(self.mf_iter):
                    uv[:, :, 0] = median_filter(uv[:, :, 0], size=self.median_filter_size, mode='reflect')
                    uv[:, :, 1] = median_filter(uv[:, :, 1], size=self.median_filter_size, mode='reflect')

        return uv

    def flow_operator(self, uv, duv=None, It=None, Ix=None, Iy=None):
        """Build the linear system A @ x = b for Horn-Schunck flow estimation.

        Uses Laplacian spatial regularization and quadratic data term.

        Args:
            uv: Current flow estimate (H, W, 2).
            duv, It, Ix, Iy: Unused (provided for interface compatibility).

        Returns:
            A: Sparse system matrix.
            b: Right-hand side vector.
            params: None (no auxiliary params).
            iterative: True.
        """
        # Compute derivatives
        It_c, Ix_c, Iy_c = partial_deriv(
            self.images, uv, self.interpolation_method, self.deriv_filter
        )

        sz = (Ix_c.shape[0], Ix_c.shape[1])
        npixels = sz[0] * sz[1]

        # Laplacian operator for spatial term
        L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
        F = make_imfilter_mat(L, sz, boundary='replicate', shape='same')

        # Block diagonal for u and v
        Z = sparse.csc_matrix((npixels, npixels))
        M = sparse.bmat([[F, Z], [Z, F]])

        # Average across color channels if needed
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

        # Data term matrices (diagonal)
        duu = sparse.diags(Ix2.ravel(order='F'), 0, shape=(npixels, npixels))
        dvv = sparse.diags(Iy2.ravel(order='F'), 0, shape=(npixels, npixels))
        duv = sparse.diags(Ixy.ravel(order='F'), 0, shape=(npixels, npixels))

        # System matrix
        A = sparse.bmat([[duu, duv], [duv, dvv]]) / self.sigmaD2 \
            - self.lambda_ * M / self.sigmaS2

        # Right-hand side
        uv_vec = uv.ravel(order='F')
        b = (self.lambda_ * M @ uv_vec / self.sigmaS2
             - np.concatenate([Itx.ravel(order='F'), Ity.ravel(order='F')]) / self.sigmaD2)

        return A, b, None, True
