"""
Abstract base class for optical flow estimation methods.
"""
import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, LinearOperator

from optical_flow.robust.robust_function import RobustFunction
from optical_flow.utils.image_processing import (
    scale_image, fspecial_gaussian, structure_texture_decomposition_rof
)
from optical_flow.utils.pyramid import compute_image_pyramid
from optical_flow.utils.warping import resample_flow
from optical_flow.utils.derivatives import partial_deriv


class BaseOpticalFlow(ABC):
    """Base class for variational optical flow estimation."""

    def __init__(self):
        self.images = None
        self.lambda_ = 1.0
        self.lambda_q = 1.0
        self.solver = 'backslash'
        self.pcg_rtol = 1e-3
        self.pcg_maxiter = 200
        self.sor_max_iters = 10000
        self.interpolation_method = 'cubic'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.blend = 0.5
        self.texture = False
        self.fc = False
        self.median_filter_size = None
        self.limit_update = True
        self.display = False
        self.color_images = None
        self.auto_level = True
        self.alp = 0.95

        # Pyramid settings
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0

        # GNC settings
        self.gnc_iters = 1
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25
        self.alpha = 1.0

        # Iteration settings
        self.max_iters = 10
        self.max_linear = 1

        # Spatial filters and robust functions
        self.spatial_filters = [np.array([[1, -1]]), np.array([[1], [-1]])]
        method = 'quadratic'
        self.rho_spatial_u = [RobustFunction(method, 1), RobustFunction(method, 1)]
        self.rho_spatial_v = [RobustFunction(method, 1), RobustFunction(method, 1)]
        self.rho_data = RobustFunction(method, 1)

        # Cache for sparse convolution matrices (avoids re-building per IRLS iteration)
        self._cached_conv_mats = {}

    def parse_input_parameter(self, params):
        """Set parameters from a dictionary or list of key-value pairs.

        Args:
            params: dict or list of [key, value, key, value, ...].
        """
        if isinstance(params, dict):
            for key, val in params.items():
                attr = 'lambda_' if key == 'lambda' else key
                if hasattr(self, attr):
                    setattr(self, attr, val)
        elif isinstance(params, (list, tuple)):
            # Handle MATLAB-style {key, val, key, val, ...} format
            i = 0
            while i < len(params) - 1:
                key = params[i]
                val = params[i + 1]
                attr = 'lambda_' if key == 'lambda' else key
                if hasattr(self, attr):
                    setattr(self, attr, val)
                i += 2

    def _solve_linear_system(self, A, b, uv_shape, x0=None):
        """Solve sparse linear system A @ x = b.

        Solvers:
            'pcg': Diagonal-preconditioned conjugate gradient (default, fast).
            'backslash': Direct solve via spsolve (exact but slow for large systems).
            'sor': Successive Over-Relaxation (legacy).

        Args:
            A: Sparse matrix (should be SPD for pcg).
            b: Right-hand side vector.
            uv_shape: Shape to reshape solution into.
            x0: Optional initial guess (flat vector) for iterative solvers.

        Returns:
            x: Solution reshaped to uv_shape.
        """
        solver = self.solver.lower()
        if solver == 'pcg':
            x = self._pcg_solve(A, b, x0=x0)
        elif solver == 'backslash':
            x = spsolve(A.tocsc(), b)
        elif solver == 'sor':
            x = self._sor_solve(A, b, 1.9, self.sor_max_iters, 1e-2)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return x.reshape(uv_shape, order='F')

    def _pcg_solve(self, A, b, x0=None):
        """Diagonal-preconditioned conjugate gradient solver.

        Uses the diagonal of A as a preconditioner, which is cheap to compute
        and effective for the diagonally-dominant systems in optical flow.

        Args:
            A: Sparse SPD matrix.
            b: Right-hand side vector.
            x0: Optional initial guess for warm-starting CG.

        Returns:
            x: Solution vector.
        """
        A_csc = A.tocsc()
        diag = A_csc.diagonal()
        diag_inv = np.where(np.abs(diag) > 1e-12, 1.0 / diag, 0.0)
        M_prec = LinearOperator(A_csc.shape, matvec=lambda v: diag_inv * v)
        x, info = cg(A_csc, b, x0=x0, M=M_prec, maxiter=self.pcg_maxiter,
                      rtol=self.pcg_rtol)
        return x

    def _sor_solve(self, A, b, omega=1.9, max_iters=10000, tol=1e-2):
        """Successive Over-Relaxation solver.

        Args:
            A: Sparse matrix (CSR format preferred).
            b: Right-hand side.
            omega: Relaxation parameter.
            max_iters: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            x: Solution vector.
        """
        A = A.tocsr()
        n = A.shape[0]
        x = np.zeros(n)
        diag = A.diagonal()

        for iteration in range(max_iters):
            x_old = x.copy()
            for i in range(n):
                if abs(diag[i]) < 1e-15:
                    continue
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                cols = A.indices[row_start:row_end]
                vals = A.data[row_start:row_end]
                sigma = np.dot(vals, x[cols]) - diag[i] * x[i]
                x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / diag[i]

            # Check convergence
            if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x):
                break

        return x

    def _build_pyramid(self, images, levels, spacing):
        """Build Gaussian image pyramid.

        Args:
            images: Input images.
            levels: Number of pyramid levels.
            spacing: Downsampling ratio.

        Returns:
            pyramid: List of images from finest to coarsest.
        """
        factor = np.sqrt(2)
        smooth_sigma = np.sqrt(spacing) / factor
        ksize = 2 * round(1.5 * smooth_sigma) + 1
        f = fspecial_gaussian(int(ksize), smooth_sigma)
        ratio = 1.0 / spacing
        return compute_image_pyramid(images, f, levels, ratio)

    def _auto_pyramid_levels(self, images):
        """Automatically determine pyramid levels based on image size."""
        min_dim = min(images.shape[0], images.shape[1])
        return 1 + int(np.floor(np.log(min_dim / 16.0) / np.log(self.pyramid_spacing)))

    def _get_cached_convmat(self, F, sz, shape='full', pad=None):
        """Get a cached sparse convolution matrix, building it if needed.

        Args:
            F: Filter kernel (2D array).
            sz: Image size (H, W).
            shape: 'full', 'same', 'valid'.
            pad: Optional padding mode.

        Returns:
            M: Sparse convolution matrix.
        """
        from optical_flow.utils.sparse_ops import make_convn_mat
        cache_key = (F.tobytes(), F.shape, sz, shape, pad)
        if cache_key not in self._cached_conv_mats:
            self._cached_conv_mats[cache_key] = make_convn_mat(F, sz, shape, pad)
        return self._cached_conv_mats[cache_key]

    def _get_cached_convmat_csr(self, F, sz, shape='full', pad=None):
        """Get cached convolution matrix with precomputed CSR row-weight info.

        Returns (FMi, FMi_csr, nnz_per_row) for fast weighted triple-product
        FMi.T @ diag(w) @ FMi via CSR row-scaling.

        Args:
            F: Filter kernel (2D array).
            sz: Image size (H, W).
            shape: 'full', 'same', 'valid'.
            pad: Optional padding mode.

        Returns:
            FMi: Sparse convolution matrix (CSC).
            FMi_csr: Same matrix in CSR format.
            nnz_per_row: Number of nonzeros per row (for np.repeat).
        """
        cache_key = ('csr', F.tobytes(), F.shape, sz, shape, pad)
        if cache_key not in self._cached_conv_mats:
            FMi = self._get_cached_convmat(F, sz, shape, pad)
            FMi_csr = FMi.tocsr()
            nnz_per_row = np.diff(FMi_csr.indptr)
            self._cached_conv_mats[cache_key] = (FMi, FMi_csr, nnz_per_row)
        return self._cached_conv_mats[cache_key]

    @staticmethod
    def _weighted_filter_product(FMi, FMi_csr, nnz_per_row, w):
        """Compute FMi.T @ diag(w) @ FMi efficiently via CSR row-scaling.

        Instead of constructing diag(w) and doing two sparse matrix products,
        scales the rows of FMi by w and does a single FMi.T @ wFMi product.

        Args:
            FMi: Sparse convolution matrix (for .T).
            FMi_csr: Same matrix in CSR format.
            nnz_per_row: Precomputed np.diff(FMi_csr.indptr).
            w: Weight vector (length = FMi.shape[0]).

        Returns:
            Result: Sparse matrix FMi.T @ diag(w) @ FMi.
        """
        row_weights = np.repeat(w, nnz_per_row)
        data_scaled = FMi_csr.data * row_weights
        wFMi = sparse.csr_matrix(
            (data_scaled, FMi_csr.indices, FMi_csr.indptr),
            shape=FMi_csr.shape
        )
        return FMi.T @ wFMi

    def clear_conv_cache(self):
        """Clear the cached convolution matrices."""
        self._cached_conv_mats = {}

    @abstractmethod
    def compute_flow(self, init=None, gt=None):
        """Compute optical flow."""
        pass

    @abstractmethod
    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level."""
        pass

    @abstractmethod
    def flow_operator(self, uv, duv, It, Ix, Iy):
        """Build the linear system A @ x = b for flow estimation."""
        pass
