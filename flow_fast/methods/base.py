"""
Abstract base class for optical flow estimation methods.

Uses accelerated solver dispatch instead of raw spsolve.
"""
import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse

from flow_fast.robust.robust_function import RobustFunction
from flow_fast.utils.image_processing import (
    scale_image, fspecial_gaussian, structure_texture_decomposition_rof
)
from flow_fast.utils.pyramid import compute_image_pyramid
from flow_fast.utils.warping import resample_flow
from flow_fast.utils.derivatives import partial_deriv
from flow_fast.solvers.dispatch import get_solver


class BaseOpticalFlow(ABC):
    """Base class for variational optical flow estimation."""

    def __init__(self):
        self.images = None
        self.lambda_ = 1.0
        self.lambda_q = 1.0
        self.solver = 'auto'
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

        # Cache for sparse convolution matrices
        self._cached_conv_mats = {}

        # Solver backend (lazy init)
        self._solver_backend = None

    def _get_solver_backend(self):
        """Get or create the solver backend."""
        if self._solver_backend is None:
            self._solver_backend = get_solver(self.solver)
        return self._solver_backend

    def parse_input_parameter(self, params):
        """Set parameters from a dictionary or list of key-value pairs."""
        if isinstance(params, dict):
            for key, val in params.items():
                attr = 'lambda_' if key == 'lambda' else key
                if hasattr(self, attr):
                    setattr(self, attr, val)
        elif isinstance(params, (list, tuple)):
            i = 0
            while i < len(params) - 1:
                key = params[i]
                val = params[i + 1]
                attr = 'lambda_' if key == 'lambda' else key
                if hasattr(self, attr):
                    setattr(self, attr, val)
                i += 2

    def _solve_linear_system(self, A, b, uv_shape, x0=None):
        """Solve sparse linear system A @ x = b using the best available solver."""
        solver_name = self.solver.lower()

        # Legacy solver names map to dispatch
        if solver_name == 'backslash':
            solver_name = 'auto'
        elif solver_name == 'pcg':
            solver_name = 'pcg'

        if solver_name == 'sor':
            # Keep legacy SOR for compatibility
            x = self._sor_solve(A, b, 1.9, self.sor_max_iters, 1e-2)
            return x.reshape(uv_shape, order='F')

        backend = get_solver(solver_name)
        return backend.solve(A, b, uv_shape, x0=x0)

    def _sor_solve(self, A, b, omega=1.9, max_iters=10000, tol=1e-2):
        """Successive Over-Relaxation solver (legacy fallback)."""
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

            if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x):
                break

        return x

    def _build_pyramid(self, images, levels, spacing):
        """Build Gaussian image pyramid."""
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
        """Get a cached sparse convolution matrix."""
        from flow_fast.utils.sparse_ops import make_convn_mat
        cache_key = (F.tobytes(), F.shape, sz, shape, pad)
        if cache_key not in self._cached_conv_mats:
            self._cached_conv_mats[cache_key] = make_convn_mat(F, sz, shape, pad)
        return self._cached_conv_mats[cache_key]

    def _get_cached_convmat_csr(self, F, sz, shape='full', pad=None):
        """Get cached convolution matrix with precomputed CSR row-weight info."""
        cache_key = ('csr', F.tobytes(), F.shape, sz, shape, pad)
        if cache_key not in self._cached_conv_mats:
            FMi = self._get_cached_convmat(F, sz, shape, pad)
            FMi_csr = FMi.tocsr()
            nnz_per_row = np.diff(FMi_csr.indptr)
            self._cached_conv_mats[cache_key] = (FMi, FMi_csr, nnz_per_row)
        return self._cached_conv_mats[cache_key]

    @staticmethod
    def _weighted_filter_product(FMi, FMi_csr, nnz_per_row, w):
        """Compute FMi.T @ diag(w) @ FMi efficiently via CSR row-scaling."""
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
        pass

    @abstractmethod
    def compute_flow_base(self, uv):
        pass

    @abstractmethod
    def flow_operator(self, uv, duv, It, Ix, Iy):
        pass
