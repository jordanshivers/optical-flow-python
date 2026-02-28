"""
CHOLMOD-based direct solver for sparse symmetric positive-definite systems.

Wraps scikit-sparse's CHOLMOD factorization with caching of the symbolic
(fill-reducing ordering) phase so that repeated solves on the same sparsity
pattern only perform a cheaper numeric refactorisation.

If scikit-sparse is not installed the module still imports cleanly;
`CHOLMODSolver.available` will be ``False`` and attempts to solve will
raise an informative error.
"""

import logging
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe for scikit-sparse at import time
# ---------------------------------------------------------------------------
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky  # type: ignore[import-untyped]
    _HAS_CHOLMOD = True
except ImportError:
    _HAS_CHOLMOD = False
    logger.debug(
        "scikit-sparse not found; CHOLMODSolver will not be available. "
        "Install with: pip install scikit-sparse"
    )


class CHOLMODSolver:
    """Solve *A x = b* where *A* is sparse SPD via CHOLMOD.

    The solver caches the symbolic factorization (fill-reducing ordering)
    between calls that share the same sparsity pattern.  When only the
    numerical values of *A* change the much cheaper ``cholesky_inplace``
    path is used.  If the sparsity pattern changes a fresh factorization
    is computed automatically.

    Parameters
    ----------
    None

    Attributes
    ----------
    available : bool (class-level)
        ``True`` when scikit-sparse is installed and CHOLMOD can be used.

    Examples
    --------
    >>> solver = CHOLMODSolver()
    >>> x = solver.solve(A, b, uv_shape=(rows, cols, 2))
    """

    available: bool = _HAS_CHOLMOD

    def __init__(self) -> None:
        self._factor = None
        self._pattern_nnz: int | None = None
        self._pattern_shape: tuple | None = None
        self._pattern_indices_hash: int | None = None

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def solve(
        self,
        A: sparse.spmatrix,
        b: np.ndarray,
        uv_shape: tuple,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the sparse SPD system *A x = b*.

        Parameters
        ----------
        A : scipy.sparse matrix (N x N, SPD)
            The system matrix.  Will be converted to CSC internally if
            needed.
        b : ndarray, shape (N,)
            Right-hand side vector.
        uv_shape : tuple
            Desired shape for the output, typically ``(rows, cols, 2)`` for
            an optical-flow field.  The solution is reshaped using Fortran
            (column-major) ordering.
        x0 : ndarray or None, optional
            Ignored.  Present only so the interface matches other solvers.

        Returns
        -------
        x : ndarray, shape *uv_shape*
            The solution reshaped to the requested shape.

        Raises
        ------
        RuntimeError
            If scikit-sparse is not installed.
        """
        if not _HAS_CHOLMOD:
            raise RuntimeError(
                "CHOLMODSolver requires scikit-sparse.  "
                "Install it with: pip install scikit-sparse"
            )

        # CHOLMOD requires CSC format
        if not sparse.issparse(A):
            A = sparse.csc_matrix(A)
        elif not sparse.isspmatrix_csc(A):
            A = A.tocsc()

        b = np.asarray(b, dtype=np.float64).ravel()

        # Determine whether the sparsity pattern is unchanged
        pattern_hash = _sparsity_hash(A)
        same_pattern = (
            self._factor is not None
            and self._pattern_nnz == A.nnz
            and self._pattern_shape == A.shape
            and self._pattern_indices_hash == pattern_hash
        )

        if same_pattern:
            # Re-use the symbolic factorization; only update numerics
            try:
                self._factor.cholesky_inplace(A)
                logger.debug("CHOLMOD: numeric refactorisation (cached ordering)")
            except Exception:
                # Fall back to full factorization if inplace update fails
                logger.debug("CHOLMOD: inplace update failed; full refactorisation")
                self._factor = cholmod_cholesky(A)
        else:
            # First call or sparsity pattern changed – full factorization
            logger.debug("CHOLMOD: full symbolic + numeric factorisation")
            self._factor = cholmod_cholesky(A)
            self._pattern_nnz = A.nnz
            self._pattern_shape = A.shape
            self._pattern_indices_hash = pattern_hash

        x = self._factor(b)
        return x.reshape(uv_shape, order='F')

    def reset(self) -> None:
        """Discard any cached factorization."""
        self._factor = None
        self._pattern_nnz = None
        self._pattern_shape = None
        self._pattern_indices_hash = None

    # --------------------------------------------------------------------- #
    #  Dunder helpers                                                        #
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        status = "ready" if self._factor is not None else "no cached factor"
        return f"CHOLMODSolver(available={self.available}, {status})"


# --------------------------------------------------------------------------- #
#  Private helpers                                                             #
# --------------------------------------------------------------------------- #

def _sparsity_hash(A: sparse.csc_matrix) -> int:
    """Cheap hash of the sparsity pattern (indices + indptr) of a CSC matrix."""
    return hash((A.indices.data.tobytes(), A.indptr.data.tobytes()))
