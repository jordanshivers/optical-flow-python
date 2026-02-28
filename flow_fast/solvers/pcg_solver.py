"""
Preconditioned Conjugate Gradient solver for sparse SPD systems.

Uses ``scipy.sparse.linalg.cg`` with a diagonal (Jacobi)
preconditioner for fast convergence on optical-flow systems.

The Jacobi preconditioner is essentially free to build and converges
reliably on the saddle-point / IRLS systems arising in variational
optical-flow estimation, unlike ILU which often fails to converge and
has expensive factorization overhead.

This module does **not** require any optional dependencies beyond SciPy.
"""

import logging
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator

logger = logging.getLogger(__name__)


def pcg_solve(
    A: sparse.spmatrix,
    b: np.ndarray,
    uv_shape: tuple,
    x0: np.ndarray | None = None,
    rtol: float = 1e-3,
    maxiter: int = 200,
) -> np.ndarray:
    """Solve *A x = b* using Preconditioned Conjugate Gradient.

    Parameters
    ----------
    A : scipy.sparse matrix (N x N)
        The system matrix.
    b : ndarray, shape (N,)
        Right-hand side vector.
    uv_shape : tuple
        Desired output shape (e.g. ``(rows, cols, 2)``).  The solution is
        reshaped with Fortran (column-major) ordering.
    x0 : ndarray or None, optional
        Initial guess.  If ``None`` the zero vector is used.
    rtol : float, optional
        Relative tolerance for convergence (default ``1e-3``).
    maxiter : int, optional
        Maximum number of CG iterations (default ``200``).

    Returns
    -------
    x : ndarray, shape *uv_shape*
        The solution reshaped to the requested shape.
    """
    # Ensure CSC format (efficient for cg)
    if not sparse.issparse(A):
        A = sparse.csc_matrix(A)
    elif not sparse.isspmatrix_csc(A):
        A = A.tocsc()

    b = np.asarray(b, dtype=np.float64).ravel()
    n = A.shape[0]

    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        if x0.shape[0] != n:
            logger.warning(
                "x0 length %d does not match system size %d; ignoring x0",
                x0.shape[0], n,
            )
            x0 = None

    # ------------------------------------------------------------------
    # Build Jacobi (diagonal) preconditioner — fast and reliable
    # ------------------------------------------------------------------
    diag = A.diagonal().copy()
    diag[np.abs(diag) < 1e-15] = 1.0  # avoid division by zero
    inv_diag = 1.0 / diag
    M = LinearOperator(
        shape=(n, n),
        matvec=lambda v: inv_diag * v,
        dtype=A.dtype,
    )

    # ------------------------------------------------------------------
    # Solve with CG
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sparse.SparseEfficiencyWarning)
        x, info = cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter, M=M)

    if info > 0:
        logger.debug(
            "PCG did not converge within %d iterations (info=%d)", maxiter, info
        )
    elif info < 0:
        logger.warning("PCG encountered an error (info=%d)", info)

    return x.reshape(uv_shape, order='F')


class PCGSolver:
    """Object-oriented wrapper around :func:`pcg_solve`.

    Provides the same ``.solve()`` interface as :class:`CHOLMODSolver`
    and is usable as a drop-in replacement when CHOLMOD is unavailable.

    Parameters
    ----------
    rtol : float
        Relative convergence tolerance (default ``1e-3``).
    maxiter : int
        Maximum CG iterations (default ``200``).
    """

    available: bool = True  # always available (only needs SciPy)

    def __init__(self, rtol: float = 1e-3, maxiter: int = 200) -> None:
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(
        self,
        A: sparse.spmatrix,
        b: np.ndarray,
        uv_shape: tuple,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve *A x = b* via Preconditioned Conjugate Gradient.

        See :func:`pcg_solve` for full documentation.
        """
        return pcg_solve(
            A, b, uv_shape, x0=x0, rtol=self.rtol, maxiter=self.maxiter
        )

    def reset(self) -> None:
        """No-op (PCG is stateless)."""

    def __repr__(self) -> str:
        return f"PCGSolver(rtol={self.rtol}, maxiter={self.maxiter})"
