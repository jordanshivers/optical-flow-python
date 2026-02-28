"""
Solver dispatcher -- automatically selects the best available sparse
linear solver for optical-flow estimation.

Usage
-----
>>> from flow_fast.solvers.dispatch import get_solver
>>> solver = get_solver()           # auto-detect best backend
>>> x = solver.solve(A, b, uv_shape)

>>> solver = get_solver('cholmod')  # force a specific backend
>>> solver = get_solver('pcg')
>>> solver = get_solver('backslash')
"""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from flow_fast.solvers.cholmod_solver import CHOLMODSolver
from flow_fast.solvers.pcg_solver import PCGSolver

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  "Backslash" (spsolve) wrapper                                              #
# --------------------------------------------------------------------------- #

class SpSolveSolver:
    """Thin wrapper around :func:`scipy.sparse.linalg.spsolve`.

    This is the simplest available solver and requires no optional
    dependencies.  It uses SuperLU under the hood and is a reasonable
    fallback when neither CHOLMOD nor a good ILU preconditioner are
    available.
    """

    available: bool = True

    def solve(
        self,
        A: sparse.spmatrix,
        b: np.ndarray,
        uv_shape: tuple,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve *A x = b* via ``scipy.sparse.linalg.spsolve``.

        Parameters
        ----------
        A : scipy.sparse matrix (N x N)
            System matrix.
        b : ndarray, shape (N,)
            Right-hand side.
        uv_shape : tuple
            Desired output shape; reshaped with Fortran ordering.
        x0 : ndarray or None
            Ignored (present for interface compatibility).

        Returns
        -------
        x : ndarray, shape *uv_shape*
        """
        if not sparse.issparse(A):
            A = sparse.csc_matrix(A)
        elif not sparse.isspmatrix_csc(A):
            A = A.tocsc()

        b = np.asarray(b, dtype=np.float64).ravel()
        x = spsolve(A, b)
        return x.reshape(uv_shape, order='F')

    def reset(self) -> None:
        """No-op (spsolve is stateless)."""

    def __repr__(self) -> str:
        return "SpSolveSolver()"


# --------------------------------------------------------------------------- #
#  Dispatcher                                                                  #
# --------------------------------------------------------------------------- #

_VALID_PREFERENCES = ('auto', 'cholmod', 'pcg', 'backslash')


def get_solver(preference: str = 'auto'):
    """Return the best available solver.

    Parameters
    ----------
    preference : str
        One of ``'auto'``, ``'cholmod'``, ``'pcg'``, ``'backslash'``.

        ``'auto'`` (the default) probes backends in order of expected
        performance and returns the first that is available:

        1. **CHOLMOD** -- direct, supernodal Cholesky (requires
           scikit-sparse).
        2. **PCG with ILU** -- iterative CG preconditioned with
           incomplete LU (pure SciPy, always available).
        3. **spsolve** -- SuperLU-based direct solve (pure SciPy,
           always available).

        When a specific backend is requested it is returned if available
        or a :class:`RuntimeError` is raised.

    Returns
    -------
    solver : object
        An object with a ``.solve(A, b, uv_shape, x0=None)`` method.

    Raises
    ------
    ValueError
        If *preference* is not recognised.
    RuntimeError
        If the requested backend is not available.
    """
    preference = preference.lower().strip()
    if preference not in _VALID_PREFERENCES:
        raise ValueError(
            f"Unknown solver preference {preference!r}; "
            f"choose from {_VALID_PREFERENCES}"
        )

    if preference == 'cholmod':
        if not CHOLMODSolver.available:
            raise RuntimeError(
                "CHOLMOD solver requested but scikit-sparse is not installed. "
                "Install with: pip install scikit-sparse"
            )
        logger.info("Solver: CHOLMOD (requested)")
        return CHOLMODSolver()

    if preference == 'pcg':
        logger.info("Solver: PCG (requested)")
        return PCGSolver()

    if preference == 'backslash':
        logger.info("Solver: spsolve / backslash (requested)")
        return SpSolveSolver()

    # --- auto mode ---
    if CHOLMODSolver.available:
        logger.info("Solver: CHOLMOD (auto-detected)")
        return CHOLMODSolver()

    logger.info("Solver: PCG (auto; scikit-sparse not available)")
    return PCGSolver()
