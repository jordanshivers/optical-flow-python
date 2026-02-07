"""RobustFunction class wrapping penalty functions.

Provides a unified interface for evaluating robust penalty functions,
their derivatives, and IRLS weights (derivative/x).

Reimplementation of MATLAB code from:
    Sun, Roth, Black. "Secrets of Optical Flow Estimation and Their Principles."
    CVPR 2010.
"""
import numpy as np
from optical_flow.robust.penalties import (
    quadratic, lorentzian, charbonnier, generalized_charbonnier,
    geman_mcclure, huber, tukey, gaussian, tdist, tdist_unnorm
)

PENALTY_MAP = {
    'quadratic': quadratic,
    'lorentzian': lorentzian,
    'charbonnier': charbonnier,
    'generalized_charbonnier': generalized_charbonnier,
    'geman_mcclure': geman_mcclure,
    'huber': huber,
    'tukey': tukey,
    'gaussian': gaussian,
    'tdist': tdist,
    'tdist_unnorm': tdist_unnorm,
}


class RobustFunction:
    """Wrapper for robust penalty functions with sigma parameters.

    Provides evaluate(), deriv(), and deriv_over_x() methods that
    delegate to the underlying penalty function with the stored
    parameters.

    Examples:
        >>> rf = RobustFunction('charbonnier', 0.01)
        >>> x = np.linspace(-1, 1, 100)
        >>> values = rf.evaluate(x)
        >>> weights = rf.deriv_over_x(x)

        >>> rf = RobustFunction('generalized_charbonnier', 0.01, 0.45)
        >>> rf = RobustFunction('tdist', 5.0, 0.1)
    """

    def __init__(self, method, *args):
        """Initialize RobustFunction.

        Args:
            method: String name of penalty function. One of:
                'quadratic', 'lorentzian', 'charbonnier',
                'generalized_charbonnier', 'geman_mcclure', 'huber',
                'tukey', 'gaussian', 'tdist', 'tdist_unnorm'.
            *args: Parameters for the penalty function.
                For most functions: just sigma (a single scalar).
                For generalized_charbonnier: sigma, a (scale and exponent).
                For tdist / tdist_unnorm: r, s (degrees of freedom and scale).
        """
        if method not in PENALTY_MAP:
            raise ValueError(
                f"Unknown penalty method '{method}'. "
                f"Available: {list(PENALTY_MAP.keys())}"
            )
        self.method = method
        self._func = PENALTY_MAP[method]

        # Store parameters as a numpy array
        if method == 'generalized_charbonnier':
            if len(args) >= 2:
                self.sigma = np.array([args[0], args[1]], dtype=float)
            else:
                self.sigma = np.atleast_1d(np.asarray(args[0], dtype=float))
        elif method in ('tdist', 'tdist_unnorm'):
            if len(args) >= 2:
                self.sigma = np.array([args[0], args[1]], dtype=float)
            else:
                self.sigma = np.atleast_1d(np.asarray(args[0], dtype=float))
        else:
            if len(args) > 0:
                self.sigma = np.atleast_1d(np.asarray(args[0], dtype=float))
            else:
                self.sigma = np.array([1.0])

    @property
    def param(self):
        """Return the parameter array."""
        return self.sigma

    def evaluate(self, x):
        """Evaluate the penalty function rho(x).

        Args:
            x: Input values (scalar or numpy array).

        Returns:
            y: Penalty values, same shape as x.
        """
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 0)

    def deriv(self, x):
        """Evaluate the first derivative rho'(x).

        Args:
            x: Input values (scalar or numpy array).

        Returns:
            y: Derivative values, same shape as x.
        """
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 1)

    def deriv_over_x(self, x):
        """Evaluate rho'(x)/x, the IRLS weight.

        This is the weight used in iteratively reweighted least squares.
        For penalty functions where rho'(x)/x is well-defined at x=0,
        the limit value is returned.

        Args:
            x: Input values (scalar or numpy array).

        Returns:
            y: IRLS weight values, same shape as x.
        """
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 2)

    def evaluate_log(self, x):
        """Evaluate log of the penalty (same as evaluate for most penalties).

        For penalties that are already in log form (like gaussian, tdist),
        this is the same as evaluate().

        Args:
            x: Input values (scalar or numpy array).

        Returns:
            y: Log-penalty values, same shape as x.
        """
        return self.evaluate(x)

    def __repr__(self):
        return f"RobustFunction('{self.method}', sigma={self.sigma})"
