"""RobustFunction class wrapping penalty functions.

Provides a unified interface for evaluating robust penalty functions,
their derivatives, and IRLS weights (derivative/x).
"""
import numpy as np
from flow_fast.robust.penalties import (
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
    """Wrapper for robust penalty functions with sigma parameters."""

    def __init__(self, method, *args):
        if method not in PENALTY_MAP:
            raise ValueError(
                f"Unknown penalty method '{method}'. "
                f"Available: {list(PENALTY_MAP.keys())}"
            )
        self.method = method
        self._func = PENALTY_MAP[method]

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
        return self.sigma

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 0)

    def deriv(self, x):
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 1)

    def deriv_over_x(self, x):
        x = np.asarray(x, dtype=float)
        return self._func(x, self.sigma, 2)

    def evaluate_log(self, x):
        return self.evaluate(x)

    def __repr__(self):
        return f"RobustFunction('{self.method}', sigma={self.sigma})"
