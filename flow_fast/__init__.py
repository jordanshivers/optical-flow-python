"""
Flow Fast - High-Performance Optical Flow Estimation

Accelerated reimplementation using Numba JIT, OpenCV, and optimized sparse solvers.
Same API as the optical_flow package but significantly faster.
"""

from flow_fast.interface import estimate_flow
from flow_fast.io.flo_io import read_flo, write_flo
from flow_fast.viz.flow_color import flow_to_color
from flow_fast.viz.plot_flow import plot_flow
from flow_fast.evaluation.metrics import flow_angular_error
from flow_fast.methods.config import load_of_method

__all__ = [
    'estimate_flow',
    'read_flo',
    'write_flo',
    'flow_to_color',
    'plot_flow',
    'flow_angular_error',
    'load_of_method',
    'warmup',
]


def warmup():
    """Pre-compile all Numba JIT functions to avoid first-call latency.

    Call this once at startup if you want to avoid JIT compilation
    overhead on the first estimate_flow() call.
    """
    from flow_fast._accel import warmup as _warmup
    _warmup()
