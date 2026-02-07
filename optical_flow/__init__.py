"""
Optical Flow Estimation Package

Python reimplementation of the "Secrets of Optical Flow Estimation and Their Principles"
by Sun, Roth & Black (CVPR 2010).

Implements Horn-Schunck, Black-Anandan, Classic+NL, and Alternative BA optical flow methods.
"""

from optical_flow.interface import estimate_flow
from optical_flow.io.flo_io import read_flo, write_flo
from optical_flow.viz.flow_color import flow_to_color
from optical_flow.viz.plot_flow import plot_flow
from optical_flow.evaluation.metrics import flow_angular_error
from optical_flow.methods.config import load_of_method

__all__ = [
    'estimate_flow',
    'read_flo',
    'write_flo',
    'flow_to_color',
    'plot_flow',
    'flow_angular_error',
    'load_of_method',
]
