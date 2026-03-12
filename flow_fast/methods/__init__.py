"""Optical flow estimation methods."""
from flow_fast.methods.hs import HSOpticalFlow
from flow_fast.methods.ba import BAOpticalFlow
from flow_fast.methods.classic_nl import ClassicNLOpticalFlow
from flow_fast.methods.alt_ba import AltBAOpticalFlow
from flow_fast.methods.lk import LKOpticalFlow
from flow_fast.methods.config import load_of_method

__all__ = [
    'HSOpticalFlow',
    'BAOpticalFlow',
    'ClassicNLOpticalFlow',
    'AltBAOpticalFlow',
    'LKOpticalFlow',
    'load_of_method',
]
