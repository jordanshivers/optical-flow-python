"""Optical flow estimation methods."""
from optical_flow.methods.hs import HSOpticalFlow
from optical_flow.methods.ba import BAOpticalFlow
from optical_flow.methods.classic_nl import ClassicNLOpticalFlow
from optical_flow.methods.alt_ba import AltBAOpticalFlow
from optical_flow.methods.config import load_of_method

__all__ = [
    'HSOpticalFlow',
    'BAOpticalFlow',
    'ClassicNLOpticalFlow',
    'AltBAOpticalFlow',
    'load_of_method',
]
