"""Deep learning optical flow methods.

Provides RAFT, SEA-RAFT, and WAFT wrappers accessible through estimate_flow().
Requires PyTorch (install with: pip install optical_flow[deep]).
"""


def load_deep_method(method):
    """Load a deep learning flow method by name.

    Args:
        method: Method name ('raft', 'sea-raft', 'waft').

    Returns:
        Configured DeepFlowBase instance.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If method name is unknown.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for deep learning flow methods. "
            "Install with: pip install optical_flow[deep]"
        )

    method = method.lower()

    if method == 'raft':
        from optical_flow.methods.deep.raft import RAFTFlow
        return RAFTFlow()
    elif method == 'sea-raft':
        from optical_flow.methods.deep.sea_raft import SEARAFTFlow
        return SEARAFTFlow()
    elif method == 'waft':
        from optical_flow.methods.deep.waft import WAFTFlow
        return WAFTFlow()
    else:
        raise ValueError(
            f"Unknown deep learning method: '{method}'. "
            f"Available: 'raft', 'sea-raft', 'waft'"
        )


from optical_flow.methods.deep.raft import RAFTFlow
from optical_flow.methods.deep.sea_raft import SEARAFTFlow
from optical_flow.methods.deep.waft import WAFTFlow

__all__ = [
    'load_deep_method',
    'RAFTFlow',
    'SEARAFTFlow',
    'WAFTFlow',
]
