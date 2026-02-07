"""Shared fixtures for optical flow tests."""
import os
import numpy as np
import pytest


@pytest.fixture
def data_dir():
    """Path to the test data directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data'
    )


@pytest.fixture
def rubberwhale_images(data_dir):
    """Load RubberWhale test images as float64 arrays."""
    from PIL import Image
    im1_path = os.path.join(data_dir, 'other-data', 'RubberWhale', 'frame10.png')
    im2_path = os.path.join(data_dir, 'other-data', 'RubberWhale', 'frame11.png')
    im1 = np.array(Image.open(im1_path)).astype(np.float64)
    im2 = np.array(Image.open(im2_path)).astype(np.float64)
    return im1, im2


@pytest.fixture
def rubberwhale_gray(rubberwhale_images):
    """RubberWhale images converted to grayscale."""
    im1, im2 = rubberwhale_images
    gray1 = 0.2989 * im1[:,:,0] + 0.5870 * im1[:,:,1] + 0.1140 * im1[:,:,2]
    gray2 = 0.2989 * im2[:,:,0] + 0.5870 * im2[:,:,1] + 0.1140 * im2[:,:,2]
    return gray1, gray2


@pytest.fixture
def rubberwhale_gt(data_dir):
    """Load RubberWhale ground truth flow."""
    from optical_flow.io.flo_io import read_flo
    gt_path = os.path.join(data_dir, 'other-gt-flow', 'RubberWhale', 'flow10.flo')
    gt = read_flo(gt_path)
    return gt[:, :, 0], gt[:, :, 1]


@pytest.fixture
def synthetic_pair():
    """Create a simple synthetic image pair with known horizontal shift."""
    np.random.seed(42)
    H, W = 64, 64
    im1 = np.random.rand(H, W) * 255
    # Shift right by 1 pixel
    im2 = np.zeros_like(im1)
    im2[:, 1:] = im1[:, :-1]
    im2[:, 0] = im1[:, 0]
    return im1, im2
