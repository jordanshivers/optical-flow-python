"""Read and write .flo optical flow files (Middlebury format)."""
import numpy as np
import os

TAG_FLOAT = 202021.25


def read_flo(filename):
    """Read a .flo optical flow file."""
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        if tag != TAG_FLOAT:
            raise ValueError(
                f'Invalid .flo file tag: {tag} (expected {TAG_FLOAT})'
            )
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32)
    return data.reshape((h, w, 2))


def write_flo(flow, filename):
    """Write a .flo optical flow file."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Flow must be (H, W, 2) array, got shape {flow.shape}")
    h, w = flow.shape[:2]
    with open(filename, 'wb') as f:
        np.array([TAG_FLOAT], dtype=np.float32).tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)
        flow.tofile(f)


def read_flow_file(seq_name, i_seq, data_dir=None):
    """Load images and ground truth flow for a Middlebury sequence."""
    from PIL import Image

    if data_dir is None:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_dir = os.path.join(pkg_dir, 'data')

    img_dir = os.path.join(data_dir, 'other-data', seq_name)
    im1_path = os.path.join(img_dir, f'frame{i_seq:02d}.png')
    im2_path = os.path.join(img_dir, f'frame{i_seq + 1:02d}.png')

    im1 = np.array(Image.open(im1_path)).astype(np.float64)
    im2 = np.array(Image.open(im2_path)).astype(np.float64)

    gt_dir = os.path.join(data_dir, 'other-gt-flow', seq_name)
    gt_path = os.path.join(gt_dir, f'flow{i_seq:02d}.flo')

    if os.path.exists(gt_path):
        gt = read_flo(gt_path)
        tu = gt[:, :, 0]
        tv = gt[:, :, 1]
    else:
        tu = None
        tv = None

    return im1, im2, tu, tv
