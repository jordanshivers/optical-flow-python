"""Read and write .flo optical flow files (Middlebury format).

The .flo file format stores optical flow as:
    - 4 bytes: magic number 202021.25 (float32)
    - 4 bytes: width (int32)
    - 4 bytes: height (int32)
    - width * height * 2 * 4 bytes: flow data (float32, interleaved u,v)

Also provides a convenience function to load image pairs and ground truth
flow for Middlebury benchmark sequences.
"""
import numpy as np
import os

TAG_FLOAT = 202021.25


def read_flo(filename):
    """Read a .flo optical flow file.

    Args:
        filename: Path to .flo file.

    Returns:
        flow: (H, W, 2) float32 array with (u, v) flow components.

    Raises:
        ValueError: If file magic number doesn't match.
        FileNotFoundError: If file does not exist.
    """
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        if tag != TAG_FLOAT:
            raise ValueError(
                f'Invalid .flo file tag: {tag} (expected {TAG_FLOAT})'
            )
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32)

    # Data is stored row-major, interleaved: u0,v0,u1,v1,...
    flow = data.reshape((h, w, 2))
    return flow


def write_flo(flow, filename):
    """Write a .flo optical flow file.

    Args:
        flow: (H, W, 2) array with (u, v) flow components.
        filename: Output path.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"Flow must be (H, W, 2) array, got shape {flow.shape}"
        )

    h, w = flow.shape[:2]
    with open(filename, 'wb') as f:
        np.array([TAG_FLOAT], dtype=np.float32).tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)
        flow.tofile(f)


def read_flow_file(seq_name, i_seq, data_dir=None):
    """Load images and ground truth flow for a Middlebury sequence.

    Expects the standard Middlebury directory layout:
        data_dir/other-data/{seq_name}/frame{i_seq:02d}.png
        data_dir/other-gt-flow/{seq_name}/flow{i_seq:02d}.flo

    Args:
        seq_name: Sequence name (e.g. 'RubberWhale').
        i_seq: Frame index (e.g. 10 for frame10.png / frame11.png).
        data_dir: Base data directory. If None, defaults to the
            package's data/ directory (two levels up from this file,
            then into the sibling data/ folder).

    Returns:
        im1: First image as float64 array (H, W) or (H, W, 3).
        im2: Second image as float64 array (H, W) or (H, W, 3).
        tu: Ground truth horizontal flow (H, W) float32, or None.
        tv: Ground truth vertical flow (H, W) float32, or None.
    """
    from PIL import Image

    if data_dir is None:
        # Default: go up from optical_flow/io/ to flow_code_python/data/
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_dir = os.path.join(pkg_dir, 'data')

    # Load images
    img_dir = os.path.join(data_dir, 'other-data', seq_name)
    im1_path = os.path.join(img_dir, f'frame{i_seq:02d}.png')
    im2_path = os.path.join(img_dir, f'frame{i_seq + 1:02d}.png')

    im1 = np.array(Image.open(im1_path)).astype(np.float64)
    im2 = np.array(Image.open(im2_path)).astype(np.float64)

    # Load ground truth flow (if available)
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
