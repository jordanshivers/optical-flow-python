"""Flow visualization utilities."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optical_flow.viz.flow_color import flow_to_color


def plot_flow(uv, style='color', ax=None, max_flow=None, step=1):
    """Plot optical flow field.

    Args:
        uv: (H, W, 2) flow field.
        style: 'color' (Middlebury), 'quiver', 'magnitude', 'hsv'.
        ax: matplotlib axes. If None, creates new figure.
        max_flow: Max flow for normalization.
        step: Step size for quiver plots.

    Returns:
        ax: The matplotlib axes used.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    u = uv[:, :, 0].copy()
    v = uv[:, :, 1].copy()

    # Zero out unknown flow markers (sentinel value ~1e9 in .flo files)
    UNKNOWN_THRESH = 1e9
    unknown = (np.abs(u) > UNKNOWN_THRESH) | (np.abs(v) > UNKNOWN_THRESH)
    u[unknown] = 0
    v[unknown] = 0

    if style == 'color':
        img = flow_to_color(uv, max_flow=max_flow)
        ax.imshow(img)
        ax.set_title('Optical Flow (Color)')
    elif style == 'quiver':
        H, W = u.shape
        Y, X = np.mgrid[0:H:step, 0:W:step]
        ax.quiver(X, Y, u[::step, ::step], v[::step, ::step],
                  angles='xy')
        ax.set_ylim(H, 0)
        ax.set_xlim(0, W)
        ax.set_aspect('equal')
        ax.set_title('Optical Flow (Quiver)')
    elif style == 'magnitude':
        mag = np.sqrt(u ** 2 + v ** 2)
        ax.imshow(mag, cmap='jet')
        ax.set_title('Flow Magnitude')
    elif style == 'hsv':
        H_img, W_img = u.shape
        hsv = np.zeros((H_img, W_img, 3))
        mag = np.sqrt(u ** 2 + v ** 2)
        ang = np.arctan2(v, u)
        hsv[:, :, 0] = (ang + np.pi) / (2 * np.pi)
        hsv[:, :, 1] = 1.0
        max_mag = mag.max() if max_flow is None else max_flow
        hsv[:, :, 2] = np.clip(mag / max(max_mag, 1e-8), 0, 1)
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        ax.imshow(rgb)
        ax.set_title('Optical Flow (HSV)')
    else:
        raise ValueError(f"Unknown style: {style}")

    ax.axis('off')
    return ax
