"""
Lucas-Kanade optical flow estimation.

B.D. Lucas and T. Kanade. "An iterative image registration technique
with an application to stereo vision." IJCAI, 1981.

A local, window-based method that solves a 2x2 linear system per pixel
using a Gaussian-weighted neighborhood (structure tensor approach).
"""
import copy

import numpy as np
from scipy.ndimage import correlate, median_filter

from optical_flow.methods.base import BaseOpticalFlow
from optical_flow.utils.image_processing import (
    fspecial_gaussian, scale_image, structure_texture_decomposition_rof
)
from optical_flow.utils.derivatives import partial_deriv
from optical_flow.utils.warping import resample_flow


class LKOpticalFlow(BaseOpticalFlow):
    """Lucas-Kanade optical flow with Gaussian-weighted structure tensor.

    Unlike the global variational methods (HS, BA, Classic+NL), Lucas-Kanade
    is a local method: it estimates flow at each pixel by solving a weighted
    least-squares problem over a local window. This makes it fast but less
    accurate in regions with low texture (the aperture problem).

    Uses a coarse-to-fine pyramid with iterative warping refinement.
    """

    def __init__(self):
        super().__init__()
        # LK-specific parameters
        self.window_size = 15
        self.sigma = None  # Gaussian sigma; None = window_size / 4
        self.eigen_threshold = 1e-4  # min eigenvalue for reliable flow

        # Pyramid and iteration settings
        self.pyramid_levels = 4
        self.pyramid_spacing = 2.0
        self.max_warping_iters = 10
        self.interpolation_method = 'cubic'
        self.deriv_filter = np.array([1, -8, 0, 8, -1]) / 12.0
        self.texture = False
        self.limit_update = True
        self.display = False
        self.color_images = None
        self.mf_iter = 1

        # Reliability map (minimum eigenvalue of structure tensor at finest level)
        self.reliability = None

    def compute_flow(self, init=None, gt=None):
        """Compute flow field using coarse-to-fine pyramid.

        Args:
            init: Initial flow (H, W, 2). Default: zeros.
            gt: Optional ground truth (unused).

        Returns:
            uv: Estimated flow field (H, W, 2).
        """
        sz = (self.images.shape[0], self.images.shape[1])

        if init is None:
            uv = np.zeros((*sz, 2))
        else:
            uv = init.copy()

        # Preprocess images
        if self.texture:
            images = structure_texture_decomposition_rof(self.images)
        else:
            images = scale_image(self.images, 0, 255)

        # Auto pyramid levels
        self.pyramid_levels = self._auto_pyramid_levels(images)

        # Build pyramid
        pyramid_images = self._build_pyramid(
            images, self.pyramid_levels, self.pyramid_spacing
        )

        # Coarse-to-fine
        for l in range(self.pyramid_levels - 1, -1, -1):
            if self.display:
                print(f"Pyramid level: {l + 1}")

            small = self._copy_with_images(pyramid_images[l])
            nsz = (pyramid_images[l].shape[0], pyramid_images[l].shape[1])
            uv = resample_flow(uv, nsz)
            uv = small.compute_flow_base(uv)

            # Keep reliability from finest level
            if l == 0:
                self.reliability = small.reliability

        # Final median filter
        if self.median_filter_size is not None:
            uv[:, :, 0] = median_filter(
                uv[:, :, 0], size=self.median_filter_size, mode='reflect'
            )
            uv[:, :, 1] = median_filter(
                uv[:, :, 1], size=self.median_filter_size, mode='reflect'
            )

        return uv

    def _copy_with_images(self, images):
        """Create a copy of this object with different images."""
        small = copy.copy(self)
        small.images = images
        small.pyramid_levels = 1
        return small

    def compute_flow_base(self, uv):
        """Compute flow at a single pyramid level using Lucas-Kanade.

        Iteratively warps the second image toward the first and solves the
        local 2x2 structure tensor system at each pixel.

        Args:
            uv: Current flow estimate (H, W, 2).

        Returns:
            uv: Updated flow estimate.
        """
        sigma = self.sigma if self.sigma is not None else self.window_size / 4.0
        G = fspecial_gaussian(self.window_size, sigma)

        for i in range(self.max_warping_iters):
            # Compute spatiotemporal derivatives
            It, Ix, Iy = partial_deriv(
                self.images, uv, self.interpolation_method, self.deriv_filter
            )

            # Average across color channels
            if Ix.ndim == 3:
                Ix = np.mean(Ix, axis=2)
                Iy = np.mean(Iy, axis=2)
                It = np.mean(It, axis=2)

            # Structure tensor components (Gaussian-weighted sums)
            Sxx = correlate(Ix * Ix, G, mode='reflect')
            Sxy = correlate(Ix * Iy, G, mode='reflect')
            Syy = correlate(Iy * Iy, G, mode='reflect')
            Sxt = correlate(Ix * It, G, mode='reflect')
            Syt = correlate(Iy * It, G, mode='reflect')

            # Determinant and eigenvalue-based reliability
            det = Sxx * Syy - Sxy * Sxy
            trace = Sxx + Syy
            discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
            eig_min = (trace - np.sqrt(discriminant)) / 2.0
            reliable = eig_min > self.eigen_threshold

            # Store reliability map (updated each iteration, final value kept)
            self.reliability = eig_min

            # Solve 2x2 system via Cramer's rule where reliable
            inv_det = np.where(reliable, 1.0 / np.maximum(np.abs(det), 1e-20), 0.0)
            du = (-Sxt * Syy + Syt * Sxy) * inv_det
            dv = (-Syt * Sxx + Sxt * Sxy) * inv_det
            du[~reliable] = 0.0
            dv[~reliable] = 0.0

            # Limit update
            if self.limit_update:
                du = np.clip(du, -1, 1)
                dv = np.clip(dv, -1, 1)

            if self.display:
                norm = np.sqrt(np.mean(du**2 + dv**2))
                print(f"  Iteration: {i + 1}  (norm: {norm:.6f})")

            # Early termination
            if np.sqrt(np.mean(du**2 + dv**2)) < 1e-4:
                break

            uv[:, :, 0] += du
            uv[:, :, 1] += dv

            # Median filter per iteration
            if self.median_filter_size is not None:
                for _ in range(self.mf_iter):
                    uv[:, :, 0] = median_filter(
                        uv[:, :, 0], size=self.median_filter_size, mode='reflect'
                    )
                    uv[:, :, 1] = median_filter(
                        uv[:, :, 1], size=self.median_filter_size, mode='reflect'
                    )

        return uv

    def flow_operator(self, uv, duv=None, It=None, Ix=None, Iy=None):
        """Not used by Lucas-Kanade (local method, no global linear system)."""
        raise NotImplementedError(
            "Lucas-Kanade is a local method and does not build a global "
            "linear system. Use compute_flow_base() instead."
        )
