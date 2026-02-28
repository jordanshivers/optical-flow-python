"""Smoke tests for the flow_fast accelerated package.

Verifies that flow_fast produces valid results and uses the same API
as optical_flow.
"""
import numpy as np
import pytest


class TestFlowFastImports:
    """Verify all public API imports work."""

    def test_top_level_imports(self):
        from flow_fast import (
            estimate_flow, read_flo, write_flo,
            flow_to_color, plot_flow, flow_angular_error,
            load_of_method, warmup,
        )

    def test_submodule_imports(self):
        from flow_fast.methods.config import load_of_method
        from flow_fast.robust.robust_function import RobustFunction
        from flow_fast.robust.penalties import quadratic, lorentzian, charbonnier
        from flow_fast.utils.derivatives import partial_deriv
        from flow_fast.utils.pyramid import compute_image_pyramid
        from flow_fast.utils.image_processing import structure_texture_decomposition_rof
        from flow_fast.solvers.dispatch import get_solver

    def test_accel_imports(self):
        from flow_fast._accel.weighted_median_nb import weighted_median_filter_numba
        from flow_fast._accel.penalties_nb import generalized_charbonnier_deriv_over_x
        from flow_fast._accel.bicubic_interp_nb import eval_bicubic_polynomial


class TestFlowFastMethods:
    """Verify all methods can be loaded and run."""

    def test_load_all_methods(self):
        from flow_fast.methods.config import load_of_method
        methods = [
            'classic+nl-fast', 'classic+nl', 'classic+nl-full',
            'hs-brightness', 'hs',
            'ba-brightness', 'ba', 'classic-l',
            'classic-c', 'classic-c-brightness',
            'classic++', 'classic-c-a',
        ]
        for m in methods:
            ope = load_of_method(m)
            assert ope is not None, f"Failed to load method: {m}"


class TestFlowFastSolver:
    """Verify solver dispatch works."""

    def test_auto_solver(self):
        from flow_fast.solvers.dispatch import get_solver
        solver = get_solver('auto')
        assert solver is not None
        assert hasattr(solver, 'solve')

    def test_pcg_solver(self):
        from flow_fast.solvers.dispatch import get_solver
        from scipy import sparse
        solver = get_solver('pcg')
        # Solve a trivial 4x4 system
        A = sparse.diags([2.0, 3.0, 4.0, 5.0], 0)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(A, b, (2, 2))
        # Solution reshaped with Fortran ordering: [0.5, 0.75, 2/3, 0.8]
        np.testing.assert_allclose(x.ravel(), [0.5, 0.75, 2/3, 0.8], rtol=1e-3)


class TestFlowFastHS:
    """Test Horn-Schunck with flow_fast."""

    def test_zero_flow_identical_frames(self):
        """Identical frames should produce near-zero flow."""
        from flow_fast.methods.hs import HSOpticalFlow
        H, W = 32, 32
        np.random.seed(42)
        img = np.random.rand(H, W) * 255
        hs = HSOpticalFlow()
        hs.images = np.stack([img, img], axis=2)
        hs.lambda_ = 80
        hs.lambda_q = 80
        hs.pyramid_levels = 1
        hs.max_warping_iters = 3
        uv = hs.compute_flow(np.zeros((H, W, 2)))
        assert uv.shape == (H, W, 2)
        assert not np.any(np.isnan(uv)), "Flow contains NaN values"
        np.testing.assert_allclose(uv, 0, atol=0.1)


class TestFlowFastEstimateFlow:
    """Integration test using the high-level estimate_flow API."""

    def test_estimate_flow_synthetic(self):
        """Run estimate_flow on a synthetic pair."""
        from flow_fast import estimate_flow
        np.random.seed(42)
        H, W = 48, 64
        im1 = np.random.rand(H, W, 3) * 255
        im2 = im1.copy()
        # Small shift right
        im2[:, 1:, :] = im1[:, :-1, :]
        uv = estimate_flow(im1, im2, method='hs-brightness')
        assert uv.shape == (H, W, 2)
        assert not np.any(np.isnan(uv)), "Flow contains NaN values"

    @pytest.mark.slow
    def test_rubberwhale_classic_nl_fast(self, rubberwhale_images):
        """Classic+NL-Fast on RubberWhale should achieve good accuracy."""
        from flow_fast import estimate_flow, flow_angular_error
        from flow_fast.io.flo_io import read_flo
        import os

        im1, im2 = rubberwhale_images
        uv = estimate_flow(im1, im2, method='classic+nl-fast')

        assert uv.shape == (im1.shape[0], im1.shape[1], 2)
        assert not np.any(np.isnan(uv)), "Flow contains NaN values"

        # Check against ground truth
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        gt_path = os.path.join(data_dir, 'other-gt-flow', 'RubberWhale', 'flow10.flo')
        if os.path.exists(gt_path):
            gt = read_flo(gt_path)
            aae, _, aepe = flow_angular_error(
                gt[:, :, 0], gt[:, :, 1], uv[:, :, 0], uv[:, :, 1]
            )
            assert aae < 5.0, f"AAE too high: {aae:.2f}"
            assert aepe < 0.2, f"AEPE too high: {aepe:.4f}"


class TestFlowFastAPICompatibility:
    """Verify flow_fast has the same public API as optical_flow."""

    def test_same_functions_exported(self):
        import optical_flow
        import flow_fast

        of_public = {name for name in dir(optical_flow)
                     if not name.startswith('_') and callable(getattr(optical_flow, name))}
        ff_public = {name for name in dir(flow_fast)
                     if not name.startswith('_') and callable(getattr(flow_fast, name))}

        # flow_fast should have at least everything optical_flow has
        missing = of_public - ff_public
        assert not missing, f"flow_fast missing public API: {missing}"
