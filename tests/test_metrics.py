"""Tests for flow evaluation metrics."""
import numpy as np
from optical_flow.evaluation.metrics import flow_angular_error


class TestFlowAngularError:
    """Test AAE and EPE computation."""

    def test_perfect_flow(self):
        """Zero error for identical flows."""
        H, W = 20, 20
        u = np.random.randn(H, W)
        v = np.random.randn(H, W)
        aae, std_ae, aepe = flow_angular_error(u, v, u, v)
        np.testing.assert_allclose(aae, 0.0, atol=1e-5)
        np.testing.assert_allclose(aepe, 0.0, atol=1e-10)

    def test_zero_flow(self):
        """Zero ground truth and zero estimate should give zero error."""
        H, W = 10, 10
        zeros = np.zeros((H, W))
        aae, std_ae, aepe = flow_angular_error(zeros, zeros, zeros, zeros)
        np.testing.assert_allclose(aae, 0.0)
        np.testing.assert_allclose(aepe, 0.0)

    def test_endpoint_error(self):
        """EPE should be euclidean distance between flow vectors."""
        H, W = 1, 1
        tu = np.array([[3.0]])
        tv = np.array([[0.0]])
        u = np.array([[0.0]])
        v = np.array([[0.0]])
        aae, std_ae, aepe = flow_angular_error(tu, tv, u, v)
        np.testing.assert_allclose(aepe, 3.0)

    def test_border_cropping(self):
        """Border parameter should exclude border pixels."""
        H, W = 20, 20
        tu = np.random.randn(H, W)
        tv = np.random.randn(H, W)
        # With large border, should only evaluate interior
        aae1, _, _ = flow_angular_error(tu, tv, tu, tv, border=0)
        aae2, _, _ = flow_angular_error(tu, tv, tu, tv, border=5)
        np.testing.assert_allclose(aae1, 0.0, atol=1e-5)
        np.testing.assert_allclose(aae2, 0.0, atol=1e-5)

    def test_angular_error_range(self):
        """AAE should be between 0 and 180 degrees."""
        H, W = 20, 20
        tu = np.random.randn(H, W)
        tv = np.random.randn(H, W)
        u = np.random.randn(H, W)
        v = np.random.randn(H, W)
        aae, std_ae, aepe = flow_angular_error(tu, tv, u, v)
        assert 0 <= aae <= 180

    def test_unknown_flow_filtering(self):
        """Unknown flow values (>1e9) should be filtered out."""
        H, W = 5, 5
        tu = np.zeros((H, W))
        tv = np.zeros((H, W))
        u = np.zeros((H, W))
        v = np.zeros((H, W))
        # Set some to unknown
        tu[0, 0] = 1e10
        tv[0, 0] = 1e10
        aae, std_ae, aepe = flow_angular_error(tu, tv, u, v)
        np.testing.assert_allclose(aae, 0.0, atol=1e-10)
        np.testing.assert_allclose(aepe, 0.0, atol=1e-10)
