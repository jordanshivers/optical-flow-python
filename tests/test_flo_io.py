"""Tests for .flo file I/O."""
import os
import numpy as np
import pytest
from optical_flow.io.flo_io import read_flo, write_flo, read_flow_file


class TestFloIO:
    """Test .flo read/write round-trip."""

    def test_write_read_roundtrip(self, tmp_path):
        """Write a flow field and read it back."""
        flow = np.random.randn(100, 200, 2).astype(np.float32)
        flo_path = str(tmp_path / 'test.flo')
        write_flo(flow, flo_path)
        flow_read = read_flo(flo_path)
        np.testing.assert_array_equal(flow, flow_read)

    def test_read_rubberwhale(self, data_dir):
        """Read the RubberWhale ground truth .flo file."""
        gt_path = os.path.join(data_dir, 'other-gt-flow', 'RubberWhale', 'flow10.flo')
        flow = read_flo(gt_path)
        assert flow.ndim == 3
        assert flow.shape[2] == 2
        assert flow.dtype == np.float32

    def test_write_wrong_shape_raises(self, tmp_path):
        """Writing non-(H,W,2) array should raise."""
        flow_bad = np.zeros((10, 10))
        with pytest.raises(ValueError):
            write_flo(flow_bad, str(tmp_path / 'bad.flo'))

    def test_read_invalid_tag(self, tmp_path):
        """Reading a file with wrong tag should raise."""
        flo_path = str(tmp_path / 'invalid.flo')
        with open(flo_path, 'wb') as f:
            np.array([12345.0], dtype=np.float32).tofile(f)
            np.array([10, 10], dtype=np.int32).tofile(f)
            np.zeros(200, dtype=np.float32).tofile(f)
        with pytest.raises(ValueError, match="Invalid .flo"):
            read_flo(flo_path)

    def test_small_flow(self, tmp_path):
        """Test with tiny 1x1 flow."""
        flow = np.array([[[0.5, -0.3]]], dtype=np.float32)
        flo_path = str(tmp_path / 'tiny.flo')
        write_flo(flow, flo_path)
        flow_read = read_flo(flo_path)
        np.testing.assert_allclose(flow, flow_read)


class TestReadFlowFile:
    """Test the Middlebury dataset loader."""

    def test_load_rubberwhale(self, data_dir):
        """Load the RubberWhale test sequence."""
        im1, im2, tu, tv = read_flow_file('RubberWhale', 10, data_dir=data_dir)
        assert im1.shape == im2.shape
        assert im1.ndim == 3  # Color image
        assert tu is not None
        assert tv is not None
        assert tu.shape == im1.shape[:2]
