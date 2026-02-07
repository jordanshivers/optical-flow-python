"""Tests for sparse convolution matrix construction."""
import numpy as np
from scipy.signal import convolve2d
from optical_flow.utils.sparse_ops import convmtxn, make_convn_mat, make_imfilter_mat


class TestConvmtxn:
    """Test full convolution matrix."""

    def test_identity_filter(self):
        """Identity filter should reproduce the input."""
        F = np.array([[1.0]])
        sz = (3, 3)
        M = convmtxn(F, sz)
        x = np.arange(9, dtype=float)
        y = M @ x
        np.testing.assert_allclose(y, x)

    def test_matches_convolve2d(self):
        """convmtxn should match scipy's convolve2d in 'full' mode."""
        np.random.seed(0)
        F = np.array([[1, -1]])
        sz = (4, 5)
        img = np.random.randn(*sz)
        M = convmtxn(F, sz)
        result_matrix = (M @ img.ravel(order='F')).reshape(
            sz[0] + F.shape[0] - 1, sz[1] + F.shape[1] - 1, order='F'
        )
        result_direct = convolve2d(img, F, mode='full')
        np.testing.assert_allclose(result_matrix, result_direct, atol=1e-12)

    def test_output_shape(self):
        """Check output dimensions."""
        F = np.array([[1, 2], [3, 4]])
        sz = (5, 6)
        M = convmtxn(F, sz)
        assert M.shape == (6 * 7, 5 * 6)


class TestMakeConvnMat:
    """Test convolution matrix with shape control."""

    def test_same_shape(self):
        """'same' mode output size matches input size."""
        F = np.array([[1, -1]])
        sz = (4, 5)
        M = make_convn_mat(F, sz, shape='same')
        assert M.shape[0] == sz[0] * sz[1]

    def test_valid_shape(self):
        """'valid' mode output size is correct."""
        F = np.array([[1, -1]])
        sz = (4, 5)
        M = make_convn_mat(F, sz, shape='valid')
        valid_sz = (sz[0] - F.shape[0] + 1, sz[1] - F.shape[1] + 1)
        assert M.shape[0] == valid_sz[0] * valid_sz[1]

    def test_sameswap(self):
        """'valid' with 'sameswap' should have same output size as input."""
        F = np.array([[1, -1]])
        sz = (4, 5)
        M = make_convn_mat(F, sz, shape='valid', pad='sameswap')
        assert M.shape[0] == sz[0] * sz[1]

    def test_full_matches_convmtxn(self):
        """'full' should be identical to convmtxn."""
        F = np.array([[1, 2], [3, 4]])
        sz = (3, 4)
        M_full = make_convn_mat(F, sz, shape='full')
        M_ref = convmtxn(F, sz)
        np.testing.assert_allclose(M_full.toarray(), M_ref.toarray())


class TestMakeImfilterMat:
    """Test image filtering matrix."""

    def test_identity_filter(self):
        """Identity filter reproduces input."""
        F = np.array([[1.0]])
        sz = (4, 4)
        M = make_imfilter_mat(F, sz)
        x = np.arange(16, dtype=float)
        y = M @ x
        np.testing.assert_allclose(y, x)

    def test_zero_boundary(self):
        """Zero boundary should pad with zeros."""
        F = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])  # Shift right
        sz = (3, 3)
        M = make_imfilter_mat(F, sz, boundary='0')
        img = np.ones(9)
        result = M @ img
        # Rightmost column should see zero from boundary
        result_2d = result.reshape(3, 3, order='F')
        assert result_2d[0, -1] == 0  # Rightmost wraps to zero

    def test_output_shape(self):
        """Output matches input for 'same'."""
        F = np.array([[1, -1]])
        sz = (5, 6)
        M = make_imfilter_mat(F, sz)
        assert M.shape == (30, 30)

    def test_replicate_boundary(self):
        """Replicate should not introduce zeros at boundary."""
        F = np.array([[0, 1, 0]])
        sz = (3, 3)
        M = make_imfilter_mat(F, sz, boundary='replicate')
        img = np.ones(9)
        result = M @ img
        # All ones should remain ones with shift + replicate
        np.testing.assert_allclose(result, np.ones(9))
