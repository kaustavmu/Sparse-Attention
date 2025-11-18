import pytest
import numpy as np
import sys

sys.path.append("./python")

from needle.autograd import SparseTensor, Tensor
from needle.ops.sparse_ops import (
    sparse_add,
    sparse_multiply,
    sparse_matmul,
    sparse_add_scalar,
    sparse_mul_scalar,
    sparse_transpose,
    sparse_relu,
)

# If you want to test CSR NDArray end-to-end as well:
from needle import ndarray_sparse as nds


# Helper function to convert SparseTensor -> NumPy array
def dense_equiv(x):
    if isinstance(x, SparseTensor):
        x.realize_cached_data()
        data = x.cached_data
        # Needle NDArray
        if hasattr(data, "numpy"):
            return data.numpy()
        # SciPy CSR (in case you ever use it again)
        if hasattr(data, "toarray"):
            return data.toarray()
        # Fallback: assume NumPy-like
        return np.asarray(data)
    return np.asarray(x)


@pytest.fixture(params=["numpy", "csr"])
def sample_sparse_matrices(request):
    """Fixture that can test both NumPy-backed and CSR-backed SparseTensors."""
    dense_A = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    dense_B = np.array([[0., 3.], [4., 0.]], dtype=np.float32)

    if request.param == "numpy":
        A = SparseTensor(dense_A)
        B = SparseTensor(dense_B)
    else:
        # CSR NDArray backend
        A_nd = nds.array(dense_A, device=nds.csr())
        B_nd = nds.array(dense_B, device=nds.csr())
        A = SparseTensor(A_nd)
        B = SparseTensor(B_nd)

    return A, B


def test_sparse_add(sample_sparse_matrices):
    A, B = sample_sparse_matrices
    out = sparse_add(A, B)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) + dense_equiv(B),
    )


def test_sparse_multiply(sample_sparse_matrices):
    A, B = sample_sparse_matrices
    out = sparse_multiply(A, B)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) * dense_equiv(B),
    )


def test_sparse_mul_scalar(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_mul_scalar(A, 3.0)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) * 3.0,
    )


def test_sparse_add_scalar(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_add_scalar(A, 2.0)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) + 2.0,
    )


def test_sparse_matmul(sample_sparse_matrices):
    A, B = sample_sparse_matrices
    out = sparse_matmul(A, B)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) @ dense_equiv(B),
    )


def test_sparse_transpose(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_transpose(A)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A).T,
    )


def test_sparse_relu(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_relu(A)
    np.testing.assert_allclose(
        dense_equiv(out),
        np.maximum(dense_equiv(A), 0),
    )

def test_sparse_matmul_backward_numpy():
    # simple dense-backed SparseTensors
    A = SparseTensor(np.array([[1., 0.], [0., 2.]], dtype=np.float32), requires_grad=True)
    B = SparseTensor(np.array([[0., 3.], [4., 0.]], dtype=np.float32), requires_grad=True)

    out = sparse_matmul(A, B)     # 2x2
    # Reduce to scalar so backward is defined
    loss = Tensor(out.realize_cached_data().sum(), requires_grad=True)
    loss.backward()

    # Check that gradients exist and shapes match
    assert A.grad.shape == A.shape
    assert B.grad.shape == B.shape
