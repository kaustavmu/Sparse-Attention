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
import needle as ndl
from needle import backend_ndarray as nd

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
        A_nd = nd.array(dense_A, device=nd.csr())
        B_nd = nd.array(dense_B, device=nd.csr())
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

def test_sparse_matmul_backward():
    from needle.autograd import SparseTensor
    from needle.ops.sparse_ops import sparse_matmul

    A = SparseTensor(np.array([[1., 0.], [0., 2.]], dtype=np.float32), requires_grad=True)
    B = SparseTensor(np.array([[0., 3.], [4., 0.]], dtype=np.float32), requires_grad=True)

    # Forward
    C = sparse_matmul(A, B)  # shape (2,2)

    # Use SparseTensor.sum(), NOT Tensor
    loss = C.sum()
    loss.backward()

    # Expected grads
    expected_grad_A = np.array([[3., 0.], [0., 0.]], dtype=np.float32)
    expected_grad_B = np.array([[1., 0.], [0., 2.]], dtype=np.float32)

    np.testing.assert_allclose(A.grad.cached_data, expected_grad_A)
    np.testing.assert_allclose(B.grad.cached_data, expected_grad_B)

def test_sparse_sum(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    denseA = dense_equiv(A)

    # sum entire matrix
    out = A.sum()
    np.testing.assert_allclose(dense_equiv(out), denseA.sum())

    # sum along axis 0
    out = A.sum(axes=0)
    np.testing.assert_allclose(dense_equiv(out), denseA.sum(axis=0))

    # sum along axis 1
    out = A.sum(axes=1)
    np.testing.assert_allclose(dense_equiv(out), denseA.sum(axis=1))

def test_sparse_sum_backward():
    A = SparseTensor(np.array([[1., 0.], [3., -2.]], dtype=np.float32), requires_grad=True)

    loss = A.sum()   # scalar
    loss.backward()

    np.testing.assert_allclose(dense_equiv(A.grad), np.ones_like(dense_equiv(A)))
