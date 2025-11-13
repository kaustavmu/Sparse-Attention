import pytest
import numpy as np
import sys

sys.path.append("./python")
from needle.autograd import SparseTensor
from needle.ops.sparse_ops import (
    sparse_add,
    sparse_multiply,
    sparse_matmul,
    sparse_add_scalar,
    sparse_mul_scalar,
    sparse_transpose,
    sparse_relu,
)

# Helper function to convert SparseTensor -> NumPy array
def dense_equiv(x):
    if isinstance(x, SparseTensor):
        x.realize_cached_data()
        return x.cached_data.toarray()
    return x

# Example fixture for testing
@pytest.fixture
def sample_sparse_matrices():
    # Use NumPy arrays instead of Python lists
    A = SparseTensor(np.array([[1, 0], [0, 2]]))
    B = SparseTensor(np.array([[0, 3], [4, 0]]))
    return A, B

def test_sparse_add(sample_sparse_matrices):
    A, B = sample_sparse_matrices
    out = sparse_add(A, B)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) + dense_equiv(B)
    )

def test_sparse_mul_scalar(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_mul_scalar(A, 3.0)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) * 3.0
    )

def test_sparse_matmul(sample_sparse_matrices):
    A, B = sample_sparse_matrices
    out = sparse_matmul(A, B)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A) @ dense_equiv(B)
    )

def test_sparse_transpose(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_transpose(A)
    np.testing.assert_allclose(
        dense_equiv(out),
        dense_equiv(A).T
    )

def test_sparse_relu(sample_sparse_matrices):
    A, _ = sample_sparse_matrices
    out = sparse_relu(A)
    np.testing.assert_allclose(
        dense_equiv(out),
        np.maximum(dense_equiv(A), 0)
    )
