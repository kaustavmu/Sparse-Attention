import numpy as np
from ..autograd import SparseTensorOp, SparseTensor


###########################################################
#                      Helper utils                       #
###########################################################

def _to_dense(x):
    """
    Convert underlying data to a dense NumPy array.

    Supports:
    - NumPy arrays
    - Needle NDArray (has .numpy())
    - Anything array-like that np.asarray can handle
    """
    # Needle NDArray: has numpy() method
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _is_needle_array(x) -> bool:
    """Detect Needle NDArray-like object (we want to stay in that world if possible)."""
    return hasattr(x, "numpy") and hasattr(x, "__add__") and hasattr(x, "__matmul__")


###########################################################
#                  Public sparse wrappers                 #
###########################################################

def sparse_add(a, b):
    return SparseAdd()(a, b)

def sparse_multiply(a, b):
    return SparseMul()(a, b)

def sparse_matmul(a, b):
    return SparseMatMul()(a, b)

def sparse_add_scalar(a, scalar):
    return SparseAddScalar(scalar)(a)

def sparse_mul_scalar(a, scalar):
    return SparseMulScalar(scalar)(a)

def sparse_transpose(a):
    return SparseTranspose()(a)

def sparse_relu(a):
    return SparseReLU()(a)


###########################################################
#                  Operator implementations               #
###########################################################

class SparseAdd(SparseTensorOp):
    """Elementwise addition for sparse matrices."""

    def compute(self, a, b):
        # If these are Needle NDArrays, use their + directly to preserve sparsity.
        if _is_needle_array(a) and _is_needle_array(b):
            return a + b
        return _to_dense(a) + _to_dense(b)

    def gradient(self, out_grad, node):
        # d/dA (A + B) = 1, d/dB (A + B) = 1
        return out_grad, out_grad


class SparseMul(SparseTensorOp):
    """Elementwise multiplication for sparse matrices."""

    def compute(self, a, b):
        if _is_needle_array(a) and _is_needle_array(b):
            return a * b
        return _to_dense(a) * _to_dense(b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # Gradients use the same algebra (and stay sparse if a/b are sparse NDArrays)
        grad_a = out_grad * b
        grad_b = out_grad * a
        return grad_a, grad_b


class SparseMatMul(SparseTensorOp):
    """Matrix multiplication supporting sparse inputs."""

    def compute(self, a, b):
        if _is_needle_array(a) and _is_needle_array(b):
            # Let NDArray (possibly CSR) handle matmul
            return a @ b
        # Fallback dense NumPy
        return _to_dense(a) @ _to_dense(b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # d/dA (A @ B) = out_grad @ B^T
        # d/dB (A @ B) = A^T @ out_grad
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        return grad_a, grad_b


class SparseAddScalar(SparseTensorOp):
    """Add scalar to sparse or dense input."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if _is_needle_array(a):
            return a + self.scalar
        return _to_dense(a) + self.scalar

    def gradient(self, out_grad, node):
        # d/dA (A + c) = 1
        return out_grad


class SparseMulScalar(SparseTensorOp):
    """Multiply scalar to sparse or dense input."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if _is_needle_array(a):
            return a * self.scalar
        return _to_dense(a) * self.scalar

    def gradient(self, out_grad, node):
        # d/dA (c * A) = c
        return out_grad * self.scalar


class SparseTranspose(SparseTensorOp):
    """Transpose sparse or dense input."""

    def compute(self, a):
        if _is_needle_array(a):
            # NDArray’s transpose is usually called permute, but SparseTensor will
            # pass whatever .transpose() returns, so we keep this simple:
            return a.permute((1, 0)) if hasattr(a, "permute") else _to_dense(a).T
        return _to_dense(a).T

    def gradient(self, out_grad, node):
        # d/dA (A^T) = (dL/dA)^T
        return out_grad.transpose()


class SparseReLU(SparseTensorOp):
    """ReLU activation for sparse tensors."""

    def compute(self, a):
        if _is_needle_array(a):
            # ReLU(a) = max(a, 0)
            return a.maximum(0.0)
        return np.maximum(_to_dense(a), 0)

    def gradient(self, out_grad, node):
        """Gradient: mask out gradients where input <= 0."""
        a = node.inputs[0]
        if _is_needle_array(a.realize_cached_data()):
            # Use elementwise comparison & masking in NDArray
            data = a.realize_cached_data()
            mask = (data > 0)
            return out_grad * mask
        mask = (_to_dense(a.realize_cached_data()) > 0).astype(float)
        return out_grad * mask
