import numpy as np
import scipy.sparse as sp
from ..autograd import SparseTensorOp, SparseTensor


###########################################################
#                      Helper utils                       #
###########################################################

def _to_dense(x):
    return np.asarray(x)

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
#                  Operator base classes                  #
###########################################################

class SparseAdd(SparseTensorOp):
    """Elementwise addition for sparse matrices."""

    def compute(self, a, b):
        return np.add(_to_dense(a), _to_dense(b))

    def gradient(self, out_grad, node):
        # d/dA (A + B) = 1, d/dB (A + B) = 1
        return out_grad, out_grad


class SparseMul(SparseTensorOp):
    """Elementwise multiplication for sparse matrices."""

    def compute(self, a, b):
        return np.multiply(_to_dense(a), _to_dense(b))

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b
        grad_b = out_grad * a
        return grad_a, grad_b


class SparseMatMul(SparseTensorOp):
    """Matrix multiplication supporting sparse inputs."""

    def compute(self, a, b):
        return np.dot(_to_dense(a), _to_dense(b))

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
        return a + self.scalar

    def gradient(self, out_grad, node):
        # d/dA (A + c) = 1
        return out_grad


class SparseMulScalar(SparseTensorOp):
    """Multiply scalar to sparse or dense input."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a * self.scalar

    def gradient(self, out_grad, node):
        # d/dA (c * A) = c
        return out_grad * self.scalar


class SparseTranspose(SparseTensorOp):
    """Transpose sparse or dense input."""

    def compute(self, a):
        return np.transpose(_to_dense(a))

    def gradient(self, out_grad, node):
        # d/dA (A^T) = (dL/dA)^T
        return out_grad.transpose()


class SparseReLU(SparseTensorOp):
    """ReLU activation for sparse tensors."""
    def compute(self, a):
        return np.maximum(_to_dense(a), 0)

    def gradient(self, out_grad, node):
        """Gradient: mask out gradients where input <= 0."""
        a = node.inputs[0]
        mask = (_to_dense(a) > 0).astype(float)
        return out_grad * mask