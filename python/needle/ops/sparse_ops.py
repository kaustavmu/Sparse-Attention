import numpy as np
import scipy.sparse as sp
from ..autograd import SparseTensorOp, SparseTensor


###########################################################
#                      Helper utils                       #
###########################################################

def _is_sparse(x):
    return sp.issparse(x)

def _to_dense(x):
    return x.toarray() if _is_sparse(x) else np.asarray(x)

def _wrap_tensor(x):
    if sp.issparse(x):
        x = x.toarray()
    return SparseTensor(x)



###########################################################
#                  Public sparse wrappers                 #
###########################################################

def sparse_add(a, b):
    a, b = _wrap_tensor(a), _wrap_tensor(b)
    return SparseAdd()(a, b)

def sparse_multiply(a, b):
    a, b = _wrap_tensor(a), _wrap_tensor(b)
    return SparseMul()(a, b)

def sparse_matmul(a, b):
    a, b = _wrap_tensor(a), _wrap_tensor(b)
    return SparseMatMul()(a, b)

def sparse_add_scalar(a, scalar):
    a = _wrap_tensor(a)
    return SparseAddScalar(scalar)(a)

def sparse_mul_scalar(a, scalar):
    a = _wrap_tensor(a)
    return SparseMulScalar(scalar)(a)

def sparse_transpose(a):
    a = _wrap_tensor(a)
    return SparseTranspose()(a)

def sparse_relu(a):
    a = _wrap_tensor(a)
    return SparseReLU()(a)

###########################################################
#                  Operator base classes                  #
###########################################################

class SparseAdd(SparseTensorOp):
    """Elementwise addition for sparse matrices."""

    def compute(self, a, b):
        if _is_sparse(a) and _is_sparse(b):
            return a + b
        return np.add(_to_dense(a), _to_dense(b))

    def gradient(self, out_grad, node):
        # d/dA (A + B) = 1, d/dB (A + B) = 1
        return out_grad, out_grad


class SparseMul(SparseTensorOp):
    """Elementwise multiplication for sparse matrices."""

    def compute(self, a, b):
        if _is_sparse(a) and _is_sparse(b):
            return a.multiply(b)
        return np.multiply(_to_dense(a), _to_dense(b))

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b
        grad_b = out_grad * a
        return grad_a, grad_b


class SparseMatMul(SparseTensorOp):
    """Matrix multiplication supporting sparse inputs."""

    def compute(self, a, b):
        if _is_sparse(a) and _is_sparse(b):
            return a.dot(b)
        if _is_sparse(a):
            return a.dot(_to_dense(b))
        if _is_sparse(b):
            return _to_dense(a).dot(b.toarray())
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
        if _is_sparse(a):
            if self.scalar == 0:
                return a.copy()
            return a + self.scalar  # will densify
        return _to_dense(a) + self.scalar

    def gradient(self, out_grad, node):
        # d/dA (A + c) = 1
        return out_grad


class SparseMulScalar(SparseTensorOp):
    """Multiply scalar to sparse or dense input."""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if _is_sparse(a):
            return a * self.scalar
        return _to_dense(a) * self.scalar

    def gradient(self, out_grad, node):
        # d/dA (c * A) = c
        return out_grad * self.scalar


class SparseTranspose(SparseTensorOp):
    """Transpose sparse or dense input."""

    def compute(self, a):
        if _is_sparse(a):
            return a.transpose()
        return np.transpose(_to_dense(a))

    def gradient(self, out_grad, node):
        # d/dA (A^T) = (dL/dA)^T
        return out_grad.transpose()


class SparseReLU(SparseTensorOp):
    """ReLU activation for sparse tensors."""

    def compute(self, a):
        if _is_sparse(a):
            a_coo = a.tocoo()
            mask = a_coo.data > 0
            if mask.sum() == 0:
                return sp.csr_matrix(a.shape)
            return sp.coo_matrix(
                (a_coo.data[mask], (a_coo.row[mask], a_coo.col[mask])),
                shape=a.shape,
            ).tocsr()
        return np.maximum(_to_dense(a), 0)

    def gradient(self, out_grad, node):
        """Gradient: mask out gradients where input <= 0."""
        a = node.inputs[0]
        if _is_sparse(a):
            dense_a = a.toarray()
            mask = (dense_a > 0).astype(float)
            return out_grad * mask
        mask = (_to_dense(a) > 0).astype(float)
        return out_grad * mask
