import numpy as np
from ..autograd import SparseTensorOp, SparseTensor, TensorOp
from .. import backend_ndarray as nd


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
    arr = np.asarray(x)
    if arr.dtype == object:
        flat = [ _to_dense(elem) for elem in arr.ravel() ]
        if not flat:
            return np.asarray([])
        first = np.asarray(flat[0])
        stacked = np.stack([np.asarray(elem) for elem in flat])
        return stacked.reshape(arr.shape + first.shape)
    return arr


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

def sparse_sum(a, axes=None):
    return Summation(axes)(a)


def dense_to_sparse(a, threshold=0.0, use_csr=True):
    return DenseToSparse(threshold=threshold, use_csr=use_csr)(a)


def sparse_to_dense(a):
    return SparseToDense()(a)


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
        def _realize(val):
            return val.realize_cached_data() if hasattr(val, "realize_cached_data") else val

        def _is_ndarray(obj):
            return _is_needle_array(obj)

        def _transpose_data(data):
            if _is_ndarray(data):
                if hasattr(data, "permute"):
                    return data.permute((1, 0))
                if hasattr(data, "transpose"):
                    return data.transpose()
            return _to_dense(data).T

        def _matmul(lhs, rhs):
            if _is_ndarray(lhs) and _is_ndarray(rhs):
                return lhs @ rhs
            return _to_dense(lhs) @ _to_dense(rhs)

        a, b = node.inputs
        a_data = _realize(a)
        b_data = _realize(b)
        out_data = _realize(out_grad)

        grad_a_data = _matmul(out_data, _transpose_data(b_data))
        grad_b_data = _matmul(_transpose_data(a_data), out_data)

        grad_a = SparseTensor.make_const(grad_a_data)
        grad_b = SparseTensor.make_const(grad_b_data)
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

class Summation(SparseTensorOp):
    """
    Summation over one or more axes.

    Works for:
      - NumPy dense arrays
      - Needle NDArray (dense or sparse/CSR)
    """

    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        # Handle NDArray device: use its own reduce_sum
        if _is_needle_array(a):
            return a.reduce_sum(self.axes)

        # Fall back to NumPy
        return _to_dense(a).sum(axis=self.axes)

    def gradient(self, out_grad, node):
        """
        If y = sum(x, axis),
        then dy/dx = broadcast_to(out_grad, x.shape)
        """
        x = node.inputs[0]
        x_shape = x.shape
        axes = self.axes

        # Case 1: sum entire tensor → out_grad is scalar
        if axes is None:
            return out_grad.broadcast_to(x_shape)

        # Normalize axes into a tuple
        if isinstance(axes, int):
            axes = (axes,)

        # After summation, dimensions are removed.
        # We need to reshape out_grad by re-inserting the reduced dims.
        new_shape = list(out_grad.shape)
        for ax in sorted(axes):
            new_shape.insert(ax, 1)

        reshaped = out_grad.reshape(new_shape)
        return reshaped.broadcast_to(x_shape)


class DenseToSparse(SparseTensorOp):
    """
    Convert a dense Tensor into a SparseTensor while optionally thresholding
    small entries and moving the storage to the CSR backend.
    """

    def __init__(self, threshold=0.0, use_csr=True):
        self.threshold = threshold
        self.use_csr = use_csr

    def compute(self, a):
        data = _to_dense(a).astype(np.float32)

        if self.threshold is not None and self.threshold > 0:
            mask = np.abs(data) >= self.threshold
            data = data * mask

        if self.use_csr and nd.csr().enabled():
            return nd.array(data, device=nd.csr())
        return data

    def gradient(self, out_grad, node):
        return (sparse_to_dense(out_grad),)


class SparseToDense(TensorOp):
    """Convert SparseTensor back to a dense Tensor."""

    def compute(self, a):
        dense = _to_dense(a).astype(np.float32)
        return nd.array(dense)

    def gradient(self, out_grad, node):
        return (dense_to_sparse(out_grad),)
