from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp, SparseTensor, SparseTensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from ..ops import sparse_ops as sops

from ..backend_selection import array_api, BACKEND 

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    norm = []
    for ax in axes:
        norm.append(ax if ax >= 0 else ndim + ax)
    return tuple(sorted(norm))


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # maxi = array_api.max(Z, axis=-1, keepdims=True)
        maxi = Z.max(axis=-1, keepdims=True)
        Z_shifted = Z - maxi
        exp_Z = array_api.exp(Z_shifted)
        sum_exp_Z = array_api.sum(exp_Z, axis=-1, keepdims=True)
        return Z_shifted - array_api.log(sum_exp_Z)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        log_softmax_Z = node
        softmax = exp(log_softmax_Z)
        sum_out_grad = summation(out_grad, axes=(-1,))
        # reshape to (batch, 1) so it broadcasts correctly
        sum_out_grad = reshape(sum_out_grad, out_grad.shape[:-1] + (1,))
        return out_grad - sum_out_grad * softmax
    
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    if isinstance(a, SparseTensor):
        return SparseLogSoftmax()(a)
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        axes = self.axes
        maxi = Z.max(axis=axes, keepdims=True)

        if axes is None:
            reshaped_maxi = maxi.reshape(tuple(1 for _ in Z.shape))
        else:
            reshaped_maxi = maxi

        broadcast_maxi = reshaped_maxi.broadcast_to(Z.shape)
        shifted = Z - broadcast_maxi
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=axes, keepdims=True)
        log_sum = sum_exp.log()
        result = log_sum + reshaped_maxi

        if axes is None:
            return result.reshape((1,))

        axes_tuple = _normalize_axes(axes, Z.ndim)
        new_shape = [dim for idx, dim in enumerate(result.shape) if idx not in axes_tuple]
        if not new_shape:
            new_shape = (1,)
        else:
            new_shape = tuple(new_shape)
        return result.reshape(new_shape)
    
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        # Normalize axes
        if self.axes is None:
            axes = tuple(range(len(z.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = tuple(self.axes)

        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    if isinstance(a, SparseTensor):
        return SparseLogSumExp(axes=axes)(a)
    return LogSumExp(axes=axes)(a)


class SparseLogSoftmax(SparseTensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        maxi = Z.max(axis=-1, keepdims=True)
        shifted = Z - maxi
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
        return shifted - sum_exp.log()

    def gradient(self, out_grad: SparseTensor, node: SparseTensor):
        out_data = out_grad.realize_cached_data()
        log_softmax_data = node.realize_cached_data()
        softmax = log_softmax_data.exp()
        sum_out = out_data.sum(axis=-1, keepdims=True)
        broadcast_sum = sum_out.broadcast_to(out_data.shape)
        grad_data = out_data - broadcast_sum * softmax
        return SparseTensor.make_const(grad_data)


class SparseLogSumExp(SparseTensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        axes = self.axes
        maxi = Z.max(axis=axes, keepdims=True)
        shifted = Z - maxi
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=axes, keepdims=True)
        log_sum = sum_exp.log() + maxi
        if axes is None:
            return log_sum.reshape((1,))
        axes_tuple = _normalize_axes(axes, Z.ndim)
        out_shape = [dim for idx, dim in enumerate(Z.shape) if idx not in axes_tuple]
        if not out_shape:
            out_shape = (1,)
        else:
            out_shape = tuple(out_shape)
        return log_sum.reshape(out_shape)

    def gradient(self, out_grad: SparseTensor, node: SparseTensor):
        z_data = node.inputs[0].realize_cached_data()
        axes_tuple = _normalize_axes(self.axes, z_data.ndim) if self.axes is not None else tuple(range(z_data.ndim))

        max_vals = z_data.max(axis=self.axes, keepdims=True)
        exp_z = (z_data - max_vals).exp()
        sum_exp = exp_z.sum(axis=self.axes)

        out_data = out_grad.realize_cached_data()
        grad_sum = out_data / sum_exp

        expand_shape = list(z_data.shape)
        for axis in axes_tuple:
            expand_shape[axis] = 1
        grad_expand = grad_sum.reshape(tuple(expand_shape)).broadcast_to(z_data.shape)
        grad_data = grad_expand * exp_z
        return SparseTensor.make_const(grad_data)