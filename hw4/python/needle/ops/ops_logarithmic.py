from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        M = array_api.max(Z, axis=1, keepdims=True)
        Z_shifted = Z - M
        Z_exp = array_api.exp(Z_shifted)
        Z_sum = array_api.sum(Z_exp, axis=1, keepdims=True)
        logsumexp_result = array_api.log(Z_sum) + M
        return Z - logsumexp_result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        logsumexp_Z = logsumexp(Z, (1,))
        logsumexp_Z_broadcasted = broadcast_to(reshape(logsumexp_Z, (Z.shape[0], 1)), Z.shape)
        softmax_probs = exp(Z - logsumexp_Z_broadcasted)
        
        out_grad_sum = summation(out_grad, axes=(1,))
        out_grad_sum_broadcasted = broadcast_to(reshape(out_grad_sum, (Z.shape[0], 1)), Z.shape)
        
        return (out_grad - softmax_probs * out_grad_sum_broadcasted,)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # Reduce over all axes - M will be a scalar
            M = array_api.max(a)
            # For scalar operations, we can subtract directly since M is a scalar
            # But we need to ensure M is treated as a scalar value, not an array
            M_scalar = M.numpy().item()  # Extract scalar value
            a_shifted = a - M_scalar     # This uses scalar subtraction
            a_exp = array_api.exp(a_shifted)
            a_sum = array_api.sum(a_exp)
            return array_api.log(a_sum) + M_scalar
        else:
            # Handle single axis only
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            axis = axes[0]
            
            # Get max with keepdims=True for broadcasting
            M = array_api.max(a, axis=axis, keepdims=True)
            
            # Manually broadcast M to match a's shape for subtraction
            M_broadcasted = array_api.broadcast_to(M, a.shape)
            
            # Now we can subtract since shapes match exactly
            a_shifted = a - M_broadcasted
            a_exp = array_api.exp(a_shifted)
            a_sum = array_api.sum(a_exp, axis=axis, keepdims=True)
            log_sum = array_api.log(a_sum)
            result_with_keepdims = log_sum + M
            
            # Remove the keepdims dimension to get final result
            final_result = array_api.sum(result_with_keepdims, axis=axis)
            return final_result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        logsumexp_output = node
        
        if self.axes is None:
            broadcasted_logsumexp = broadcast_to(logsumexp_output, a.shape)
            result = out_grad * exp(a - broadcasted_logsumexp)
        else:
            axes_list = list(self.axes) if isinstance(self.axes, tuple) else [self.axes]
            
            # Expand logsumexp output to match input shape
            expanded_shape = list(a.shape)
            for axis in axes_list:
                expanded_shape[axis] = 1
            logsumexp_reshaped = reshape(logsumexp_output, [expanded_shape[i] for i in range(len(expanded_shape)) if i not in axes_list])
            
            # Add back the summed dimensions
            current_tensor = logsumexp_reshaped
            remaining_axes = sorted([i for i in range(len(a.shape)) if i not in axes_list])
            insert_pos = 0
            
            final_shape = [1] * len(a.shape)
            for i, axis in enumerate(remaining_axes):
                final_shape[axis] = current_tensor.shape[i]
            
            logsumexp_broadcasted = broadcast_to(reshape(current_tensor, final_shape), a.shape)
            softmax_probs = exp(a - logsumexp_broadcasted)
            
            # Expand out_grad similarly
            out_grad_final_shape = [1] * len(a.shape)
            for i, axis in enumerate(remaining_axes):
                out_grad_final_shape[axis] = out_grad.shape[i]
            
            out_grad_broadcasted = broadcast_to(reshape(out_grad, out_grad_final_shape), a.shape)
            result = out_grad_broadcasted * softmax_probs
        
        return (result,)

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)