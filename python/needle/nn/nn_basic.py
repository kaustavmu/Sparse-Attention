"""The module.
"""
from typing import Any
from needle.autograd import Tensor, SparseTensor
from needle import ops
from needle.ops import sparse_ops as sops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _ensure_dense(x):
    if isinstance(x, SparseTensor):
        return sops.sparse_to_dense(x)
    return x


def _ensure_sparse(x, threshold=0.0):
    if isinstance(x, SparseTensor):
        return x
    if isinstance(x, Tensor):
        use_csr = len(x.shape) <= 2
        return sops.dense_to_sparse(x, threshold=threshold, use_csr=use_csr)
    return x


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x



class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
        use_sparse: bool = True,
        sparse_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_sparse = use_sparse
        self.sparse_threshold = sparse_threshold

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).reshape((1, out_features)))
        else:
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.use_sparse:
            x_sparse = _ensure_sparse(X, threshold=self.sparse_threshold)
            w_sparse = _ensure_sparse(self.weight, threshold=0.0)
            out = _ensure_dense(sops.sparse_matmul(x_sparse, w_sparse))
        else:
            dense_input = _ensure_dense(X)
            out = dense_input.matmul(self.weight)
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        new_shape = (X.shape[0], int(np.prod(X.shape[1:])))
        if isinstance(X, SparseTensor):
            return _ensure_dense(X.reshape(new_shape))
        dense_x = _ensure_dense(X)
        return ops.reshape(dense_x, new_shape)
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        dense_x = _ensure_dense(x)
        return ops.tanh(dense_x)
        ### END YOUR SOLUTION
        
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if isinstance(x, SparseTensor):
            return _ensure_dense(sops.sparse_relu(x))
        dense_x = _ensure_dense(x)
        return ops.relu(dense_x)    
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        dense_logits = _ensure_dense(logits)
        one_hot_y = init.one_hot(
            dense_logits.shape[1],
            y,
            device=dense_logits.device,
            dtype=dense_logits.dtype,
        )
        return (
            ops.summation(ops.logsumexp(dense_logits, (1,)) / dense_logits.shape[0])
            - ops.summation(one_hot_y * dense_logits / dense_logits.shape[0])
        )
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = _ensure_dense(x)

        if self.training:
            mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean = ops.reshape(mean, (1, mean.shape[0]))
            mean = mean.broadcast_to(x.shape)

            var = ops.summation((x - mean) ** 2, axes=(0,)) / x.shape[0]
            var = ops.reshape(var, (1, var.shape[0]))
            var = var.broadcast_to(x.shape)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * ops.reshape(ops.summation(x, axes=(0,)) / x.shape[0], self.running_mean.shape).detach()
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * ops.reshape(ops.summation((x - mean) ** 2, axes=(0,)) / x.shape[0], self.running_var.shape).detach()
            
            weight = ops.reshape(self.weight, (1, self.dim)).broadcast_to(x.shape)
            bias = ops.reshape(self.bias, (1, self.dim)).broadcast_to(x.shape)
            return weight * x_normalized + bias
        
        else:
            mean = ops.reshape(self.running_mean.detach(), (1, self.running_mean.shape[0])).broadcast_to(x.shape)
            var = ops.reshape(self.running_var.detach(), (1, self.running_var.shape[0])).broadcast_to(x.shape)
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
            weight = ops.reshape(self.weight, (1, self.dim)).broadcast_to(x.shape)
            bias = ops.reshape(self.bias, (1, self.dim)).broadcast_to(x.shape)
            return weight * x_normalized + bias
    
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        x = _ensure_dense(x)
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = _ensure_dense(x)
        assert len(x.shape) == 2 and x.shape[1] == self.dim

        n = x.shape[0]
        w = ops.reshape(self.weight, (1, self.dim)).broadcast_to((n, self.dim)) # (n, d)
        b = ops.reshape(self.bias, (1, self.dim)).broadcast_to((n, self.dim)) # (n, d)

        mu = (x.sum(axes=(1,)) / self.dim).reshape((n, 1)) # (n, 1)
        mu = ops.broadcast_to(mu, (n, self.dim)) # (n, d)

        var = ((x - mu) ** 2).sum(axes=(1,)) / self.dim # (n,)
        std = ((var + self.eps) ** 0.5).reshape((n, 1)) # (n, 1)
        std = ops.broadcast_to(std, (n, self.dim)) # (n, d)

        return w * ((x - mu) / std) + b
        ### END YOUR SOLUTION
        
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        dense_x = _ensure_dense(x)
        if self.training:
            mask = init.randb(*dense_x.shape, p=1 - self.p, device=dense_x.device, dtype=dense_x.dtype)
            return (dense_x * mask) / (1 - self.p)
        else:
            return dense_x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return _ensure_dense(self.fn(x)) + _ensure_dense(x)
        ### END YOUR SOLUTION
