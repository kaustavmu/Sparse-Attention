"""Core data structures."""
import needle
from typing import Any, Dict, List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
from needle import init

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

from .backend_selection import array_api, NDArray, Device, cpu, all_devices

SparseNDArray = Any

class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)

class SparseTensorOp(Op):
    """Operator specialized for sparse tensor operations."""

    def __call__(self, *args):
        return SparseTensor.make_from_op(self, args)

    def compute(self, *args: Tuple["SparseNDArray"]):
        """Compute forward pass using sparse array operations."""
        raise NotImplementedError()

    def gradient(
        self, out_grad: "SparseTensor", node: "SparseTensor"
    ) -> Union["SparseTensor", Tuple["SparseTensor"]]:
        """Compute partial adjoint(s) for each sparse input."""
        raise NotImplementedError()

class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


### Not needed in HW1
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        dev = getattr(data, "device", None)
        if isinstance(dev, str):
            # Normalized string labels to backend devices.
            if dev.lower() == "cpu":
                return cpu()
            if dev.lower().startswith("cuda"):
                return needle.cuda()
        if dev is None:
            return cpu()
        return dev

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)



    __radd__ = __add__
    __rmul__ = __mul__

class SparseTensor(Value):
    """A value node representing a sparse tensor in the computation graph."""

    def __init__(self, sparse_array, *, requires_grad=True):
        # Optionally validate it’s a supported sparse format
        assert hasattr(sparse_array, "shape"), "Must have shape attribute"
        self._init(
            None,
            [],
            cached_data=sparse_array,
            requires_grad=requires_grad,
        )

    def backward(self, out_grad=None):
        """
        Backprop starting from this SparseTensor.
        If out_grad is None, assume scalar output -> grad = 1.
        """
        from . import ops  # DO NOT import ones_like

        if out_grad is None:
            # Create a scalar 1 - convert to appropriate format
            cached = self.realize_cached_data()
            import numpy as np
            if hasattr(cached, "numpy"):
                # It's an NDArray, create scalar NDArray
                try:
                    from needle.backend_ndarray import ndarray_sparse as nd
                    device = cached.device if hasattr(cached, 'device') else nd.default_device()
                    scalar_nd = nd.NDArray(np.array([1.0], dtype=np.float32), device=device)
                    one = SparseTensor(scalar_nd, requires_grad=False)
                except ImportError:
                    # Fallback to numpy
                    one = SparseTensor(np.array([1.0], dtype=np.float32), requires_grad=False)
            else:
                # It's a numpy array, create scalar numpy array
                one = SparseTensor(np.array([1.0], dtype=np.float32), requires_grad=False)
            out_grad = one

        compute_gradient_of_variables(self, out_grad)

    def sum(self, axes=None):
        return needle.sparse_ops.Summation(axes)(self)
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        return needle.sparse_ops.SparseMatMul()(self, other)
    
    def matmul(self, other):
        """Matrix multiplication."""
        return needle.sparse_ops.SparseMatMul()(self, other)
    
    def broadcast_to(self, shape):
        """Broadcast sparse tensor to new shape."""
        cached = self.realize_cached_data()
        if hasattr(cached, "broadcast_to"):
            # It's an NDArray, use its broadcast_to
            return SparseTensor.make_const(cached.broadcast_to(shape))
        else:
            # It's a numpy array, use numpy's broadcast_to
            import numpy as np
            broadcasted = np.broadcast_to(cached, shape)
            return SparseTensor.make_const(broadcasted)
    
    def reshape(self, shape):
        """Reshape sparse tensor to new shape."""
        cached = self.realize_cached_data()
        if hasattr(cached, "reshape"):
            # It's an NDArray, use its reshape
            return SparseTensor.make_const(cached.reshape(shape))
        else:
            # It's a numpy array, use numpy's reshape
            import numpy as np
            reshaped = np.reshape(cached, shape)
            return SparseTensor.make_const(reshaped)
    
    def transpose(self, axes=None):
        """Transpose sparse tensor. For 2D matrices, transposes rows and columns."""
        # Use SparseTranspose op for 2D matrices
        # If axes is provided, we could use permute for NDArrays, but for now
        # we'll just transpose 2D matrices (which is the common case)
        if axes is not None and axes != (1, 0) and axes != [1, 0]:
            # For non-standard axes, use the underlying array's transpose/permute
            cached = self.realize_cached_data()
            if hasattr(cached, "permute"):
                # It's an NDArray, use permute with the axes
                if isinstance(axes, (list, tuple)) and len(axes) == 2:
                    return SparseTensor.make_const(cached.permute(tuple(axes)))
            # Fallback to numpy transpose
            import numpy as np
            transposed = np.transpose(cached, axes)
            return SparseTensor.make_const(transposed)
        # Default: use SparseTranspose for 2D transpose
        return needle.sparse_ops.SparseTranspose()(self)
    
    @property
    def shape(self):
        return self.realize_cached_data().shape

    def detach(self):
        """Detach sparse tensor from computation graph."""
        return SparseTensor.make_const(self.realize_cached_data())

    @staticmethod
    def make_from_op(op: Op, inputs: List["SparseTensor"]):
        tensor = SparseTensor.__new__(SparseTensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = SparseTensor.__new__(SparseTensor)
        tensor._init(None, [], cached_data=data, requires_grad=requires_grad)
        return tensor

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    # node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # We allow both Tensor and SparseTensor
    node_to_output_grads_list: Dict[Any, List[Any]] = {}

    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for i in reverse_topo_order:
        if getattr(i, "requires_grad", False):

            grads = sum_node_list(node_to_output_grads_list[i])
            i.grad = grads

            if i.op is not None:
                in_grads = i.op.gradient_as_tuple(grads, i)

                for j, n in enumerate(i.inputs):
                    if getattr(n, "requires_grad", False):
                        if n not in node_to_output_grads_list:
                            node_to_output_grads_list[n] = []
                        node_to_output_grads_list[n].append(in_grads[j])

    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Return a topological order of the computation graph."""
    visited = set()
    topo_order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for inp in node.inputs:
            dfs(inp)
        topo_order.append(node)

    for node in node_list:
        dfs(node)

    return topo_order



def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for inp in node.inputs:
        topo_sort_dfs(inp, visited, topo_order)
    topo_order.append(node)


##############################
####### Helper Methods #######
##############################

def sum_node_list(grads):
    """Sum a list of grad nodes (Tensor or SparseTensor)."""
    if len(grads) == 0:
        raise ValueError("No gradients to sum")

    out = grads[0]
    for g in grads[1:]:
        out = out + g
    return out
