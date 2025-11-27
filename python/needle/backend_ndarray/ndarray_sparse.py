# ndarray.py  -- CSR-aware NDArray wrapper
import math
import operator
from functools import reduce
from typing import Any, Callable, Iterable, Union

import numpy as np

from . import ndarray_backend_numpy
# ndarray_backend_cpu and ndarray_backend_csr are imported conditionally in their respective functions


# math.prod not in Python 3.7
def prod(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name: str, mod: Any) -> None:
        self.name: str = name
        self.mod: Any = mod

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackendDevice) and self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mod, name)

    def enabled(self) -> bool:
        return self.mod is not None

    def randn(self, *shape: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n: int, i: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape: tuple[int, ...], dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr

    def ones(self, *shape: int, dtype: str = "float32") -> "NDArray":
        """Allocate an array of ones on this device."""
        return self.full(shape, 1.0, dtype)

    def zeros(self, *shape: int, dtype: str = "float32") -> "NDArray":
        """Allocate an array of zeros on this device."""
        return self.full(shape, 0.0, dtype)


def cuda() -> BackendDevice:
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda  # type: ignore[attr-defined]

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy() -> BackendDevice:
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu() -> BackendDevice:
    """Return cpu device"""
    try:
        from . import ndarray_backend_cpu  # type: ignore[attr-defined]
        return BackendDevice("cpu", ndarray_backend_cpu)
    except ImportError:
        return BackendDevice("cpu", None)

def csr() -> BackendDevice:
    """Return CSR sparse device"""
    try:
        from . import ndarray_backend_csr  # type: ignore[attr-defined]
        return BackendDevice("csr", ndarray_backend_csr)
    except ImportError:
        return BackendDevice("csr", None)

def default_device() -> BackendDevice:
    return cpu_numpy()


def all_devices() -> list[BackendDevice]:
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy(), csr()]


def _is_csr_device(device: BackendDevice) -> bool:
    """Detect CSR backend by module attribute -- backend should set __device_name__ = 'csr'."""
    try:
        return getattr(device.mod, "__device_name__", "") == "csr"
    except Exception:
        return False


class NDArray:
    """A generic ND array class that may contain multiple different backends
    (Numpy, CPU native, GPU, CSR sparse backend).  This class contains only
    the functions required by the homework / experiments.
    """

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _device: BackendDevice
    _handle: Any

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # creates a copy on target device
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            # backend-specific from_numpy populates the backend Array handle
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # try to create a numpy array and delegate
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray") -> None:
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Utility function to compute compact strides (element strides)."""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        device: BackendDevice | None = None,
        handle: Any = None,
        offset: int = 0,
    ) -> "NDArray":
        """Create a new NDArray with the given properties. Allocates memory if handle=None."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def dtype(self) -> str:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)

    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self) -> str:
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: BackendDevice) -> "NDArray":
        """Convert between devices, using to/from numpy as the bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """Convert to a dense numpy array (makes a copy when needed)."""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def __array__(self, dtype=None, copy=None):
        """Enable implicit conversion to numpy via np.asarray."""
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            return arr.copy()
        return arr

    def is_compact(self) -> bool:
        """Return true if array is compact in memory and internal size equals product of shape."""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self) -> "NDArray":
        """Convert a matrix to be compact (dense backends). For CSR backend this will call backend.compact
        which should produce a CSR representation for the full view. """
        # For CSR backend don't force a densify; allow backend to produce a compact handle
        out = NDArray.make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> "NDArray":
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self) -> "NDArray":
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        if prod(self.shape) != prod(new_shape):
            raise ValueError("Total size of new array must be unchanged")
        # Some backends (CSR) can reshape by just updating shape metadata if they support it.
        if not self.is_compact():
            # require compact for reshape on dense backends; for csr backend, device.compact was used earlier to get a compact handle
            raise ValueError("Reshape only supported for compact arrays")
        return NDArray.make(new_shape, device=self.device, handle=self._handle)

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        if sorted(new_axes) != list(range(self.ndim)):
            raise ValueError("new_axes must be a permutation of the existing axes")
        new_shape = tuple(self.shape[i] for i in new_axes)
        new_strides = tuple(self.strides[i] for i in new_axes)
        return NDArray.make(new_shape, strides=new_strides, device=self.device, handle=self._handle, offset=self._offset)

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        if len(new_shape) != self.ndim:
            raise ValueError("new_shape must have the same number of dimensions as the original shape")
        new_strides = []
        for i in range(self.ndim):
            if self.shape[i] == new_shape[i]:
                new_strides.append(self.strides[i])
            elif self.shape[i] == 1:
                new_strides.append(0)
            else:
                raise ValueError(f"Cannot broadcast dimension {i} from size {self.shape[i]} to size {new_shape[i]}")
        return NDArray.make(new_shape, strides=tuple(new_strides), device=self.device, handle=self._handle, offset=self._offset)

    ### Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim] + start
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = []
        new_strides = []
        new_offset = self._offset
        for i, s in enumerate(slices):
            start, stop, step = s.start, s.stop, s.step
            if stop <= start:
                raise ValueError(f"Slice {s} has non-positive size")
            if step <= 0:
                raise ValueError(f"Slice {s} has non-positive step")
            new_shape.append((stop - start + step - 1) // step)
            new_strides.append(self.strides[i] * step)
            new_offset += self.strides[i] * start
        return NDArray.make(tuple(new_shape), strides=tuple(new_strides), device=self.device, handle=self._handle, offset=new_offset)

    def __setitem__(self, idxs: int | slice | tuple[int | slice, ...], other: Union["NDArray", float]) -> None:
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            # If both are on CSR backend, avoid compacting; else compact before set.
            if _is_csr_device(self.device) and _is_csr_device(other.device) and self.device == other.device:
                self.device.ewise_setitem(
                    other._handle,
                    view._handle,
                    view.shape,
                    view.strides,
                    view._offset,
                )
            else:
                self.device.ewise_setitem(
                    other.compact()._handle,
                    view._handle,
                    view.shape,
                    view.strides,
                    view._offset,
                )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Elementwise and scalar wrapper that routes to backend efficiently
    def ewise_or_scalar(
        self,
        other: Union["NDArray", float],
        ewise_func: Callable[[Any, Any, Any], None],
        scalar_func: Callable[[Any, Any, Any], None],
    ) -> "NDArray":
        """Run elementwise or scalar function. Specialized for CSR backend to avoid unnecessary compaction."""
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            # If both operands are CSR on same device, call backend functions directly with their handles
            if _is_csr_device(self.device) and self.device == other.device:
                ewise_func(self._handle, other._handle, out._handle)
            else:
                ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            # scalar path: for CSR backends we may still want to operate on CSR handle directly
            if _is_csr_device(self.device):
                scalar_func(self._handle, other, out._handle)
            else:
                scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: Union["NDArray", float]) -> "NDArray":
        return self + (-other)

    def __rsub__(self, other: Union["NDArray", float]) -> "NDArray":
        return other + (-self)

    def __mul__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self) -> "NDArray":
        return self * (-1)

    def __pow__(self, other: float) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        if _is_csr_device(self.device):
            self.device.scalar_power(self._handle, other, out._handle)
        else:
            self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary comparators (return binary float arrays)
    def __eq__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: Any) -> "NDArray":
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return 1 - (self == other)

    def __gt__(self, other: Any) -> "NDArray":
        return (self >= other) * (self != other)

    def __lt__(self, other: Any) -> "NDArray":
        return 1 - (self >= other)

    def __le__(self, other: Any) -> "NDArray":
        return 1 - (self > other)

    ### Elementwise functions
    def log(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        if _is_csr_device(self.device):
            self.device.ewise_log(self._handle, out._handle)
        else:
            self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        if _is_csr_device(self.device):
            self.device.ewise_exp(self._handle, out._handle)
        else:
            self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        if _is_csr_device(self.device):
            self.device.ewise_tanh(self._handle, out._handle)
        else:
            self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other: "NDArray") -> "NDArray":
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # If CSR backend present and both arrays are on the same CSR device,
        # call backend.matmul directly without densifying.
        if _is_csr_device(self.device) and self.device == other.device:
            out = NDArray.make((m, p), device=self.device)
            # the csr backend's matmul is implemented to handle CSR @ dense, or CSR@CSR (fallback)
            self.device.matmul(self._handle, other._handle, out._handle, m, n, p)
            return out

        # If only lhs is CSR and rhs is dense on same device, still call matmul directly
        if _is_csr_device(self.device) and self.device == other.device:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(self._handle, other._handle, out._handle, m, n, p)
            return out

        # tiled path (existing logic) for dense CPU backends
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):
            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            # default: compact both and call matmul
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions
    def reduce_view_out(self, axis: int | tuple[int, ...] | list[int] | None, keepdims: bool = False) -> tuple["NDArray", "NDArray"]:
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            # reduce all -> reshape to (1, prod(shape))
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        # For CSR backend we can possibly avoid densifying if backend supports reduce on CSR handle
        if _is_csr_device(self.device):
            self.device.reduce_sum(view._handle, out._handle, view.shape[-1])
        else:
            self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        if _is_csr_device(self.device):
            self.device.reduce_max(view._handle, out._handle, view.shape[-1])
        else:
            self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def reduce_sum(self, axes=None):
        # Use the axes-based reduce_sum for CSR backend
        if _is_csr_device(self.device):
            result = self.device.reduce_sum_axes(self._handle, axes)
            # Convert numpy result back to NDArray if needed
            if isinstance(result, np.ndarray):
                return NDArray(result, device=self.device)
            else:
                # Scalar result
                return NDArray(np.array([result], dtype=np.float32), device=self.device)
        else:
            # For dense backends, they don't have reduce_sum with axes parameter
            # Convert to numpy and use numpy's sum
            return NDArray(self.numpy().sum(axis=axes), device=self.device)


# Convenience top-level functions similar to numpy
def array(a: Any, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape: tuple[int, ...], dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape: tuple[int, ...], fill_value: float, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray | float) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return a.log()


def exp(a: NDArray) -> NDArray:
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a: NDArray, axis: int | tuple[int] | list[int] | None = None, keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)
