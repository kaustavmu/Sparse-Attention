# ndarray_backend_csr.py
import numpy as np

__device_name__ = "csr"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    """
    CSR representation container. Constructor accepts 'size' for compatibility
    with NDArray.make(size) calls; actual matrix shape is tracked when converting
    from / to dense (the backend functions receive shape argument).
    """

    def __init__(self, size):
        # size is total number of elements (rows * cols) — we store to satisfy NDArray expectations
        self._size = size
        # CSR components (empty by default)
        self.indptr = np.array([0], dtype=np.int64)
        self.indices = np.array([], dtype=np.int64)
        self.data = np.array([], dtype=_datatype)
        # store last known shape for convenience (updated on from_numpy or other ops)
        self.shape = None

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return f"CSRArray(shape={self.shape}, nnz={self.data.size})"

    @property
    def dense_flat(self):
        """Return dense flattened numpy array (makes a copy)."""
        if self.shape is None:
            # unknown shape -> return zeros
            return np.zeros(self._size, dtype=_datatype)
        dense = np.zeros(self.shape, dtype=_datatype)
        rows = self.shape[0]
        for r in range(rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            cols = self.indices[start:end]
            vals = self.data[start:end]
            dense[r, cols] = vals
        return dense.reshape(-1)


# -------------------- helpers --------------------

def _csr_from_dense(arr: np.ndarray):
    """Convert 2D numpy array -> CSR arrays (indptr, indices, data)"""
    assert arr.ndim == 2
    rows, cols = arr.shape
    data_list = []
    idx_list = []
    indptr = [0]
    nnz = 0
    for r in range(rows):
        row = arr[r]
        nz_cols = np.nonzero(row)[0]
        if nz_cols.size > 0:
            idx_list.append(nz_cols.astype(np.int64))
            data_list.append(row[nz_cols].astype(_datatype))
            nnz += nz_cols.size
        indptr.append(nnz)
    if nnz == 0:
        indices = np.array([], dtype=np.int64)
        data = np.array([], dtype=_datatype)
    else:
        indices = np.concatenate(idx_list).astype(np.int64) if idx_list else np.array([], dtype=np.int64)
        data = np.concatenate(data_list).astype(_datatype) if data_list else np.array([], dtype=_datatype)
    indptr = np.array(indptr, dtype=np.int64)
    return indptr, indices, data


def _dense_from_csr(indptr, indices, data, shape):
    """Convert CSR -> dense numpy array with given shape (2D)."""
    rows, cols = shape
    dense = np.zeros((rows, cols), dtype=_datatype)
    for r in range(rows):
        start = indptr[r]
        end = indptr[r + 1]
        if end > start:
            dense[r, indices[start:end]] = data[start:end]
    return dense


def _as_2d(shape):
    """Return (rows, cols) from NDArray shape expected to be 2D."""
    if len(shape) != 2:
        raise ValueError("CSR backend expects 2D shapes for dense<->csr conversions")
    return shape[0], shape[1]


# -------------------- API required by NDArray --------------------

def to_numpy(a: Array, shape, strides, offset):
    """
    Return a dense numpy view for the requested shape/strides/offset.
    We'll produce a dense array (copy) and then use numpy stride_tricks to
    create desired view (mimicking the dense backend).
    """
    rows, cols = _as_2d(shape) if len(shape) == 2 else (shape[0], int(np.prod(shape[1:])))
    dense = _dense_from_csr(a.indptr, a.indices, a.data, (rows, cols))
    # flatten then return as_strided similar to dense backend
    flat = dense.reshape(-1)
    # offset is element offset into flattened array; strides are element-strides
    # convert element-strides into byte-strides for as_strided
    byte_strides = tuple([s * _datetype_size for s in strides])
    return np.lib.stride_tricks.as_strided(flat[offset:], shape, byte_strides)


def from_numpy(a: np.ndarray, out: Array):
    """Convert a dense numpy array into CSR stored in out."""
    arr = np.ascontiguousarray(a)
    # handle general shapes: treat arr as 2D (rows x cols)
    if arr.ndim == 1:
        # interpret as single-row matrix
        rows = 1
        cols = arr.shape[0]
        arr2 = arr.reshape(1, -1)
    elif arr.ndim == 2:
        arr2 = arr
        rows, cols = arr2.shape
    else:
        # flatten trailing dims into columns
        rows = arr.shape[0]
        cols = int(np.prod(arr.shape[1:]))
        arr2 = arr.reshape(rows, cols)
    indptr, indices, data = _csr_from_dense(arr2)
    out.indptr = indptr
    out.indices = indices
    out.data = data
    out.shape = (rows, cols)
    out._size = rows * cols


def fill(out: Array, val):
    """Fill the entire array slice with scalar val. If val == 0 -> empty CSR."""
    if val == 0:
        out.indptr = np.array([0], dtype=np.int64)
        out.indices = np.array([], dtype=np.int64)
        out.data = np.array([], dtype=_datatype)
        # shape remains whatever it was; if unknown, keep None
    else:
        if out.shape is None:
            # cannot choose shape; treat as single row of length size
            rows = 1
            cols = out._size
            out.shape = (rows, cols)
        rows, cols = out.shape
        dense = np.full((rows, cols), val, dtype=_datatype)
        indptr, indices, data = _csr_from_dense(dense)
        out.indptr, out.indices, out.data = indptr, indices, data


def compact(a: Array, out: Array, shape, strides, offset):
    """
    Compact: produce a compact representation (CSR is already compact).
    We'll interpret a with the requested view -> dense -> CSR into out.
    """
    dense_view = to_numpy(a, shape, strides, offset).copy()
    # dense_view has shape==shape; reshape into 2D if needed
    if len(shape) == 1:
        rows, cols = 1, shape[0]
        dense2 = dense_view.reshape(1, -1)
    elif len(shape) == 2:
        rows, cols = shape
        dense2 = dense_view.reshape(rows, cols)
    else:
        rows = shape[0]
        cols = int(np.prod(shape[1:]))
        dense2 = dense_view.reshape(rows, cols)
    indptr, indices, data = _csr_from_dense(dense2)
    out.indptr, out.indices, out.data = indptr, indices, data
    out.shape = (rows, cols)
    out._size = rows * cols


def ewise_setitem(a: Array, out: Array, shape, strides, offset):
    """
    Set the subarray specified by (out, shape, strides, offset) to the dense
    values represented by CSR a. We'll convert the out region to dense, reshape
    into 2D, place the values, then convert back to CSR.
    """
    # get dense view of out (as numpy array) then assign
    dense_out = to_numpy(out, shape, strides, offset)
    # a.dense_flat is a copy; reshape appropriately
    a_dense = _dense_from_csr(a.indptr, a.indices, a.data, (shape[0], int(np.prod(shape[1:]))) if len(shape) > 1 else (1, shape[0]))
    # Now assign values into the view
    dense_out[:] = a_dense.reshape(-1)
    # Finally write dense_out back into CSR for the region (we will write to out's whole shape)
    # If shape equals entire array shape and offset==0 and compact strides, then replace out entirely
    if offset == 0 and strides == tuple([1] * len(shape)):
        from_numpy(dense_out.reshape(shape), out)
    else:
        # perform full-densify of out and then write - simpler but correct
        full_dense = to_numpy(out, out.shape, out.shape, 0).copy()
        full_dense.reshape(shape)[:] = dense_out.reshape(shape)
        from_numpy(full_dense.reshape(shape), out)


def scalar_setitem(size, val, out: Array, shape, strides, offset):
    """Set the specified view in out to scalar val."""
    # create target dense view and set to val, then write back entire out
    dense_out = to_numpy(out, shape, strides, offset)
    dense_out[:] = val
    # write back to full out
    full_dense = to_numpy(out, out.shape, out.shape, 0).copy()
    # fill the slice within full_dense: we need to map offset & strides to indices; easiest is
    # to reshape view and broadcast assignment (we already changed dense_out view which references full_dense's memory? to_numpy built a copy, so simpler to build full_dense then assign)
    # Build indices for the flattened view:
    # For simplicity: rebuild full dense from current csr -> then overwrite region
    full = _dense_from_csr(out.indptr, out.indices, out.data, out.shape) if out.shape is not None else np.zeros(out._size, dtype=_datatype)
    # Now compute the region in full to set:
    flat_view = np.lib.stride_tricks.as_strided(full.reshape(-1)[offset:], shape, tuple([s * _datetype_size for s in strides]))
    flat_view[:] = val
    from_numpy(full.reshape(out.shape), out)


# -------------------- elementwise and scalar ops --------------------

def _ensure_same_shape_csr(a: Array, b: Array):
    if a.shape is None or b.shape is None or a.shape != b.shape:
        raise ValueError("CSR arrays must have same tracked shape for elementwise ops or be converted from compatible dense arrays.")


def _dense_binary_op_from_csr(func, a: Array, b: Array, out: Array):
    """Apply a binary op (numpy ufunc-like) by densifying both args and writing result CSR to out."""
    if a.shape is None or b.shape is None:
        # densify both using recorded sizes
        a_dense = _dense_from_csr(a.indptr, a.indices, a.data, (1, a._size))
        b_dense = _dense_from_csr(b.indptr, b.indices, b.data, (1, b._size))
    else:
        a_dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
        b_dense = _dense_from_csr(b.indptr, b.indices, b.data, b.shape)
    res = func(a_dense, b_dense)
    from_numpy(res, out)


def ewise_add(a, b, out):
    _dense_binary_op_from_csr(np.add, a, b, out)


def scalar_add(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = dense + val
    from_numpy(res, out)


def ewise_mul(a, b, out):
    _dense_binary_op_from_csr(np.multiply, a, b, out)


def scalar_mul(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = dense * val
    from_numpy(res, out)


def ewise_div(a, b, out):
    _dense_binary_op_from_csr(np.divide, a, b, out)


def scalar_div(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = dense / val
    from_numpy(res, out)


def scalar_power(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = dense ** val
    from_numpy(res, out)


def ewise_maximum(a, b, out):
    _dense_binary_op_from_csr(np.maximum, a, b, out)


def scalar_maximum(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = np.maximum(dense, val)
    from_numpy(res, out)


def ewise_eq(a, b, out):
    _dense_binary_op_from_csr(lambda x, y: (x == y).astype(_datatype), a, b, out)


def scalar_eq(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = (dense == val).astype(_datatype)
    from_numpy(res, out)


def ewise_ge(a, b, out):
    _dense_binary_op_from_csr(lambda x, y: (x >= y).astype(_datatype), a, b, out)


def scalar_ge(a, val, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = (dense >= val).astype(_datatype)
    from_numpy(res, out)


def ewise_log(a, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = np.log(dense)
    from_numpy(res, out)


def ewise_exp(a, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = np.exp(dense)
    from_numpy(res, out)


def ewise_tanh(a, out):
    dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    res = np.tanh(dense)
    from_numpy(res, out)

def reduce_sum(a, axes=None):
    return a.reduce_sum(axes)


# -------------------- matmul --------------------

def matmul(a: Array, b: Array, out: Array, m: int, n: int, p: int):
    """
    Multiplication of two arrays. We expect a.shape == (m,n) and b.shape == (n,p).
    We'll implement a CSR @ dense path where a is CSR and b is dense (fast).
    If b is CSR too, we densify b (simple fallback).
    The result is written into out as CSR.
    """
    # get dense shape info for a and b
    a_rows, a_cols = (a.shape if a.shape is not None else (m, n))
    b_rows, b_cols = (b.shape if b.shape is not None else (n, p))

    # build dense view for b — if b is sparse, densify
    if getattr(b, "shape", None) is not None and b.shape is not None and b.indices.size == 0:
        # b is empty
        b_dense = np.zeros((n, p), dtype=_datatype)
    elif hasattr(b, "indptr") and hasattr(b, "indices") and hasattr(b, "data"):
        # b is CSR -> densify
        b_dense = _dense_from_csr(b.indptr, b.indices, b.data, (n, p))
    else:
        # fallback: if b has dense_flat attribute
        try:
            b_dense = b.dense_flat.reshape(n, p)
        except Exception:
            # final fallback: zero
            b_dense = np.zeros((n, p), dtype=_datatype)

    # a is CSR: iterate rows and perform row-wise multiply (efficient)
    res = np.zeros((m, p), dtype=_datatype)
    for r in range(m):
        start = a.indptr[r]
        end = a.indptr[r + 1]
        if end > start:
            cols = a.indices[start:end]
            vals = a.data[start:end].astype(_datatype)
            # multiply vals (len k) with b_dense[cols,:] -> sum over k
            # Compute weighted sum of rows of b_dense at columns 'cols' scaled by 'vals'
            # Equivalent to res[r,:] = vals @ b_dense[cols, :]
            # We'll vectorize: res[r] += sum_j vals[j] * b_dense[cols[j], :]
            res_r = (vals.reshape(-1, 1) * b_dense[cols, :]).sum(axis=0)
            res[r, :] = res_r

    from_numpy(res, out)


# -------------------- reductions --------------------

def reduce_max(a: Array, out: Array, reduce_size):
    """
    reduce_size is number of elements to reduce on last axis;
    We reconstruct dense, reshape and reduce.
    """
    if a.shape is None:
        dense = np.zeros((1, a._size), dtype=_datatype)
    else:
        dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    # flatten and reshape to (-1, reduce_size)
    flat = dense.reshape(-1)
    resh = flat.reshape(-1, reduce_size)
    res = resh.max(axis=1)
    from_numpy(res, out)


def reduce_sum(a, axes=None):
    """
    Reduce sum for CSR NDArray.
    Supports:
      - sum all elements
      - sum along axis 0 or 1
    """
    data = a.data
    indices = a.indices
    indptr = a.indptr
    m, n = a.shape

    if axes is None:
        # Sum all stored values
        return np.sum(data)

    # Normalize to tuple
    if isinstance(axes, int):
        axes = (axes,)

    # Only 2D sparse matrices supported
    if axes == (0,):
        # Column sum: go through stored elements
        out = np.zeros(n, dtype=data.dtype)
        for row in range(m):
            start, end = indptr[row], indptr[row + 1]
            cols = indices[start:end]
            vals = data[start:end]
            out[cols] += vals
        return out

    if axes == (1,):
        # Row sum: sum slices
        out = np.zeros(m, dtype=data.dtype)
        for row in range(m):
            start, end = indptr[row], indptr[row + 1]
            out[row] = np.sum(data[start:end])
        return out

    raise NotImplementedError(f"CSR reduce_sum does not support axes={axes}")
