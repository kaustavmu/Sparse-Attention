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
    # Handle 1D shapes specially - CSR stores as 2D but we need 1D output
    if len(shape) == 1:
        if a.shape is None:
            result = np.zeros(shape[0], dtype=_datatype)
        else:
            # CSR structure is 2D (rows, cols), but we want 1D output
            rows, cols = a.shape
            # If the CSR has shape (rows, 1), it's likely a reduction result
            # Extract the first (and only) column
            if cols == 1:
                dense_2d = _dense_from_csr(a.indptr, a.indices, a.data, (rows, cols))
                result = dense_2d[:, 0]  # Extract first column
            else:
                # General case: flatten the 2D structure
                dense_2d = _dense_from_csr(a.indptr, a.indices, a.data, (rows, cols))
                flat = dense_2d.reshape(-1)
                result = flat[offset:offset+shape[0]]
        byte_strides = tuple([s * _datetype_size for s in strides])
        return np.lib.stride_tricks.as_strided(result, shape, byte_strides)
    
    # For 2D+ shapes, convert to 2D representation
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
    else:
        if out.shape is None:
            rows = 1
            cols = out._size
            out.shape = (rows, cols)
        rows, cols = out.shape
        # Fill with non-zero: all elements are non-zero
        total = rows * cols
        out.data = np.full(total, val, dtype=_datatype)
        out.indices = np.tile(np.arange(cols, dtype=np.int64), rows)
        out.indptr = np.arange(0, total + 1, cols, dtype=np.int64)


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
    Set the subarray specified by (out, shape, strides, offset) to the values in CSR a.
    If the view covers the entire array with standard strides, just copy a's structure.
    """
    # Simple case: replacing entire array with compact strides
    if offset == 0 and strides == tuple([1] * len(shape)) and shape == out.shape:
        out.indptr = a.indptr.copy()
        out.indices = a.indices.copy()
        out.data = a.data.copy()
        out.shape = a.shape
        out._size = a._size
        return
    
    # Complex case with slicing - need more sophisticated handling
    # For now, fall back to dense (this is the hard case that would need slice translation)
    dense_out = to_numpy(out, out.shape, tuple([1] * len(out.shape)), 0).copy()
    a_dense = _dense_from_csr(a.indptr, a.indices, a.data, a.shape)
    
    # Apply the assignment using numpy's advanced indexing
    view = np.lib.stride_tricks.as_strided(
        dense_out.reshape(-1)[offset:], 
        shape, 
        tuple([s * _datetype_size for s in strides])
    )
    view[:] = a_dense.reshape(-1)[:np.prod(shape)]
    
    from_numpy(dense_out.reshape(out.shape), out)


def scalar_setitem(size, val, out: Array, shape, strides, offset):
    """Set the specified view in out to scalar val, working on sparse structure."""
    
    # Simple case: setting entire array with standard strides
    if offset == 0 and strides == tuple([1] * len(shape)) and shape == out.shape:
        fill(out, val)
        return
    
    # Complex case: need to handle partial array updates
    # This is challenging in pure sparse - we need to:
    # 1. Identify which (row, col) positions are in the slice
    # 2. Update/insert those positions with val (or remove if val==0)
    
    # For 2D arrays with row slicing, we can handle efficiently
    if len(shape) == 2 and out.shape is not None:
        rows, cols = out.shape
        target_rows, target_cols = shape
        
        # Simple row-contiguous case with offset as row offset
        if strides == (cols, 1) and target_cols == cols:
            row_start = offset // cols
            row_end = row_start + target_rows
            
            if val == 0:
                # Remove values in these rows
                new_data = []
                new_indices = []
                new_indptr = [0]
                
                for row in range(rows):
                    if row < row_start or row >= row_end:
                        # Keep this row
                        start, end = out.indptr[row], out.indptr[row + 1]
                        new_indices.extend(out.indices[start:end])
                        new_data.extend(out.data[start:end])
                    # else: skip rows in range (set to zero)
                    new_indptr.append(len(new_data))
                
                out.indptr = np.array(new_indptr, dtype=np.int64)
                out.indices = np.array(new_indices, dtype=np.int64)
                out.data = np.array(new_data, dtype=_datatype)
            else:
                # Set these rows to val (dense rows)
                new_data = []
                new_indices = []
                new_indptr = [0]
                
                for row in range(rows):
                    if row < row_start or row >= row_end:
                        # Keep this row as-is
                        start, end = out.indptr[row], out.indptr[row + 1]
                        new_indices.extend(out.indices[start:end])
                        new_data.extend(out.data[start:end])
                    else:
                        # Set this row to all val
                        new_indices.extend(range(cols))
                        new_data.extend([val] * cols)
                    new_indptr.append(len(new_data))
                
                out.indptr = np.array(new_indptr, dtype=np.int64)
                out.indices = np.array(new_indices, dtype=np.int64)
                out.data = np.array(new_data, dtype=_datatype)
            return
    
    # Fallback for complex slicing: densify, modify, sparsify
    # This is the general case that's hard to avoid for arbitrary slices
    full = _dense_from_csr(out.indptr, out.indices, out.data, out.shape) if out.shape is not None else np.zeros(out._size, dtype=_datatype).reshape(1, -1)
    flat_view = np.lib.stride_tricks.as_strided(
        full.reshape(-1)[offset:], 
        shape, 
        tuple([s * _datetype_size for s in strides])
    )
    flat_view[:] = val
    from_numpy(full.reshape(out.shape), out)


# -------------------- elementwise and scalar ops --------------------

def _csr_add_sparse(a: Array, b: Array, out: Array):
    """Add two CSR matrices directly in sparse format."""
    assert a.shape == b.shape, "Shapes must match"
    rows, cols = a.shape
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        b_start, b_end = b.indptr[row], b.indptr[row + 1]
        
        a_cols = a.indices[a_start:a_end]
        a_vals = a.data[a_start:a_end]
        b_cols = b.indices[b_start:b_end]
        b_vals = b.data[b_start:b_end]
        
        # Merge two sorted arrays
        row_data = {}
        for col, val in zip(a_cols, a_vals):
            row_data[col] = val
        for col, val in zip(b_cols, b_vals):
            row_data[col] = row_data.get(col, 0) + val
        
        # Filter out zeros and sort
        nonzero = [(col, val) for col, val in sorted(row_data.items()) if val != 0]
        if nonzero:
            cols, vals = zip(*nonzero)
            out_indices.extend(cols)
            out_data.extend(vals)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def _csr_scalar_add(a: Array, val: float, out: Array):
    """Add scalar to CSR matrix. If val != 0, result is dense."""
    if val == 0:
        # Just copy the sparse structure
        out.indptr = a.indptr.copy()
        out.indices = a.indices.copy()
        out.data = a.data.copy()
        out.shape = a.shape
        out._size = a._size
    else:
        # Adding non-zero scalar makes matrix dense
        rows, cols = a.shape
        out_data = []
        out_indices = []
        out_indptr = [0]
        
        for row in range(rows):
            start, end = a.indptr[row], a.indptr[row + 1]
            sparse_cols = set(a.indices[start:end])
            
            # All columns become non-zero
            for col in range(cols):
                if col in sparse_cols:
                    idx = np.where(a.indices[start:end] == col)[0][0]
                    out_data.append(a.data[start + idx] + val)
                else:
                    out_data.append(val)
                out_indices.append(col)
            
            out_indptr.append(len(out_data))
        
        out.indptr = np.array(out_indptr, dtype=np.int64)
        out.indices = np.array(out_indices, dtype=np.int64)
        out.data = np.array(out_data, dtype=_datatype)
        out.shape = a.shape
        out._size = rows * cols


def _csr_multiply_sparse(a: Array, b: Array, out: Array):
    """Element-wise multiply two CSR matrices (intersection only)."""
    assert a.shape == b.shape, "Shapes must match"
    rows, cols = a.shape
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        b_start, b_end = b.indptr[row], b.indptr[row + 1]
        
        a_cols = set(a.indices[a_start:a_end])
        b_dict = {col: val for col, val in zip(b.indices[b_start:b_end], b.data[b_start:b_end])}
        
        # Only non-zero where both are non-zero
        for i in range(a_start, a_end):
            col = a.indices[i]
            if col in b_dict:
                val = a.data[i] * b_dict[col]
                if val != 0:
                    out_indices.append(col)
                    out_data.append(val)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def _csr_scalar_mul(a: Array, val: float, out: Array):
    """Multiply CSR matrix by scalar."""
    if val == 0:
        # Result is all zeros
        out.indptr = np.array([0] * (a.shape[0] + 1), dtype=np.int64)
        out.indices = np.array([], dtype=np.int64)
        out.data = np.array([], dtype=_datatype)
    else:
        # Multiply all stored values
        out.indptr = a.indptr.copy()
        out.indices = a.indices.copy()
        out.data = a.data * val
    out.shape = a.shape
    out._size = a._size


def _csr_scalar_div(a: Array, val: float, out: Array):
    """Divide CSR matrix by scalar."""
    out.indptr = a.indptr.copy()
    out.indices = a.indices.copy()
    out.data = a.data / val
    out.shape = a.shape
    out._size = a._size


def _csr_scalar_power(a: Array, val: float, out: Array):
    """Raise CSR matrix to scalar power."""
    # Note: zeros raised to positive power remain zero
    out.indptr = a.indptr.copy()
    out.indices = a.indices.copy()
    out.data = a.data ** val
    out.shape = a.shape
    out._size = a._size


def _csr_maximum_sparse(a: Array, b: Array, out: Array):
    """Element-wise maximum of two CSR matrices."""
    assert a.shape == b.shape, "Shapes must match"
    rows, cols = a.shape
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        b_start, b_end = b.indptr[row], b.indptr[row + 1]
        
        a_dict = {col: val for col, val in zip(a.indices[a_start:a_end], a.data[a_start:a_end])}
        b_dict = {col: val for col, val in zip(b.indices[b_start:b_end], b.data[b_start:b_end])}
        
        # Union of non-zero positions
        all_cols = sorted(set(a_dict.keys()) | set(b_dict.keys()))
        for col in all_cols:
            val = max(a_dict.get(col, 0), b_dict.get(col, 0))
            if val != 0:
                out_indices.append(col)
                out_data.append(val)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def _csr_scalar_maximum(a: Array, val: float, out: Array):
    """Element-wise maximum of CSR matrix and scalar."""
    rows, cols = a.shape
    
    if val <= 0:
        # If val <= 0, just take max with stored values (others are 0)
        out.indptr = a.indptr.copy()
        out.indices = a.indices.copy()
        out.data = np.maximum(a.data, val)
    else:
        # If val > 0, all zeros become val (matrix becomes dense)
        out_data = []
        out_indices = []
        out_indptr = [0]
        
        for row in range(rows):
            start, end = a.indptr[row], a.indptr[row + 1]
            sparse_dict = {col: v for col, v in zip(a.indices[start:end], a.data[start:end])}
            
            for col in range(cols):
                max_val = max(sparse_dict.get(col, 0), val)
                out_indices.append(col)
                out_data.append(max_val)
            
            out_indptr.append(len(out_data))
        
        out.indptr = np.array(out_indptr, dtype=np.int64)
        out.indices = np.array(out_indices, dtype=np.int64)
        out.data = np.array(out_data, dtype=_datatype)
    
    out.shape = a.shape
    out._size = rows * cols


def _csr_log(a: Array, out: Array):
    """Element-wise log of CSR matrix (only on non-zero elements)."""
    out.indptr = a.indptr.copy()
    out.indices = a.indices.copy()
    out.data = np.log(a.data)
    out.shape = a.shape
    out._size = a._size


def _csr_exp(a: Array, out: Array):
    """Element-wise exp of CSR matrix. Note: exp(0) = 1, so result is dense!"""
    rows, cols = a.shape
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        start, end = a.indptr[row], a.indptr[row + 1]
        sparse_dict = {col: val for col, val in zip(a.indices[start:end], a.data[start:end])}
        
        # All positions become non-zero (exp(0) = 1)
        for col in range(cols):
            val = np.exp(sparse_dict.get(col, 0))
            out_indices.append(col)
            out_data.append(val)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def _csr_tanh(a: Array, out: Array):
    """Element-wise tanh of CSR matrix (only on non-zero elements, tanh(0)=0)."""
    out.indptr = a.indptr.copy()
    out.indices = a.indices.copy()
    out.data = np.tanh(a.data)
    out.shape = a.shape
    out._size = a._size


def _csr_comparison(a: Array, b: Array, op, out: Array):
    """Generic comparison operation between two CSR matrices."""
    assert a.shape == b.shape, "Shapes must match"
    rows, cols = a.shape
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        b_start, b_end = b.indptr[row], b.indptr[row + 1]
        
        a_dict = {col: val for col, val in zip(a.indices[a_start:a_end], a.data[a_start:a_end])}
        b_dict = {col: val for col, val in zip(b.indices[b_start:b_end], b.data[b_start:b_end])}
        
        # Need to check all columns for comparisons
        all_cols = sorted(set(a_dict.keys()) | set(b_dict.keys()) | set(range(cols)))
        for col in all_cols:
            a_val = a_dict.get(col, 0)
            b_val = b_dict.get(col, 0)
            result = op(a_val, b_val)
            if result != 0:
                out_indices.append(col)
                out_data.append(_datatype(result))
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def _csr_scalar_comparison(a: Array, val: float, op, out: Array):
    """Generic comparison between CSR matrix and scalar."""
    rows, cols = a.shape
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        start, end = a.indptr[row], a.indptr[row + 1]
        sparse_dict = {col: v for col, v in zip(a.indices[start:end], a.data[start:end])}
        
        for col in range(cols):
            a_val = sparse_dict.get(col, 0)
            result = op(a_val, val)
            if result != 0:
                out_indices.append(col)
                out_data.append(_datatype(result))
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


# -------------------- Public API --------------------

def ewise_add(a, b, out):
    _csr_add_sparse(a, b, out)


def scalar_add(a, val, out):
    _csr_scalar_add(a, val, out)


def ewise_mul(a, b, out):
    _csr_multiply_sparse(a, b, out)


def scalar_mul(a, val, out):
    _csr_scalar_mul(a, val, out)


def ewise_div(a, b, out):
    # Division: need to handle division by zero carefully
    # For sparse representation, divide stored values where b is non-zero
    assert a.shape == b.shape, "Shapes must match"
    rows, cols = a.shape
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(rows):
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        b_start, b_end = b.indptr[row], b.indptr[row + 1]
        
        a_dict = {col: val for col, val in zip(a.indices[a_start:a_end], a.data[a_start:a_end])}
        b_dict = {col: val for col, val in zip(b.indices[b_start:b_end], b.data[b_start:b_end])}
        
        # Division is only defined where b is non-zero
        for col in a_dict:
            if col in b_dict:
                result = a_dict[col] / b_dict[col]
                if not np.isnan(result) and not np.isinf(result) and result != 0:
                    out_indices.append(col)
                    out_data.append(result)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = a.shape
    out._size = rows * cols


def scalar_div(a, val, out):
    _csr_scalar_div(a, val, out)


def scalar_power(a, val, out):
    _csr_scalar_power(a, val, out)


def ewise_maximum(a, b, out):
    _csr_maximum_sparse(a, b, out)


def scalar_maximum(a, val, out):
    _csr_scalar_maximum(a, val, out)


def ewise_eq(a, b, out):
    _csr_comparison(a, b, lambda x, y: float(x == y), out)


def scalar_eq(a, val, out):
    _csr_scalar_comparison(a, val, lambda x, y: float(x == y), out)


def ewise_ge(a, b, out):
    _csr_comparison(a, b, lambda x, y: float(x >= y), out)


def scalar_ge(a, val, out):
    _csr_scalar_comparison(a, val, lambda x, y: float(x >= y), out)


def ewise_log(a, out):
    _csr_log(a, out)


def ewise_exp(a, out):
    _csr_exp(a, out)


def ewise_tanh(a, out):
    _csr_tanh(a, out)


# -------------------- matmul --------------------

def matmul(a: Array, b: Array, out: Array, m: int, n: int, p: int):
    """
    CSR @ CSR matrix multiplication using sparse operations.
    a is (m, n) and b is (n, p), result is (m, p).
    """
    # Build column-indexed structure for b (CSC-like access)
    b_by_col = [[] for _ in range(p)]
    for row in range(n):
        start, end = b.indptr[row], b.indptr[row + 1]
        for idx in range(start, end):
            col = b.indices[idx]
            val = b.data[idx]
            b_by_col[col].append((row, val))
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for row in range(m):
        # For this row of a, compute dot products with columns of b
        a_start, a_end = a.indptr[row], a.indptr[row + 1]
        if a_end == a_start:
            # Empty row in a
            out_indptr.append(len(out_data))
            continue
        
        a_cols = a.indices[a_start:a_end]
        a_vals = a.data[a_start:a_end]
        
        # Compute each column of result
        row_result = {}
        for a_idx, a_col in enumerate(a_cols):
            a_val = a_vals[a_idx]
            # Multiply by row a_col of b
            b_start, b_end = b.indptr[a_col], b.indptr[a_col + 1]
            for b_idx in range(b_start, b_end):
                b_col = b.indices[b_idx]
                b_val = b.data[b_idx]
                row_result[b_col] = row_result.get(b_col, 0) + a_val * b_val
        
        # Add non-zeros to output
        for col in sorted(row_result.keys()):
            val = row_result[col]
            if val != 0:
                out_indices.append(col)
                out_data.append(val)
        
        out_indptr.append(len(out_data))
    
    out.indptr = np.array(out_indptr, dtype=np.int64)
    out.indices = np.array(out_indices, dtype=np.int64)
    out.data = np.array(out_data, dtype=_datatype)
    out.shape = (m, p)
    out._size = m * p


# -------------------- reductions --------------------

def reduce_max(a: Array, out: Array, reduce_size):
    """
    Reduce max along last axis directly on sparse representation.
    """
    rows = a.shape[0]
    cols = a.shape[1]
    
    # Assuming reduce along last axis (columns)
    result = []
    for row in range(rows):
        start, end = a.indptr[row], a.indptr[row + 1]
        if end > start:
            # Max of non-zero values vs implicit zeros
            max_val = np.max(a.data[start:end])
            # But need to consider if there are zeros too
            num_nonzero = end - start
            if num_nonzero < cols:
                # There are implicit zeros
                max_val = max(max_val, 0)
            result.append(max_val)
        else:
            # All zeros
            result.append(0)
    
    # Convert result to CSR
    result_arr = np.array(result, dtype=_datatype)
    from_numpy(result_arr.reshape(-1, 1), out)


def reduce_sum(a: Array, out: Array, reduce_size):
    """
    Reduce sum along last axis directly on sparse representation.
    This matches the signature expected by NDArray.sum() method.
    The view has been permuted so the reduction axis is last.
    """
    rows = a.shape[0]
    cols = a.shape[1]
    
    # Reduce along last axis (which is the reduction axis after permute)
    result = []
    for row in range(rows):
        start, end = a.indptr[row], a.indptr[row + 1]
        # Sum of non-zero values (implicit zeros don't contribute)
        row_sum = np.sum(a.data[start:end])
        result.append(row_sum)
    
    # Convert result array
    result_arr = np.array(result, dtype=_datatype)
    
    # Check if output should be 1D or 2D based on out's shape
    # If out.shape is 1D, we need to handle it specially since CSR is 2D
    # For 1D outputs, we'll store as (rows, 1) in CSR but the NDArray wrapper will handle the shape
    # For 2D outputs (keepdims=True), we can use (rows, 1) directly
    if len(result_arr.shape) == 1:
        # 1D result - store as (rows, 1) in CSR, but note the actual shape
        from_numpy(result_arr.reshape(-1, 1), out)
        # The NDArray wrapper will handle the 1D shape correctly via to_numpy
    else:
        from_numpy(result_arr.reshape(-1, 1), out)


def reduce_sum_axes(a, axes=None):
    """
    Reduce sum for CSR array with axes parameter.
    Used by NDArray.reduce_sum() method.
    Operates directly on sparse representation.
    """
    if axes is None:
        # Sum all stored values
        return np.sum(a.data)
    
    if isinstance(axes, int):
        axes = (axes,)
    
    rows, cols = a.shape
    
    if axes == (0,):
        # Column sum
        out = np.zeros(cols, dtype=a.data.dtype)
        for row in range(rows):
            start, end = a.indptr[row], a.indptr[row + 1]
            cols_idx = a.indices[start:end]
            vals = a.data[start:end]
            out[cols_idx] += vals
        return out
    
    if axes == (1,):
        # Row sum
        out = np.zeros(rows, dtype=a.data.dtype)
        for row in range(rows):
            start, end = a.indptr[row], a.indptr[row + 1]
            out[row] = np.sum(a.data[start:end])
        return out
    
    raise NotImplementedError(f"CSR reduce_sum does not support axes={axes}")