"""
Sparse Tensor Implementation for Needle
COO (Coordinate) format with basic operations

This file should be placed at: python/needle/backend_ndarray/sparse_tensor.py
"""

import numpy as np
from typing import Tuple, Optional, List


class SparseTensor:
    """
    Sparse tensor using COO (Coordinate) format.
    
    Attributes:
        indices: (2, nnz) array where indices[0] are row indices, indices[1] are col indices
        values: (nnz,) array of non-zero values
        shape: (n_rows, n_cols) shape of the sparse matrix
        nnz: number of non-zero elements
    """
    
    def __init__(self, indices: np.ndarray, values: np.ndarray, shape: Tuple[int, int]):
        """
        Initialize a sparse tensor in COO format.
        
        Args:
            indices: (2, nnz) integer array of coordinates
            values: (nnz,) array of values at those coordinates
            shape: (n_rows, n_cols) shape of the matrix
        """
        assert indices.shape[0] == 2, "Indices must be (2, nnz)"
        assert indices.shape[1] == values.shape[0], "Indices and values must have same nnz"
        assert len(shape) == 2, "Shape must be 2D"
        
        self.indices = indices.astype(np.int64)
        self.values = values.astype(np.float32)
        self.shape = tuple(shape)
        self.nnz = len(values)
        
        # Validate indices are in bounds
        assert np.all(indices[0] >= 0) and np.all(indices[0] < shape[0]), "Row indices out of bounds"
        assert np.all(indices[1] >= 0) and np.all(indices[1] < shape[1]), "Col indices out of bounds"
    
    @staticmethod
    def from_dense(dense_array: np.ndarray) -> 'SparseTensor':
        """
        Create a SparseTensor from a dense numpy array.
        
        Args:
            dense_array: 2D numpy array
            
        Returns:
            SparseTensor representation
        """
        assert len(dense_array.shape) == 2, "Input must be 2D"
        
        # Find non-zero elements
        row_indices, col_indices = np.nonzero(dense_array)
        values = dense_array[row_indices, col_indices]
        
        indices = np.stack([row_indices, col_indices], axis=0)
        
        return SparseTensor(indices, values, dense_array.shape)
    
    @staticmethod
    def from_triplets(rows: List[int], cols: List[int], values: List[float], 
                      shape: Tuple[int, int]) -> 'SparseTensor':
        """
        Create a SparseTensor from lists of row indices, column indices, and values.
        
        Args:
            rows: List of row indices
            cols: List of column indices
            values: List of values
            shape: (n_rows, n_cols) shape
            
        Returns:
            SparseTensor
        """
        indices = np.array([rows, cols], dtype=np.int64)
        values = np.array(values, dtype=np.float32)
        return SparseTensor(indices, values, shape)
    
    def to_dense(self) -> np.ndarray:
        """
        Convert sparse tensor to dense numpy array.
        
        Returns:
            Dense 2D numpy array
        """
        dense = np.zeros(self.shape, dtype=np.float32)
        dense[self.indices[0], self.indices[1]] = self.values
        return dense
    
    def transpose(self) -> 'SparseTensor':
        """
        Transpose the sparse tensor.
        
        Returns:
            Transposed SparseTensor
        """
        # Swap row and column indices
        transposed_indices = np.array([self.indices[1], self.indices[0]])
        transposed_shape = (self.shape[1], self.shape[0])
        
        return SparseTensor(transposed_indices, self.values.copy(), transposed_shape)
    
    def __getitem__(self, key: Tuple[int, int]) -> float:
        """
        Get element at (row, col). Returns 0.0 if not in sparse representation.
        Note: This is O(nnz) - not efficient for repeated access.
        
        Args:
            key: (row, col) tuple
            
        Returns:
            Value at that position
        """
        row, col = key
        assert 0 <= row < self.shape[0] and 0 <= col < self.shape[1], "Index out of bounds"
        
        # Find matching indices
        mask = (self.indices[0] == row) & (self.indices[1] == col)
        if np.any(mask):
            return float(self.values[mask][0])
        return 0.0
    
    def coalesce(self) -> 'SparseTensor':
        """
        Combine duplicate indices by summing their values.
        Returns a new SparseTensor with unique indices.
        
        Returns:
            Coalesced SparseTensor
        """
        # Convert indices to unique keys
        keys = self.indices[0] * self.shape[1] + self.indices[1]
        
        # Find unique keys and sum duplicate values
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        
        unique_values = np.zeros(len(unique_keys), dtype=np.float32)
        np.add.at(unique_values, inverse_indices, self.values)
        
        # Convert back to row, col indices
        unique_rows = unique_keys // self.shape[1]
        unique_cols = unique_keys % self.shape[1]
        unique_indices = np.stack([unique_rows, unique_cols], axis=0)
        
        return SparseTensor(unique_indices, unique_values, self.shape)
    
    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, nnz={self.nnz})"
    
    def __str__(self) -> str:
        return f"SparseTensor(shape={self.shape}, nnz={self.nnz}, density={self.nnz/(self.shape[0]*self.shape[1]):.4f})"


def sparse_add(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """
    Add two sparse tensors. They must have the same shape.
    
    Args:
        a: First SparseTensor
        b: Second SparseTensor
        
    Returns:
        Sum as SparseTensor
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Concatenate indices and values
    combined_indices = np.concatenate([a.indices, b.indices], axis=1)
    combined_values = np.concatenate([a.values, b.values])
    
    result = SparseTensor(combined_indices, combined_values, a.shape)
    
    # Coalesce to combine duplicates
    return result.coalesce()


def sparse_scalar_multiply(a: SparseTensor, scalar: float) -> SparseTensor:
    """
    Multiply sparse tensor by a scalar.
    
    Args:
        a: SparseTensor
        scalar: Scalar value
        
    Returns:
        Result as SparseTensor
    """
    new_values = a.values * scalar
    return SparseTensor(a.indices.copy(), new_values, a.shape)


def sparse_elementwise_multiply(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """
    Element-wise multiplication of two sparse tensors (Hadamard product).
    Only elements present in BOTH tensors contribute to result.
    
    Args:
        a: First SparseTensor
        b: Second SparseTensor
        
    Returns:
        Element-wise product as SparseTensor
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    # Convert indices to keys for efficient lookup
    a_keys = a.indices[0] * a.shape[1] + a.indices[1]
    b_keys = b.indices[0] * b.shape[1] + b.indices[1]
    
    # Find intersection of keys
    common_keys = np.intersect1d(a_keys, b_keys)
    
    if len(common_keys) == 0:
        # No common elements - return empty sparse tensor
        return SparseTensor(np.zeros((2, 0), dtype=np.int64), 
                          np.zeros(0, dtype=np.float32), 
                          a.shape)
    
    # Find indices in both tensors for common keys
    a_mask = np.isin(a_keys, common_keys)
    b_mask = np.isin(b_keys, common_keys)
    
    # Get values and multiply
    a_common_values = a.values[a_mask]
    b_common_values = b.values[b_mask]
    
    # Sort to ensure matching order
    a_sort_idx = np.argsort(a_keys[a_mask])
    b_sort_idx = np.argsort(b_keys[b_mask])
    
    result_values = a_common_values[a_sort_idx] * b_common_values[b_sort_idx]
    
    # Convert keys back to indices
    result_rows = common_keys // a.shape[1]
    result_cols = common_keys % a.shape[1]
    result_indices = np.stack([result_rows, result_cols], axis=0)
    
    return SparseTensor(result_indices, result_values, a.shape)


def sparse_dense_matmul(sparse: SparseTensor, dense: np.ndarray) -> np.ndarray:
    """
    Multiply sparse matrix (M x K) by dense matrix (K x N).
    Returns dense result (M x N).
    
    This is the critical operation for sparse attention!
    
    Args:
        sparse: SparseTensor of shape (M, K)
        dense: Dense array of shape (K, N)
        
    Returns:
        Dense result of shape (M, N)
    """
    assert len(dense.shape) == 2, "Dense input must be 2D"
    assert sparse.shape[1] == dense.shape[0], \
        f"Shape mismatch: sparse {sparse.shape} @ dense {dense.shape}"
    
    M, K = sparse.shape
    K_d, N = dense.shape
    
    # Initialize result
    result = np.zeros((M, N), dtype=np.float32)
    
    # Perform multiplication
    # For each non-zero element sparse[i,k] = v:
    #   result[i, :] += v * dense[k, :]
    
    row_indices = sparse.indices[0]
    col_indices = sparse.indices[1]
    values = sparse.values
    
    for idx in range(sparse.nnz):
        i = row_indices[idx]
        k = col_indices[idx]
        v = values[idx]
        
        result[i, :] += v * dense[k, :]
    
    return result


def dense_sparse_matmul(dense: np.ndarray, sparse: SparseTensor) -> np.ndarray:
    """
    Multiply dense matrix (M x K) by sparse matrix (K x N).
    Returns dense result (M x N).
    
    Args:
        dense: Dense array of shape (M, K)
        sparse: SparseTensor of shape (K, N)
        
    Returns:
        Dense result of shape (M, N)
    """
    # Use transpose trick: A @ B = (B^T @ A^T)^T
    # But more efficient to implement directly:
    
    assert len(dense.shape) == 2, "Dense input must be 2D"
    assert dense.shape[1] == sparse.shape[0], \
        f"Shape mismatch: dense {dense.shape} @ sparse {sparse.shape}"
    
    M, K = dense.shape
    K_s, N = sparse.shape
    
    # Initialize result
    result = np.zeros((M, N), dtype=np.float32)
    
    # For each non-zero element sparse[k,j] = v:
    #   result[:, j] += dense[:, k] * v
    
    row_indices = sparse.indices[0]
    col_indices = sparse.indices[1]
    values = sparse.values
    
    for idx in range(sparse.nnz):
        k = row_indices[idx]
        j = col_indices[idx]
        v = values[idx]
        
        result[:, j] += dense[:, k] * v
    
    return result


def create_random_sparse(shape: Tuple[int, int], density: float, 
                         seed: Optional[int] = None) -> SparseTensor:
    """
    Create a random sparse tensor with given density.
    
    Args:
        shape: (n_rows, n_cols)
        density: Fraction of elements that are non-zero (0 < density < 1)
        seed: Random seed for reproducibility
        
    Returns:
        Random SparseTensor
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_elements = shape[0] * shape[1]
    nnz = max(1, int(n_elements * density))
    
    # Generate random indices without replacement
    flat_indices = np.random.choice(n_elements, size=nnz, replace=False)
    
    row_indices = flat_indices // shape[1]
    col_indices = flat_indices % shape[1]
    
    indices = np.stack([row_indices, col_indices], axis=0)
    values = np.random.randn(nnz).astype(np.float32)
    
    return SparseTensor(indices, values, shape)


def create_identity_sparse(n: int) -> SparseTensor:
    """
    Create a sparse identity matrix of size n x n.
    
    Args:
        n: Size of identity matrix
        
    Returns:
        Identity SparseTensor
    """
    indices = np.stack([np.arange(n), np.arange(n)], axis=0)
    values = np.ones(n, dtype=np.float32)
    return SparseTensor(indices, values, (n, n))
