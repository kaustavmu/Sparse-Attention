"""
Unit tests for SparseTensor implementation

This file should be placed at: tests/hw3/test_sparse_tensor.py
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import from needle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'python'))

from needle.backend_ndarray.sparse_tensor import (
    SparseTensor, 
    sparse_add, 
    sparse_scalar_multiply,
    sparse_elementwise_multiply,
    sparse_dense_matmul,
    dense_sparse_matmul,
    create_random_sparse,
    create_identity_sparse
)


class TestSparseTensorBasics:
    """Test basic SparseTensor construction and conversion"""
    
    def test_construction_from_triplets(self):
        """Test creating sparse tensor from row, col, value lists"""
        rows = [0, 1, 2]
        cols = [1, 2, 0]
        values = [1.0, 2.0, 3.0]
        
        sparse = SparseTensor.from_triplets(rows, cols, values, shape=(3, 3))
        
        assert sparse.shape == (3, 3)
        assert sparse.nnz == 3
        assert sparse[0, 1] == 1.0
        assert sparse[1, 2] == 2.0
        assert sparse[2, 0] == 3.0
        assert sparse[0, 0] == 0.0  # Not in sparse representation
    
    def test_from_dense(self):
        """Test creating sparse tensor from dense array"""
        dense = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        
        sparse = SparseTensor.from_dense(dense)
        
        assert sparse.shape == (3, 3)
        assert sparse.nnz == 5
        
        # Check values
        assert sparse[0, 0] == 1.0
        assert sparse[0, 2] == 2.0
        assert sparse[1, 1] == 3.0
        assert sparse[2, 0] == 4.0
        assert sparse[2, 2] == 5.0
    
    def test_to_dense(self):
        """Test converting sparse back to dense"""
        dense_original = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        
        sparse = SparseTensor.from_dense(dense_original)
        dense_reconstructed = sparse.to_dense()
        
        np.testing.assert_array_almost_equal(dense_original, dense_reconstructed)
    
    def test_transpose(self):
        """Test transpose operation"""
        dense = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        sparse = SparseTensor.from_dense(dense)
        sparse_T = sparse.transpose()
        
        expected_T = dense.T
        result_T = sparse_T.to_dense()
        
        np.testing.assert_array_almost_equal(expected_T, result_T)
    
    def test_coalesce(self):
        """Test combining duplicate indices"""
        # Create sparse tensor with duplicate indices
        rows = [0, 0, 1, 1]
        cols = [1, 1, 2, 2]
        values = [1.0, 2.0, 3.0, 4.0]
        
        sparse = SparseTensor.from_triplets(rows, cols, values, shape=(3, 3))
        coalesced = sparse.coalesce()
        
        assert coalesced.nnz == 2  # Should have combined duplicates
        assert coalesced[0, 1] == 3.0  # 1.0 + 2.0
        assert coalesced[1, 2] == 7.0  # 3.0 + 4.0
    
    def test_empty_sparse_tensor(self):
        """Test handling of empty sparse tensors"""
        indices = np.zeros((2, 0), dtype=np.int64)
        values = np.zeros(0, dtype=np.float32)
        
        sparse = SparseTensor(indices, values, shape=(3, 3))
        
        assert sparse.nnz == 0
        assert sparse[0, 0] == 0.0
        
        dense = sparse.to_dense()
        expected = np.zeros((3, 3))
        np.testing.assert_array_almost_equal(dense, expected)


class TestSparseOperations:
    """Test sparse matrix operations"""
    
    def test_sparse_add(self):
        """Test addition of two sparse tensors"""
        # Create two sparse tensors
        dense_a = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0]
        ])
        dense_b = np.array([
            [0.0, 4.0, 2.0],
            [5.0, 3.0, 0.0]
        ])
        
        sparse_a = SparseTensor.from_dense(dense_a)
        sparse_b = SparseTensor.from_dense(dense_b)
        
        result = sparse_add(sparse_a, sparse_b)
        result_dense = result.to_dense()
        
        expected = dense_a + dense_b
        np.testing.assert_array_almost_equal(result_dense, expected)
    
    def test_scalar_multiply(self):
        """Test multiplication by scalar"""
        dense = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0]
        ])
        
        sparse = SparseTensor.from_dense(dense)
        result = sparse_scalar_multiply(sparse, 2.5)
        result_dense = result.to_dense()
        
        expected = dense * 2.5
        np.testing.assert_array_almost_equal(result_dense, expected)
    
    def test_elementwise_multiply(self):
        """Test element-wise multiplication (Hadamard product)"""
        dense_a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        dense_b = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 5.0, 0.0]
        ])
        
        sparse_a = SparseTensor.from_dense(dense_a)
        sparse_b = SparseTensor.from_dense(dense_b)
        
        result = sparse_elementwise_multiply(sparse_a, sparse_b)
        result_dense = result.to_dense()
        
        expected = dense_a * dense_b
        np.testing.assert_array_almost_equal(result_dense, expected)
    
    def test_elementwise_multiply_no_overlap(self):
        """Test element-wise multiply with no overlapping non-zeros"""
        rows_a = [0, 1]
        cols_a = [0, 1]
        values_a = [1.0, 2.0]
        
        rows_b = [0, 1]
        cols_b = [1, 0]
        values_b = [3.0, 4.0]
        
        sparse_a = SparseTensor.from_triplets(rows_a, cols_a, values_a, (2, 2))
        sparse_b = SparseTensor.from_triplets(rows_b, cols_b, values_b, (2, 2))
        
        result = sparse_elementwise_multiply(sparse_a, sparse_b)
        
        assert result.nnz == 0  # No overlapping elements


class TestMatrixMultiplication:
    """Test sparse-dense matrix multiplication"""
    
    def test_sparse_dense_matmul_simple(self):
        """Test sparse @ dense multiplication"""
        # Sparse matrix (3x3)
        sparse_dense = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        sparse = SparseTensor.from_dense(sparse_dense)
        
        # Dense matrix (3x2)
        dense = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        result = sparse_dense_matmul(sparse, dense)
        expected = sparse_dense @ dense
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_dense_sparse_matmul_simple(self):
        """Test dense @ sparse multiplication"""
        # Dense matrix (2x3)
        dense = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Sparse matrix (3x3)
        sparse_dense = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        sparse = SparseTensor.from_dense(sparse_dense)
        
        result = dense_sparse_matmul(dense, sparse)
        expected = dense @ sparse_dense
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sparse_dense_matmul_larger(self):
        """Test with larger random matrices"""
        np.random.seed(42)
        
        # Create random sparse matrix
        sparse = create_random_sparse((50, 40), density=0.1, seed=42)
        sparse_dense = sparse.to_dense()
        
        # Create random dense matrix
        dense = np.random.randn(40, 30).astype(np.float32)
        
        result = sparse_dense_matmul(sparse, dense)
        expected = sparse_dense @ dense
        
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_identity_multiplication(self):
        """Test multiplication with identity matrix"""
        identity = create_identity_sparse(5)
        
        # Test sparse identity @ dense
        dense = np.random.randn(5, 3).astype(np.float32)
        result = sparse_dense_matmul(identity, dense)
        np.testing.assert_array_almost_equal(result, dense)
        
        # Test dense @ sparse identity
        dense2 = np.random.randn(3, 5).astype(np.float32)
        result2 = dense_sparse_matmul(dense2, identity)
        np.testing.assert_array_almost_equal(result2, dense2)
    
    def test_matmul_with_zeros(self):
        """Test matrix multiplication with very sparse matrices"""
        # Create a matrix with only one non-zero element
        rows = [2]
        cols = [1]
        values = [3.0]
        sparse = SparseTensor.from_triplets(rows, cols, values, (3, 3))
        
        dense = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        result = sparse_dense_matmul(sparse, dense)
        
        # Only row 2 should be non-zero, and should equal 3.0 * dense[1, :]
        expected = np.zeros((3, 2))
        expected[2, :] = 3.0 * dense[1, :]
        
        np.testing.assert_array_almost_equal(result, expected)


class TestUtilityFunctions:
    """Test utility functions for creating sparse tensors"""
    
    def test_create_random_sparse(self):
        """Test random sparse tensor creation"""
        sparse = create_random_sparse((100, 100), density=0.1, seed=42)
        
        assert sparse.shape == (100, 100)
        expected_nnz = int(100 * 100 * 0.1)
        # Allow some tolerance in nnz
        assert abs(sparse.nnz - expected_nnz) < 50
    
    def test_create_identity_sparse(self):
        """Test identity matrix creation"""
        identity = create_identity_sparse(5)
        dense = identity.to_dense()
        
        expected = np.eye(5, dtype=np.float32)
        np.testing.assert_array_almost_equal(dense, expected)
    
    def test_random_sparse_reproducibility(self):
        """Test that random sparse creation is reproducible with seed"""
        sparse1 = create_random_sparse((10, 10), density=0.3, seed=123)
        sparse2 = create_random_sparse((10, 10), density=0.3, seed=123)
        
        np.testing.assert_array_equal(sparse1.indices, sparse2.indices)
        np.testing.assert_array_almost_equal(sparse1.values, sparse2.values)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_shape_mismatch_add(self):
        """Test that adding tensors with different shapes raises error"""
        sparse1 = create_random_sparse((3, 3), density=0.5)
        sparse2 = create_random_sparse((3, 4), density=0.5)
        
        with pytest.raises(AssertionError):
            sparse_add(sparse1, sparse2)
    
    def test_shape_mismatch_matmul(self):
        """Test that incompatible shapes for matmul raise error"""
        sparse = create_random_sparse((3, 4), density=0.5)
        dense = np.random.randn(5, 2)  # Incompatible
        
        with pytest.raises(AssertionError):
            sparse_dense_matmul(sparse, dense)
    
    def test_out_of_bounds_indices(self):
        """Test that out-of-bounds indices raise error"""
        rows = [0, 1, 5]  # 5 is out of bounds
        cols = [0, 1, 2]
        values = [1.0, 2.0, 3.0]
        
        with pytest.raises(AssertionError):
            SparseTensor.from_triplets(rows, cols, values, shape=(3, 3))
    
    def test_very_sparse_matrix(self):
        """Test with extremely sparse matrix (1 element)"""
        sparse = create_random_sparse((1000, 1000), density=0.000001, seed=42)
        
        assert sparse.nnz >= 1
        assert sparse.shape == (1000, 1000)
        
        # Should still convert to dense correctly
        dense = sparse.to_dense()
        assert dense.shape == (1000, 1000)


class TestPerformance:
    """Performance and memory tests"""
    
    def test_large_sparse_matrix(self):
        """Test handling of large sparse matrices"""
        # Create a 10000 x 10000 matrix with 0.1% density
        sparse = create_random_sparse((10000, 10000), density=0.001, seed=42)
        
        assert sparse.shape == (10000, 10000)
        assert sparse.nnz < 10000 * 10000  # Much less than dense
        
        # Test basic operations don't crash
        sparse_T = sparse.transpose()
        assert sparse_T.shape == (10000, 10000)
    
    def test_matmul_performance_comparison(self):
        """Compare sparse vs dense matmul (sparse should use less memory)"""
        # For very sparse matrices, sparse should be more efficient
        sparse = create_random_sparse((1000, 1000), density=0.01, seed=42)
        dense_small = np.random.randn(1000, 10).astype(np.float32)
        
        # This should work without running out of memory
        result = sparse_dense_matmul(sparse, dense_small)
        
        assert result.shape == (1000, 10)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
