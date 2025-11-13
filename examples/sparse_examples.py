"""
Example usage of SparseTensor implementation
Demonstrates basic operations and typical use cases

This file can be placed at: examples/sparse_examples.py or tests/hw3/examples_sparse.py
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from needle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from needle.backend_ndarray.sparse_tensor import (
    SparseTensor,
    sparse_add,
    sparse_scalar_multiply,
    sparse_dense_matmul,
    create_random_sparse,
    create_identity_sparse
)


def example_basic_creation():
    """Example 1: Creating sparse tensors"""
    print("=" * 60)
    print("Example 1: Creating Sparse Tensors")
    print("=" * 60)
    
    # Method 1: From triplets (most common for attention masks)
    rows = [0, 0, 1, 2, 2]
    cols = [0, 2, 1, 0, 2]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    sparse = SparseTensor.from_triplets(rows, cols, values, shape=(3, 3))
    print(f"\nCreated from triplets: {sparse}")
    print("Dense representation:")
    print(sparse.to_dense())
    
    # Method 2: From dense array
    dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])
    sparse2 = SparseTensor.from_dense(dense)
    print(f"\nCreated from dense: {sparse2}")
    
    # Method 3: Random sparse matrix
    sparse3 = create_random_sparse((5, 5), density=0.3, seed=42)
    print(f"\nRandom sparse matrix: {sparse3}")


def example_basic_operations():
    """Example 2: Basic operations"""
    print("\n" + "=" * 60)
    print("Example 2: Basic Operations")
    print("=" * 60)
    
    # Create two sparse matrices
    dense_a = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0]
    ])
    dense_b = np.array([
        [0.0, 4.0, 1.0],
        [5.0, 2.0, 0.0]
    ])
    
    sparse_a = SparseTensor.from_dense(dense_a)
    sparse_b = SparseTensor.from_dense(dense_b)
    
    print("\nMatrix A:")
    print(dense_a)
    print("\nMatrix B:")
    print(dense_b)
    
    # Addition
    result_add = sparse_add(sparse_a, sparse_b)
    print("\nA + B:")
    print(result_add.to_dense())
    
    # Scalar multiplication
    result_scalar = sparse_scalar_multiply(sparse_a, 2.0)
    print("\n2.0 * A:")
    print(result_scalar.to_dense())
    
    # Transpose
    result_T = sparse_a.transpose()
    print("\nA^T:")
    print(result_T.to_dense())


def example_matrix_multiplication():
    """Example 3: Sparse-Dense matrix multiplication"""
    print("\n" + "=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)
    
    # Create sparse matrix (attention weights pattern)
    sparse_dense = np.array([
        [0.9, 0.1, 0.0, 0.0],  # Token 0 attends to tokens 0, 1
        [0.2, 0.7, 0.1, 0.0],  # Token 1 attends to tokens 0, 1, 2
        [0.0, 0.3, 0.6, 0.1],  # Token 2 attends to tokens 1, 2, 3
        [0.0, 0.0, 0.2, 0.8]   # Token 3 attends to tokens 2, 3
    ])
    
    sparse = SparseTensor.from_dense(sparse_dense)
    print(f"Sparse attention matrix: {sparse}")
    print("Pattern (showing non-zeros):")
    print(sparse_dense)
    
    # Dense value matrix (embeddings)
    values = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    
    print("\nValue matrix (4 tokens, 2 dims):")
    print(values)
    
    # Compute attention output
    output = sparse_dense_matmul(sparse, values)
    print("\nAttention output (sparse @ values):")
    print(output)
    
    # Verify with dense computation
    expected = sparse_dense @ values
    print("\nVerification (should match):")
    print(expected)
    print(f"\nMax difference: {np.max(np.abs(output - expected)):.8f}")


def example_attention_mask():
    """Example 4: Creating attention masks for transformers"""
    print("\n" + "=" * 60)
    print("Example 4: Attention Mask Patterns")
    print("=" * 60)
    
    seq_len = 8
    
    # Local windowed attention (window size = 2)
    print("\n1. Local Windowed Attention (window=2)")
    rows, cols = [], []
    window = 2
    for i in range(seq_len):
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            rows.append(i)
            cols.append(j)
    
    local_mask = SparseTensor.from_triplets(
        rows, cols, [1.0] * len(rows), shape=(seq_len, seq_len)
    )
    print(f"Sparsity: {local_mask}")
    print("Pattern:")
    print(local_mask.to_dense())
    
    # Global token attention (first token is global)
    print("\n2. Global Token Attention (token 0 is global)")
    rows, cols = [], []
    
    # Local connections (same as above)
    for i in range(seq_len):
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            rows.append(i)
            cols.append(j)
    
    # Add global connections to/from token 0
    global_idx = 0
    for i in range(seq_len):
        if i != global_idx:
            rows.append(global_idx)
            cols.append(i)
            rows.append(i)
            cols.append(global_idx)
    
    global_mask = SparseTensor.from_triplets(
        rows, cols, [1.0] * len(rows), shape=(seq_len, seq_len)
    )
    global_mask = global_mask.coalesce()  # Remove duplicates
    print(f"Sparsity: {global_mask}")
    print("Pattern (1 shows connection):")
    print(global_mask.to_dense())


def example_memory_savings():
    """Example 5: Memory savings with sparse matrices"""
    print("\n" + "=" * 60)
    print("Example 5: Memory Savings")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 5000]
    density = 0.05  # 5% non-zero
    
    print(f"\nDensity: {density * 100}%")
    print(f"{'Size':<10} {'Dense (MB)':<15} {'Sparse (MB)':<15} {'Savings':<10}")
    print("-" * 60)
    
    for size in sizes:
        # Dense memory: size * size * 4 bytes (float32)
        dense_memory = size * size * 4 / (1024 * 1024)
        
        # Sparse memory: nnz * (2 * 8 + 4) bytes (2 int64 indices + 1 float32)
        nnz = int(size * size * density)
        sparse_memory = nnz * (2 * 8 + 4) / (1024 * 1024)
        
        savings = (1 - sparse_memory / dense_memory) * 100
        
        print(f"{size:<10} {dense_memory:<15.2f} {sparse_memory:<15.2f} {savings:<10.1f}%")


def example_performance_comparison():
    """Example 6: Performance comparison"""
    print("\n" + "=" * 60)
    print("Example 6: Performance Comparison")
    print("=" * 60)
    
    import time
    
    size = 1000
    density = 0.01
    
    print(f"\nMatrix size: {size} x {size}")
    print(f"Density: {density * 100}%")
    
    # Create sparse matrix
    sparse = create_random_sparse((size, size), density=density, seed=42)
    sparse_dense = sparse.to_dense()
    
    # Create dense multiplication target
    dense_B = np.random.randn(size, 100).astype(np.float32)
    
    # Time sparse multiplication
    start = time.time()
    result_sparse = sparse_dense_matmul(sparse, dense_B)
    sparse_time = time.time() - start
    
    # Time dense multiplication
    start = time.time()
    result_dense = sparse_dense @ dense_B
    dense_time = time.time() - start
    
    print(f"\nSparse matmul time: {sparse_time:.4f} seconds")
    print(f"Dense matmul time: {dense_time:.4f} seconds")
    print(f"Speedup: {dense_time / sparse_time:.2f}x")
    
    # Verify correctness
    max_diff = np.max(np.abs(result_sparse - result_dense))
    print(f"\nMax difference: {max_diff:.8f} (should be very small)")


if __name__ == "__main__":
    # Run all examples
    example_basic_creation()
    example_basic_operations()
    example_matrix_multiplication()
    example_attention_mask()
    example_memory_savings()
    
    # Uncomment for performance test (takes a bit longer)
    # example_performance_comparison()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
