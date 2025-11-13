"""
Gradient Tests for Sparse Operations
Verifies backward passes are correct using numerical differentiation

This file should be placed at: tests/hw3/test_gradients.py (optional - for gradient verification)
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from needle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'python'))

from needle.backend_ndarray.sparse_tensor import SparseTensor, create_random_sparse
from needle.ops.sparse_ops import (
    SparseMatMul, SparseMulScalar, SparseTranspose,
    SparseAdd, SparseElementwiseMul, SparseReLU
)


class Node:
    """Simple node class to mimic autograd node for testing"""
    def __init__(self, inputs):
        self.inputs = inputs


def numerical_gradient_sparse(f, sparse_tensor, eps=1e-5):
    """
    Compute numerical gradient for a sparse tensor using finite differences.
    
    Args:
        f: function that takes SparseTensor and returns scalar
        sparse_tensor: SparseTensor to compute gradient for
        eps: perturbation size
        
    Returns:
        gradient values (same shape as sparse_tensor.values)
    """
    numerical_grad = np.zeros_like(sparse_tensor.values)
    
    for i in range(sparse_tensor.nnz):
        # Perturb value up
        sparse_plus = SparseTensor(
            sparse_tensor.indices.copy(),
            sparse_tensor.values.copy(),
            sparse_tensor.shape
        )
        sparse_plus.values[i] += eps
        f_plus = f(sparse_plus)
        
        # Perturb value down
        sparse_minus = SparseTensor(
            sparse_tensor.indices.copy(),
            sparse_tensor.values.copy(),
            sparse_tensor.shape
        )
        sparse_minus.values[i] -= eps
        f_minus = f(sparse_minus)
        
        # Finite difference
        numerical_grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return numerical_grad


def test_sparse_matmul_gradient():
    """Test gradient of sparse @ dense matrix multiplication"""
    print("\n" + "=" * 60)
    print("Testing SparseMatMul Gradient")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create inputs
    sparse_a = create_random_sparse((5, 4), density=0.5, seed=42)
    dense_b = np.random.randn(4, 3).astype(np.float32)
    
    # Forward pass
    op = SparseMatMul()
    output = op.compute(sparse_a, dense_b)
    
    # Create mock gradient (pretend this comes from downstream)
    out_grad = np.random.randn(*output.shape).astype(np.float32)
    
    # Backward pass
    node = Node([sparse_a, dense_b])
    grad_a, grad_b = op.gradient(out_grad, node)
    
    print(f"Forward: sparse{sparse_a.shape} @ dense{dense_b.shape} -> {output.shape}")
    print(f"Gradient shapes: grad_a={grad_a.shape if hasattr(grad_a, 'shape') else 'sparse'}, grad_b={grad_b.shape}")
    
    # Numerical gradient check for sparse input (A)
    def loss_fn_a(s):
        result = op.compute(s, dense_b)
        return np.sum(result * out_grad)
    
    numerical_grad_a = numerical_gradient_sparse(loss_fn_a, sparse_a)
    
    # Compare
    if isinstance(grad_a, SparseTensor):
        computed_grad_a = grad_a.values
    else:
        # Extract values at sparse positions
        computed_grad_a = grad_a[sparse_a.indices[0], sparse_a.indices[1]]
    
    max_error_a = np.max(np.abs(numerical_grad_a - computed_grad_a))
    print(f"\nGradient A (sparse input):")
    print(f"  Max error: {max_error_a:.8f}")
    print(f"  Status: {'✅ PASS' if max_error_a < 1e-4 else '❌ FAIL'}")
    
    # Numerical gradient check for dense input (B)
    def loss_fn_b(b_val):
        result = op.compute(sparse_a, b_val)
        return np.sum(result * out_grad)
    
    eps = 1e-5
    numerical_grad_b = np.zeros_like(dense_b)
    
    for i in range(dense_b.shape[0]):
        for j in range(dense_b.shape[1]):
            # Perturb up
            b_plus = dense_b.copy()
            b_plus[i, j] += eps
            f_plus = loss_fn_b(b_plus)
            
            # Perturb down
            b_minus = dense_b.copy()
            b_minus[i, j] -= eps
            f_minus = loss_fn_b(b_minus)
            
            numerical_grad_b[i, j] = (f_plus - f_minus) / (2 * eps)
    
    max_error_b = np.max(np.abs(numerical_grad_b - grad_b))
    print(f"\nGradient B (dense input):")
    print(f"  Max error: {max_error_b:.8f}")
    print(f"  Status: {'✅ PASS' if max_error_b < 1e-4 else '❌ FAIL'}")
    
    return max_error_a < 1e-4 and max_error_b < 1e-4


def test_sparse_scalar_mul_gradient():
    """Test gradient of scalar multiplication"""
    print("\n" + "=" * 60)
    print("Testing SparseMulScalar Gradient")
    print("=" * 60)
    
    np.random.seed(123)
    
    # Create input
    sparse_a = create_random_sparse((4, 4), density=0.4, seed=123)
    scalar = 2.5
    
    # Forward pass
    op = SparseMulScalar(scalar)
    output = op.compute(sparse_a)
    
    # Mock gradient
    out_grad_values = np.random.randn(sparse_a.nnz).astype(np.float32)
    out_grad = SparseTensor(sparse_a.indices.copy(), out_grad_values, sparse_a.shape)
    
    # Backward pass
    node = Node([sparse_a])
    grad_a = op.gradient(out_grad, node)
    
    print(f"Forward: {scalar} * sparse{sparse_a.shape}")
    print(f"Output is sparse: {isinstance(output, SparseTensor)}")
    
    # Check: gradient should be out_grad * scalar
    expected_grad_values = out_grad_values * scalar
    actual_grad_values = grad_a.values if isinstance(grad_a, SparseTensor) else grad_a
    
    max_error = np.max(np.abs(expected_grad_values - actual_grad_values))
    print(f"\nGradient check:")
    print(f"  Max error: {max_error:.8f}")
    print(f"  Status: {'✅ PASS' if max_error < 1e-6 else '❌ FAIL'}")
    
    return max_error < 1e-6


def test_sparse_transpose_gradient():
    """Test gradient of transpose"""
    print("\n" + "=" * 60)
    print("Testing SparseTranspose Gradient")
    print("=" * 60)
    
    np.random.seed(456)
    
    # Create non-square matrix to make transpose obvious
    sparse_a = create_random_sparse((3, 5), density=0.4, seed=456)
    
    # Forward pass
    op = SparseTranspose()
    output = op.compute(sparse_a)
    
    print(f"Forward: sparse{sparse_a.shape} -> sparse{output.shape}")
    print(f"Input nnz: {sparse_a.nnz}, Output nnz: {output.nnz}")
    
    # Mock gradient (should have transposed shape)
    out_grad_values = np.random.randn(output.nnz).astype(np.float32)
    out_grad = SparseTensor(output.indices.copy(), out_grad_values, output.shape)
    
    # Backward pass
    node = Node([sparse_a])
    grad_a = op.gradient(out_grad, node)
    
    # Check: gradient should be transpose of out_grad
    print(f"\nGradient shape: {grad_a.shape}")
    print(f"Expected shape: {sparse_a.shape}")
    
    # Transpose should swap shape
    correct_shape = grad_a.shape == sparse_a.shape
    print(f"  Shape correct: {'✅ PASS' if correct_shape else '❌ FAIL'}")
    
    # Values should be same (indices swapped)
    grad_a_T = grad_a.transpose()
    values_match = np.allclose(grad_a_T.values, out_grad.values)
    print(f"  Values correct: {'✅ PASS' if values_match else '❌ FAIL'}")
    
    return correct_shape and values_match


def test_sparse_add_gradient():
    """Test gradient of sparse addition"""
    print("\n" + "=" * 60)
    print("Testing SparseAdd Gradient")
    print("=" * 60)
    
    np.random.seed(789)
    
    # Create two sparse tensors
    sparse_a = create_random_sparse((3, 3), density=0.5, seed=789)
    sparse_b = create_random_sparse((3, 3), density=0.5, seed=790)
    
    # Forward pass
    op = SparseAdd()
    output = op.compute(sparse_a, sparse_b)
    
    print(f"Forward: sparse{sparse_a.shape} + sparse{sparse_b.shape}")
    print(f"Input A nnz: {sparse_a.nnz}, Input B nnz: {sparse_b.nnz}")
    print(f"Output nnz: {output.nnz}")
    
    # Mock gradient
    out_grad = create_random_sparse((3, 3), density=0.6, seed=791)
    
    # Backward pass
    node = Node([sparse_a, sparse_b])
    grad_a, grad_b = op.gradient(out_grad, node)
    
    # Check: both gradients should equal out_grad
    correct_a = (grad_a is out_grad) or np.allclose(grad_a.values, out_grad.values)
    correct_b = (grad_b is out_grad) or np.allclose(grad_b.values, out_grad.values)
    
    print(f"\nGradient A equals out_grad: {'✅ PASS' if correct_a else '❌ FAIL'}")
    print(f"Gradient B equals out_grad: {'✅ PASS' if correct_b else '❌ FAIL'}")
    
    return correct_a and correct_b


def test_sparse_relu_gradient():
    """Test gradient of ReLU"""
    print("\n" + "=" * 60)
    print("Testing SparseReLU Gradient")
    print("=" * 60)
    
    # Create sparse tensor with mix of positive and negative values
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 2]
    values = [1.0, -2.0, 3.0, -4.0, -5.0, 6.0]
    sparse_a = SparseTensor.from_triplets(rows, cols, values, (3, 3))
    
    print(f"Input values: {sparse_a.values}")
    print(f"Positive values: {np.sum(sparse_a.values > 0)}")
    print(f"Negative values: {np.sum(sparse_a.values <= 0)}")
    
    # Forward pass
    op = SparseReLU()
    output = op.compute(sparse_a)
    
    print(f"\nForward: ReLU(sparse{sparse_a.shape})")
    print(f"Input nnz: {sparse_a.nnz}, Output nnz: {output.nnz}")
    print(f"Output values: {output.values}")
    
    # Mock gradient at output positions
    out_grad = SparseTensor(
        output.indices.copy(),
        np.ones(output.nnz, dtype=np.float32),
        output.shape
    )
    
    # Backward pass
    node = Node([sparse_a])
    grad_a = op.gradient(out_grad, node)
    
    print(f"\nGradient computed")
    print(f"Gradient nnz: {grad_a.nnz if isinstance(grad_a, SparseTensor) else 'N/A'}")
    
    # Check: gradient should pass through where input > 0, zero elsewhere
    # Since negative inputs were zeroed in forward, they shouldn't get gradients
    expected_nnz = np.sum(sparse_a.values > 0)
    actual_nnz = grad_a.nnz if isinstance(grad_a, SparseTensor) else len(grad_a)
    
    print(f"Expected nnz in gradient: {expected_nnz}")
    print(f"Actual nnz in gradient: {actual_nnz}")
    print(f"Status: {'✅ PASS' if actual_nnz == expected_nnz else '❌ FAIL'}")
    
    return actual_nnz == expected_nnz


def run_all_gradient_tests():
    """Run all gradient tests"""
    print("\n" + "=" * 70)
    print(" " * 20 + "GRADIENT TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    results['SparseMatMul'] = test_sparse_matmul_gradient()
    results['SparseMulScalar'] = test_sparse_scalar_mul_gradient()
    results['SparseTranspose'] = test_sparse_transpose_gradient()
    results['SparseAdd'] = test_sparse_add_gradient()
    results['SparseReLU'] = test_sparse_relu_gradient()
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    for op_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{op_name:<25} {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 70)
    
    if all_passed:
        print(" " * 15 + "🎉 ALL GRADIENT TESTS PASSED! 🎉")
        print("\nBackward passes are correct. Safe to use in training.")
    else:
        print(" " * 15 + "⚠️  SOME TESTS FAILED ⚠️")
        print("\nReview failed operations before using in training.")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_gradient_tests()
    exit(0 if success else 1)
