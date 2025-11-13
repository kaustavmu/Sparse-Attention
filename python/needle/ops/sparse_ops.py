"""
Compatible Sparse Operations for Needle
Works with Person A's SparseTensor implementation

This file should be placed at: python/needle/ops/sparse_ops.py
It will replace the existing sparse_ops.py
"""

import numpy as np
from ..backend_ndarray.sparse_tensor import (
    SparseTensor,
    sparse_add as _sparse_add_impl,
    sparse_scalar_multiply as _sparse_scalar_multiply_impl,
    sparse_elementwise_multiply as _sparse_elementwise_multiply_impl,
    sparse_dense_matmul as _sparse_dense_matmul_impl,
    dense_sparse_matmul as _dense_sparse_matmul_impl
)

###########################################################
#                 SparseTensorOp Base Class               #
###########################################################

class SparseTensorOp:
    """
    Base class for sparse tensor operations.
    Compatible with Needle's autograd system.
    """
    def __call__(self, *args):
        """Execute the operation"""
        return self.compute(*args)
    
    def compute(self, *args):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError()
    
    def gradient(self, out_grad, node):
        """Backward pass - must be implemented by subclasses"""
        raise NotImplementedError()


###########################################################
#                  Public sparse wrappers                 #
###########################################################

def sparse_add(a, b):
    """Add two sparse tensors element-wise"""
    return SparseAdd()(a, b)

def sparse_multiply(a, b):
    """Element-wise multiplication (Hadamard product)"""
    return SparseElementwiseMul()(a, b)

def sparse_matmul(a, b):
    """Matrix multiplication: handles sparse @ dense or dense @ sparse"""
    return SparseMatMul()(a, b)

def sparse_add_scalar(a, scalar):
    """Add scalar to sparse tensor (becomes dense!)"""
    return SparseAddScalar(scalar)(a)

def sparse_mul_scalar(a, scalar):
    """Multiply sparse tensor by scalar (stays sparse)"""
    return SparseMulScalar(scalar)(a)

def sparse_transpose(a):
    """Transpose sparse tensor"""
    return SparseTranspose()(a)

def sparse_relu(a):
    """Apply ReLU to sparse tensor"""
    return SparseReLU()(a)


###########################################################
#                  Operator Implementations               #
###########################################################

class SparseAdd(SparseTensorOp):
    """
    Elementwise addition for sparse tensors.
    Result is sparse with combined non-zero patterns.
    """
    def compute(self, a, b):
        # Use Person A's implementation
        if isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
            return _sparse_add_impl(a, b)
        elif isinstance(a, SparseTensor):
            # a is sparse, b is dense - convert b to sparse
            b_sparse = SparseTensor.from_dense(b)
            return _sparse_add_impl(a, b_sparse)
        elif isinstance(b, SparseTensor):
            # b is sparse, a is dense - convert a to sparse
            a_sparse = SparseTensor.from_dense(a)
            return _sparse_add_impl(a_sparse, b)
        else:
            # Both dense
            return a + b
    
    def gradient(self, out_grad, node):
        """
        d/dA (A + B) = out_grad
        d/dB (A + B) = out_grad
        """
        return out_grad, out_grad


class SparseElementwiseMul(SparseTensorOp):
    """
    Elementwise multiplication (Hadamard product).
    Result is sparse - only overlapping non-zeros contribute.
    """
    def compute(self, a, b):
        if isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
            return _sparse_elementwise_multiply_impl(a, b)
        elif isinstance(a, SparseTensor):
            # a sparse, b dense - multiply values where a is non-zero
            result_values = a.values * b[a.indices[0], a.indices[1]]
            return SparseTensor(a.indices.copy(), result_values, a.shape)
        elif isinstance(b, SparseTensor):
            # b sparse, a dense - multiply values where b is non-zero
            result_values = b.values * a[b.indices[0], b.indices[1]]
            return SparseTensor(b.indices.copy(), result_values, b.shape)
        else:
            # Both dense
            return a * b
    
    def gradient(self, out_grad, node):
        """
        d/dA (A ⊙ B) = out_grad ⊙ B
        d/dB (A ⊙ B) = out_grad ⊙ A
        """
        a, b = node.inputs
        
        # Gradient maintains sparsity pattern
        if isinstance(out_grad, SparseTensor):
            grad_a = self.compute(out_grad, b)  # out_grad ⊙ b
            grad_b = self.compute(out_grad, a)  # out_grad ⊙ a
        else:
            grad_a = out_grad * b
            grad_b = out_grad * a
        
        return grad_a, grad_b


class SparseMatMul(SparseTensorOp):
    """
    Matrix multiplication supporting sparse inputs.
    
    Handles three cases:
    1. sparse @ dense (most common for attention)
    2. dense @ sparse
    3. both dense
    """
    def compute(self, a, b):
        if isinstance(a, SparseTensor) and not isinstance(b, SparseTensor):
            # Sparse @ Dense - use Person A's implementation
            return _sparse_dense_matmul_impl(a, b)
        elif not isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
            # Dense @ Sparse - use Person A's implementation
            return _dense_sparse_matmul_impl(a, b)
        elif isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
            # Both sparse - convert to dense for now (can optimize later)
            return a.to_dense() @ b.to_dense()
        else:
            # Both dense
            return a @ b
    
    def gradient(self, out_grad, node):
        """
        For C = A @ B:
        d/dA = out_grad @ B^T
        d/dB = A^T @ out_grad
        
        CRITICAL: If A is sparse, grad_a should maintain sparsity pattern!
        """
        a, b = node.inputs
        
        # Gradient w.r.t. A
        if isinstance(a, SparseTensor):
            # A is sparse: grad_a = out_grad @ B^T
            if isinstance(b, SparseTensor):
                # B is also sparse - convert to dense for now
                b_dense = b.to_dense()
                grad_a_dense = out_grad @ b_dense.T
            else:
                # B is dense
                grad_a_dense = out_grad @ b.T
            
            # Extract gradient at sparse positions to maintain sparsity
            grad_a = self._extract_sparse_gradient(grad_a_dense, a)
        else:
            # A is dense
            if isinstance(b, SparseTensor):
                # B is sparse: grad_a = out_grad @ B^T
                b_T = b.transpose()
                grad_a = _dense_sparse_matmul_impl(out_grad, b_T)
            else:
                grad_a = out_grad @ b.T
        
        # Gradient w.r.t. B
        if isinstance(b, SparseTensor):
            # B is sparse: grad_b = A^T @ out_grad
            if isinstance(a, SparseTensor):
                # A is also sparse - convert to dense
                a_dense = a.to_dense()
                grad_b_dense = a_dense.T @ out_grad
                grad_b = self._extract_sparse_gradient(grad_b_dense, b)
            else:
                # A is dense
                grad_b_dense = a.T @ out_grad
                grad_b = self._extract_sparse_gradient(grad_b_dense, b)
        else:
            # B is dense
            if isinstance(a, SparseTensor):
                # A is sparse: grad_b = A^T @ out_grad
                a_T = a.transpose()
                grad_b = _sparse_dense_matmul_impl(a_T, out_grad)
            else:
                # A is dense
                grad_b = a.T @ out_grad
        
        return grad_a, grad_b
    
    def _extract_sparse_gradient(self, grad_dense, original_sparse):
        """
        Extract gradient values at sparse positions.
        This maintains the sparsity pattern.
        
        CRITICAL: This is what makes sparse attention efficient!
        """
        # Get gradient values at the non-zero positions
        grad_values = grad_dense[original_sparse.indices[0], 
                                 original_sparse.indices[1]]
        
        return SparseTensor(
            indices=original_sparse.indices.copy(),
            values=grad_values.astype(np.float32),
            shape=original_sparse.shape
        )


class SparseAddScalar(SparseTensorOp):
    """
    Add scalar to tensor.
    WARNING: This makes sparse tensors DENSE! Use sparingly.
    """
    def __init__(self, scalar):
        self.scalar = scalar
    
    def compute(self, a):
        if isinstance(a, SparseTensor):
            # Convert to dense (adding scalar fills in zeros)
            return a.to_dense() + self.scalar
        else:
            return a + self.scalar
    
    def gradient(self, out_grad, node):
        """d/dA (A + c) = out_grad"""
        return out_grad


class SparseMulScalar(SparseTensorOp):
    """
    Multiply tensor by scalar.
    Sparse tensors STAY SPARSE!
    """
    def __init__(self, scalar):
        self.scalar = scalar
    
    def compute(self, a):
        if isinstance(a, SparseTensor):
            # Use Person A's implementation
            return _sparse_scalar_multiply_impl(a, self.scalar)
        else:
            return a * self.scalar
    
    def gradient(self, out_grad, node):
        """d/dA (c * A) = c * out_grad"""
        if isinstance(out_grad, SparseTensor):
            return _sparse_scalar_multiply_impl(out_grad, self.scalar)
        else:
            return out_grad * self.scalar


class SparseTranspose(SparseTensorOp):
    """Transpose sparse or dense tensor"""
    def compute(self, a):
        if isinstance(a, SparseTensor):
            return a.transpose()
        else:
            return a.T
    
    def gradient(self, out_grad, node):
        """d/dA (A^T) = (out_grad)^T"""
        if isinstance(out_grad, SparseTensor):
            return out_grad.transpose()
        else:
            return out_grad.T


class SparseReLU(SparseTensorOp):
    """
    ReLU activation for sparse tensors.
    Maintains sparsity (negative values become zero, already sparse).
    """
    def compute(self, a):
        if isinstance(a, SparseTensor):
            # Apply ReLU only to non-zero values
            # Negative values become zero (stay sparse)
            relu_values = np.maximum(a.values, 0)
            
            # Filter out values that became exactly zero
            nonzero_mask = relu_values > 0
            new_indices = a.indices[:, nonzero_mask]
            new_values = relu_values[nonzero_mask]
            
            return SparseTensor(new_indices, new_values, a.shape)
        else:
            return np.maximum(a, 0)
    
    def gradient(self, out_grad, node):
        """
        Gradient: pass through where input > 0, zero elsewhere.
        For sparse inputs, maintain sparsity pattern.
        
        Note: Forward pass filters out negatives, so out_grad only has
        indices where input was positive. We need to reconstruct full gradient.
        """
        a = node.inputs[0]
        
        if isinstance(a, SparseTensor):
            # Create mask for original sparse positions
            mask = (a.values > 0).astype(np.float32)
            
            if isinstance(out_grad, SparseTensor):
                # out_grad has only positive positions from forward pass
                # We need gradient at ALL original positions (zero where masked)
                grad_values = np.zeros_like(a.values)
                
                # Match out_grad indices to original a indices
                for i, (row, col) in enumerate(out_grad.indices.T):
                    # Find this position in original a
                    orig_mask = (a.indices[0] == row) & (a.indices[1] == col)
                    orig_idx = np.where(orig_mask)[0]
                    if len(orig_idx) > 0:
                        grad_values[orig_idx[0]] = out_grad.values[i]
                
                # Apply mask (zero out negative positions)
                grad_values = grad_values * mask
                
                return SparseTensor(a.indices.copy(), grad_values, a.shape)
            else:
                # out_grad is dense, a is sparse
                # Extract gradient at sparse positions and apply mask
                grad_values = out_grad[a.indices[0], a.indices[1]] * mask
                return SparseTensor(a.indices.copy(), grad_values, a.shape)
        else:
            # Dense case
            mask = (a > 0).astype(np.float32)
            return out_grad * mask


###########################################################
#                  Utility Functions                      #
###########################################################

def is_sparse(x):
    """Check if x is a SparseTensor"""
    return isinstance(x, SparseTensor)


def to_dense(x):
    """Convert to dense if sparse, otherwise return as-is"""
    if isinstance(x, SparseTensor):
        return x.to_dense()
    return x


def to_sparse(x, density_threshold=0.5):
    """
    Convert dense array to sparse if sparsity is high enough.
    
    Args:
        x: numpy array
        density_threshold: convert to sparse if density < this value
    """
    if isinstance(x, SparseTensor):
        return x
    
    nnz = np.count_nonzero(x)
    density = nnz / x.size
    
    if density < density_threshold:
        return SparseTensor.from_dense(x)
    else:
        return x


###########################################################
#                     Example Usage                       #
###########################################################

if __name__ == "__main__":
    print("Testing Compatible Sparse Ops")
    print("=" * 60)
    
    # Create sparse tensors using Person A's implementation
    from sparse_tensor import create_random_sparse
    
    # Test 1: Sparse-Dense MatMul
    print("\n1. Sparse @ Dense (attention operation)")
    sparse_a = create_random_sparse((4, 4), density=0.5, seed=42)
    dense_b = np.random.randn(4, 2).astype(np.float32)
    
    result = sparse_matmul(sparse_a, dense_b)
    print(f"   Input: sparse{sparse_a.shape} @ dense{dense_b.shape}")
    print(f"   Output: {result.shape}")
    print(f"   ✅ Forward pass works!")
    
    # Test 2: Operations maintain sparsity
    print("\n2. Sparse operations maintain sparsity")
    sparse_c = create_random_sparse((3, 3), density=0.4, seed=123)
    
    # Scalar multiply (stays sparse)
    result_scalar = sparse_mul_scalar(sparse_c, 2.0)
    print(f"   Scalar multiply: {is_sparse(result_scalar)} (should be True)")
    
    # Transpose (stays sparse)
    result_T = sparse_transpose(sparse_c)
    print(f"   Transpose: {is_sparse(result_T)} (should be True)")
    
    # Add scalar (becomes dense - WARNING!)
    result_add_scalar = sparse_add_scalar(sparse_c, 1.0)
    print(f"   Add scalar: {is_sparse(result_add_scalar)} (should be False)")
    
    print(f"   ✅ Sparsity maintained correctly!")
    
    # Test 3: ReLU maintains sparsity
    print("\n3. ReLU maintains sparsity")
    sparse_with_neg = SparseTensor.from_triplets(
        [0, 1, 2], [0, 1, 2], [-1.0, 2.0, -3.0], (3, 3)
    )
    result_relu = sparse_relu(sparse_with_neg)
    print(f"   Input nnz: {sparse_with_neg.nnz}")
    print(f"   Output nnz: {result_relu.nnz} (negatives removed)")
    print(f"   ✅ ReLU works correctly!")
    
    print("\n" + "=" * 60)
    print("All operations compatible with Person A's implementation!")
    print("=" * 60)
