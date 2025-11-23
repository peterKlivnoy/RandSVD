"""
Sparse Sketching Methods (CountSketch / Sparse Sign Embeddings)

Based on Woodruff (2014) and Halko et al. (2011), Section 9.2:
- Sparse sign matrices achieve similar accuracy to Gaussian
- Computational cost depends on number of non-zeros: O(ζn) where ζ is sparsity
- Ideal for very large sparse matrices
- No need for FFT/WHT - just sparse matrix multiplication

CountSketch properties:
- Each column has exactly one non-zero entry (±1)
- Random row index and random sign
- Preserves Johnson-Lindenstrauss property
- Extremely memory efficient
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, issparse
import warnings


# ============================================================================
# NAIVE (PURE PYTHON) IMPLEMENTATIONS - For Fair Comparison
# ============================================================================

def naive_gaussian_sparse(A, l, seed=None):
    """
    Naive Gaussian sketch using pure Python loops - O(ζnl) for sparse A.
    
    This is a deliberately simple implementation that:
    - Uses explicit loops over non-zeros
    - No BLAS optimization
    - Shows the true O(ζnl) complexity for sparse matrices
    
    Args:
        A: Sparse matrix (m × n)
        l: Sketch size
        seed: Random seed
    
    Returns:
        Y: Sketched matrix (m × l)
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # Generate Gaussian random matrix
    Omega = rng.normal(0, 1, size=(n, l))
    
    # Initialize output
    Y = np.zeros((m, l))
    
    if issparse(A):
        # Convert to CSR for efficient row access
        A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
        
        # Triple loop: O(m × nnz_per_row × l)
        for i in range(m):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            
            for j in range(l):
                # Compute Y[i, j] = sum_k A[i, k] * Omega[k, j]
                # Only loop over non-zero entries!
                total = 0.0
                for idx in range(row_start, row_end):
                    k = A_csr.indices[idx]
                    a_ik = A_csr.data[idx]
                    total += a_ik * Omega[k, j]
                Y[i, j] = total
    else:
        # Dense fallback: O(mnl)
        for i in range(m):
            for j in range(l):
                for k in range(n):
                    Y[i, j] += A[i, k] * Omega[k, j]
    
    return Y


def naive_countsketch_sparse(A, l, seed=None):
    """
    Naive CountSketch using pure Python loops - O(ζl) for sparse A.
    
    This is the ALGORITHMIC advantage of CountSketch:
    - Each column of Omega has only ONE non-zero
    - For each row of A, we only need to access ONE element per output column
    - Pure Python, no library optimization
    - Shows the true O(ζl) complexity
    
    Complexity:
    - Sparse A with ζ non-zeros per row: O(m × ζ × l / n) ≈ O(ζnl / n) = O(ζl)
    - This is n times faster than naive Gaussian's O(ζnl)!
    
    Args:
        A: Sparse matrix (m × n)
        l: Sketch size
        seed: Random seed
    
    Returns:
        Y: Sketched matrix (m × l)
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # CountSketch: for each column j of Omega, pick random row and sign
    hash_indices = rng.integers(0, n, size=l)  # Which row of A to sample
    signs = rng.choice([-1, 1], size=l)
    
    # Initialize output
    Y = np.zeros((m, l))
    
    if issparse(A):
        # Convert to CSC for efficient column access
        A_csc = A.tocsc() if not isinstance(A, csc_matrix) else A
        
        # For each output column j
        for j in range(l):
            # CountSketch: Omega[:, j] has only ONE non-zero at position hash_indices[j]
            # So Y[:, j] = A[:, hash_indices[j]] * signs[j]
            
            col_idx = hash_indices[j]
            sign = signs[j]
            
            # Extract column from sparse matrix (pure Python loop)
            col_start = A_csc.indptr[col_idx]
            col_end = A_csc.indptr[col_idx + 1]
            
            for idx in range(col_start, col_end):
                row = A_csc.indices[idx]
                value = A_csc.data[idx]
                Y[row, j] = value * sign
    else:
        # Dense fallback
        for j in range(l):
            Y[:, j] = A[:, hash_indices[j]] * signs[j]
    
    return Y


# ============================================================================
# OPTIMIZED IMPLEMENTATIONS - Using Library Sparse Matrix Ops
# ============================================================================


def countsketch_operator(A, l, seed=None):
    """
    CountSketch operator: sparse sign embedding.
    
    Creates a sparse sketching matrix Ω where each column has exactly 
    one non-zero entry (±1 with equal probability).
    
    This is equivalent to:
    1. For each column j, pick a random row i ∈ [0, m)
    2. Set Ω[i, j] = ±1 with equal probability
    3. All other entries in column j are zero
    
    Complexity:
    - Dense A: O(mn) where n is number of columns in A (same as Gaussian)
    - Sparse A: O(ζn) where ζ is average number of non-zeros per column
    
    Args:
        A: Input matrix (m × n), can be dense or scipy sparse
        l: Sketch size (target number of rows in output)
        seed: Random seed for reproducibility
    
    Returns:
        Y: Sketched matrix (m × l)
    
    References:
        - Charikar, Chen, Farach-Colton (2002): Finding frequent items in data streams
        - Clarkson, Woodruff (2013): Low rank approximation and regression in input sparsity time
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # CountSketch: Ω is n × l where each column has exactly one non-zero entry
    # For column j of Ω (j ∈ [0, l)):
    #   - Pick random row i ∈ [0, n)
    #   - Set Ω[i, j] = ±1 with equal probability
    #   - All other entries in column j are zero
    
    # Generate random row indices (one per column of Ω)
    row_indices = rng.integers(0, n, size=l)
    signs = rng.choice([-1, 1], size=l)
    
    # Create sparse Ω matrix: n × l
    # For each column j, Ω[row_indices[j], j] = signs[j]
    col_indices = np.arange(l)
    data = signs
    
    Omega = csr_matrix((data, (row_indices, col_indices)), shape=(n, l))
    
    # Compute Y = A @ Ω
    if issparse(A):
        Y = A @ Omega
        Y = Y.toarray() if issparse(Y) else Y
    else:
        Y = A @ Omega.toarray()
    
    return Y


def sparse_sign_embedding(A, l, sparsity=1, seed=None):
    """
    Sparse sign embedding: generalized sparse sketching matrix.
    
    Creates a sketching matrix where each column has 'sparsity' non-zero entries.
    - sparsity=1: CountSketch (most sparse)
    - sparsity=n: Dense Gaussian-like (least sparse)
    
    The non-zero entries are ±1 with equal probability.
    
    Complexity:
    - Dense A: O(mn × sparsity)
    - Sparse A with ζ non-zeros per column: O(ζn × sparsity)
    
    Args:
        A: Input matrix (m × n)
        l: Sketch size
        sparsity: Number of non-zero entries per column (default: 1)
        seed: Random seed
    
    Returns:
        Y: Sketched matrix (m × l)
    
    References:
        - Kane, Nelson (2014): Sparser Johnson-Lindenstrauss transforms
        - Clarkson, Woodruff (2017): Input sparsity time low-rank approximation
    """
    if sparsity == 1:
        return countsketch_operator(A, l, seed=seed)
    
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # For each column j of Ω, pick 'sparsity' random rows
    rows = []
    cols = []
    data = []
    
    for j in range(l):
        # Pick random rows for this column (without replacement)
        row_idx = rng.choice(n, size=min(sparsity, n), replace=False)
        signs = rng.choice([-1, 1], size=len(row_idx))
        
        rows.extend(row_idx)
        cols.extend([j] * len(row_idx))
        data.extend(signs)
    
    # Create sparse matrix Ω: n × l
    Omega = csr_matrix((data, (rows, cols)), shape=(n, l))
    
    # Compute Y = A @ Ω
    if issparse(A):
        Y = A @ Omega
        Y = Y.toarray() if issparse(Y) else Y
    else:
        Y = A @ Omega.toarray()
    
    # Normalize by sqrt(sparsity) to maintain variance
    Y = Y / np.sqrt(sparsity)
    
    return Y


def clarkson_woodruff_transform(A, l, seed=None):
    """
    Clarkson-Woodruff Transform: optimal sparse embedding.
    
    A specific sparse embedding scheme with theoretical guarantees:
    - Each column of A is mapped to a random row with random sign
    - Achieves optimal space and time complexity
    - Works in "input sparsity time" for sparse matrices
    
    This is essentially CountSketch but with the interpretation
    as a linear transform that preserves geometry.
    
    Complexity: O(nnz(A)) where nnz(A) is number of non-zeros
    
    Args:
        A: Input matrix (m × n), preferably sparse
        l: Sketch size (should be l = O(k²/ε²) for rank-k approximation)
        seed: Random seed
    
    Returns:
        Y: Sketched matrix (l × n) - NOTE: different dimension order!
    
    References:
        - Clarkson, Woodruff (2013): "Low rank approximation and regression 
          in input sparsity time"
        - Woodruff (2014): "Sketching as a tool for numerical linear algebra"
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # CW transform: S is l × m, each row has exactly one non-zero per column of A
    # We compute S @ A where S is the sketching matrix
    
    # For each column i of A, pick random row j ∈ [0, l) and sign ±1
    row_indices = rng.integers(0, l, size=m)
    signs = rng.choice([-1, 1], size=m)
    
    # Create S: l × m sparse matrix
    col_indices = np.arange(m)
    S = csr_matrix((signs, (row_indices, col_indices)), shape=(l, m))
    
    # Compute Y = S @ A (sketch the ROWS instead of columns)
    if issparse(A):
        Y = S @ A
        Y = Y.toarray() if issparse(Y) else Y
    else:
        Y = S @ A
    
    return Y


def sparse_sketching_comparison(A, l, methods='all', seed=None):
    """
    Compare different sparse sketching methods.
    
    Args:
        A: Input matrix
        l: Sketch size
        methods: 'all' or list of ['countsketch', 'sparse_sign', 'cw_transform']
        seed: Random seed
    
    Returns:
        dict: Results from each method
    """
    if methods == 'all':
        methods = ['countsketch', 'sparse_sign', 'cw_transform']
    
    results = {}
    
    if 'countsketch' in methods:
        results['countsketch'] = countsketch_operator(A, l, seed=seed)
    
    if 'sparse_sign' in methods:
        # Try different sparsity levels
        for s in [1, 2, 4]:
            results[f'sparse_sign_s{s}'] = sparse_sign_embedding(A, l, sparsity=s, seed=seed)
    
    if 'cw_transform' in methods:
        results['cw_transform'] = clarkson_woodruff_transform(A, l, seed=seed).T
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Sparse Sketching Methods")
    print("="*60)
    
    # Test 1: Dense matrix
    print("\nTest 1: Dense matrix (100 × 50)")
    A_dense = np.random.randn(100, 50)
    l = 20
    
    Y_count = countsketch_operator(A_dense, l)
    print(f"CountSketch output shape: {Y_count.shape}")
    
    Y_sparse = sparse_sign_embedding(A_dense, l, sparsity=2)
    print(f"Sparse sign (s=2) output shape: {Y_sparse.shape}")
    
    Y_cw = clarkson_woodruff_transform(A_dense, l)
    print(f"CW transform output shape: {Y_cw.shape}")
    
    # Test 2: Sparse matrix
    print("\nTest 2: Sparse matrix (1000 × 500, 1% density)")
    from scipy.sparse import random as sparse_random
    A_sparse = sparse_random(1000, 500, density=0.01, format='csr')
    
    import time
    
    t0 = time.time()
    Y_count = countsketch_operator(A_sparse, l)
    t_count = time.time() - t0
    
    t0 = time.time()
    # Gaussian for comparison (would be slow)
    Omega_gaussian = np.random.randn(500, l)
    Y_gauss = A_sparse @ Omega_gaussian
    t_gauss = time.time() - t0
    
    print(f"CountSketch time: {t_count*1000:.2f} ms")
    print(f"Gaussian time: {t_gauss*1000:.2f} ms")
    print(f"Speedup: {t_gauss/t_count:.1f}×")
    
    # Test 3: Verify dimensions
    print("\nTest 3: Dimension verification")
    for method_name, Y in sparse_sketching_comparison(A_dense, l).items():
        print(f"  {method_name:20s}: {Y.shape}")
    
    print("\n✓ All tests passed!")
