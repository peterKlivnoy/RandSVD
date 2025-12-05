import numpy as np
from .structured_sketch import srht_operator, srft_operator, slow_gaussian_operator
from .sparse_sketching import countsketch_operator, sparse_sign_embedding


def randSVD(A, k, p, q=0, sketch_type='gaussian'):
    """
    Computes the randomized SVD of a matrix A using q power iterations and 
    a specified sketch_type.

    A is a nxm input matrix; 
    k is the approximation rank; 
    p is the oversampling parameter; 
    q is the number of power iterations (Subspace Iteration).
    sketch_type is the sketching method:
        - 'gaussian': Fast BLAS-optimized O(mnl) [DEFAULT]
        - 'srht': Subsampled Random Hadamard Transform O(mn log n)
        - 'srft': Subsampled Random Fourier Transform O(mn log n) [FASTEST for structured]
        - 'countsketch': Sparse embedding O(ζnl) where ζ is sparsity [BEST for sparse matrices]
        - 'sparse_sign': Generalized sparse embedding with variable sparsity
        - 'slow_gaussian': Slow Python loop O(mnl) [for testing only]
    """
    rng = np.random.default_rng(seed=0)
    n, m = A.shape[0], A.shape[1]
    l = k + p

    # --- STAGE A: Find an orthonormal basis (Sketching) ---
    if sketch_type == 'gaussian':
        # Fast, compiled O(mn*l) using optimized BLAS
        omega = rng.normal(loc=0, scale=1, size=(m, l))
        Y = A @ omega
    elif sketch_type == 'srht':
        # Structured Hadamard: O(mn log n)
        Y = srht_operator(A, l)
    elif sketch_type == 'srft':
        # Structured Fourier: O(mn log n), typically faster than SRHT
        Y = srft_operator(A, l)
    elif sketch_type == 'slow_gaussian':
        # Inefficient Python loop O(mn*l), used for complexity comparison
        Y = slow_gaussian_operator(A, l)
    elif sketch_type == 'countsketch':
        # Sparse embedding: O(ζnl) where ζ is sparsity per column
        Y = countsketch_operator(A, l)
    elif sketch_type.startswith('sparse_sign'):
        # Generalized sparse embedding with variable sparsity
        # Parse sparsity from sketch_type (e.g., 'sparse_sign_2' means sparsity=2)
        if '_' in sketch_type and sketch_type.split('_')[-1].isdigit():
            sparsity = int(sketch_type.split('_')[-1])
        else:
            sparsity = 1  # Default to CountSketch
        Y = sparse_sign_embedding(A, l, sparsity=sparsity)
    else:
        raise ValueError(f"Invalid sketch_type '{sketch_type}'. Must be 'gaussian', 'srht', "
                        f"'srft', 'countsketch', 'sparse_sign', or 'slow_gaussian'.")
        
    # --- Subspace Iteration (Power Method) for robustness ---
    for i in range(q):
        # QR on Y to get orthonormal basis Q_i (stabilization)
        Q, _ = np.linalg.qr(Y)
        
        # Apply A.T (implicit AA.T iteration)
        Y = A.T @ Q
        
        # QR on Y to get orthonormal basis Q_i+1 (stabilization)
        Q, _ = np.linalg.qr(Y)
        
        # Apply A: Y = A @ Q (final step of the power iteration)
        Y = A @ Q
    
    # Compute the final orthonormal basis Q for the range of Y 
    Q, _ = np.linalg.qr(Y) 
    
    # --- STAGE B: Project A onto the low-dim subspace ---
    # Form the small matrix B
    B = Q.T @ A
    
    # Compute the deterministic SVD of the small matrix B
    U_delta, Sig, V_transpose = np.linalg.svd(B, full_matrices=False)
    
    # Reconstruct the left singular vectors for A
    U = Q @ U_delta
    
    # Truncate and Return
    U_k = U[:, :k]
    Sig_k = Sig[:k]
    V_transpose_k = V_transpose[:k, :]

    return U_k, Sig_k, V_transpose_k



