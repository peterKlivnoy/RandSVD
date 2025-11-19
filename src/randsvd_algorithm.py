import numpy as np
from .structured_sketch import srht_operator, slow_gaussian_operator 

def randSVD(A, k, p, q=0, sketch_type='gaussian'):
    """
    Computes the randomized SVD of a matrix A using q power iterations and 
    a specified sketch_type ('gaussian', 'srht', or 'slow_gaussian').

    A is a nxm input matrix; 
    k is the approximation rank; 
    p is the oversampling parameter; 
    q is the number of power iterations (Subspace Iteration).
    sketch_type is the sketching method ('gaussian', 'srht', or 'slow_gaussian').
    """
    rng = np.random.default_rng(seed=0)
    n, m = A.shape[0], A.shape[1]
    l = k + p

    # --- STAGE A: Find an orthonormal basis (Sketching) ---
    if sketch_type == 'gaussian':
        # Fast, compiled O(mn*l)
        omega = rng.normal(loc=0, scale=1, size=(m, l))
        Y = A @ omega
    elif sketch_type == 'srht':
        # Structured, intended O(mn log n)
        Y = srht_operator(A, l)
    elif sketch_type == 'slow_gaussian':
        # Inefficient Python loop O(mn*l), used for complexity comparison
        Y = slow_gaussian_operator(A, l)
    else:
        raise ValueError("Invalid sketch_type. Must be 'gaussian', 'srht', or 'slow_gaussian'.")
        
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


def randSVD_HH(A, k, p):
    """
    A is a nxm input matrix; k is the approximation rank; p is the oversampling parameter
    """
    rng = np.random.default_rng(seed=0)
    n, m = A.shape[0], A.shape[1]


    omega = rng.normal(loc=0, scale=1, size=(m, k+p))
    Y = A @ omega
    Q, _ = householder_qr(Y)
    B = Q.T @ A
    U_delta, Sig, V_transpose = np.linalg.svd(B,full_matrices=False)
    U = Q @ U_delta
    
    return U, Sig, V_transpose


def householder_qr(A):
    """
    Performs a QR factorization of a matrix A using Householder reflections.
    
    Arguments:
    A -- An m x n matrix
    
    Returns:
    Q -- An m x m orthogonal matrix
    R -- An m x n upper triangular matrix
    """
  
    m, n = A.shape
    
    # Initialize R as a copy of A. R will be transformed in-place.
    R = np.copy(A)
    
    # Initialize Q as an m x m identity matrix.
    # We will apply the same transformations to Q.
    Q = np.eye(m)
    
    # Loop over the columns (k = 0 to n-1)
    for k in range(n):
        x = R[k:m, k] 

        norm_x = np.sqrt(np.sum(np.pow(x, 2)))
        sign_x0 = 1.0 if x[0] >= 0 else -1.0
        
        alpha_k = -sign_x0 * norm_x
        
   
        v_k = np.zeros(m)
        
     
        v_k[k:m] = x
      
        v_k[k] = v_k[k] - alpha_k

        beta_k = np.sum(np.pow(v_k, 2))

        if beta_k != 0:
            
            for j in range(k, n):
                # gamma_j = v_k^T * a_j (where a_j is col j of R)
                gamma_j = np.dot(v_k, R[:, j])
                
                # a_j = a_j - (2 * gamma_j / beta_k) * v_k
                R[:, j] = R[:, j] - (2 * gamma_j / beta_k) * v_k

            for j in range(m):
                # gamma_j_q = v_k^T * q_j (where q_j is col j of Q)
                gamma_j_q = np.dot(v_k, Q[:, j])
                
                # q_j = q_j - (2 * gamma_j_q / beta_k) * v_k
                Q[:, j] = Q[:, j] - (2 * gamma_j_q / beta_k) * v_k

    return Q.T, R
