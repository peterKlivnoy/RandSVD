"""
Utility functions for randomized SVD experiments.

This module provides:
- Error computation functions (Frobenius, Spectral, Per-Vector)
- Iteration methods (Simultaneous Iteration, Block Krylov)
- Matrix generation utilities
"""

import numpy as np
import scipy.sparse as sp


# =============================================================================
# ITERATION METHODS
# =============================================================================

def simultaneous_iteration(A, k, q, seed=0):
    """
    Simultaneous Power Iteration (standard method from Algorithm 4.4 in Halko et al.).
    
    Computes Z such that ZZ^T A ≈ A_k by iteratively applying (AA^T) and
    keeping only the final result.
    
    Args:
        A: Input matrix (m × n), can be dense or sparse
        k: Target rank
        q: Number of power iterations
        seed: Random seed for reproducibility
    
    Returns:
        Z: Orthonormal basis (m × k) for approximate row space
    """
    m, n = A.shape
    np.random.seed(seed)
    Omega = np.random.randn(n, k)
    
    K = A @ Omega
    
    for _ in range(q):
        K, _ = np.linalg.qr(K)
        K = A @ (A.T @ K)
    
    Z, _ = np.linalg.qr(K)
    return Z


def block_krylov_iteration(A, k, q, seed=0):
    """
    Block Krylov Iteration (from Musco & Musco 2015).
    
    Keeps ALL intermediate powers: [AΩ, (AA^T)AΩ, ..., (AA^T)^q AΩ]
    This builds a richer subspace that captures more spectral information.
    
    Args:
        A: Input matrix (m × n), can be dense or sparse
        k: Target rank
        q: Number of power iterations
        seed: Random seed for reproducibility
    
    Returns:
        Z: Orthonormal basis (m × k) for approximate row space
    """
    m, n = A.shape
    np.random.seed(seed)
    Omega = np.random.randn(n, k)
    
    K_i = A @ Omega
    blocks = [K_i.copy()]
    
    for _ in range(q):
        K_i, _ = np.linalg.qr(K_i)
        K_i = A @ (A.T @ K_i)
        blocks.append(K_i.copy())
    
    # Stack all blocks to form full Krylov basis
    K = np.hstack(blocks)
    Z, _ = np.linalg.qr(K)
    
    # Truncate to top k directions via SVD
    B = Z.T @ A
    U_B, _, _ = np.linalg.svd(B, full_matrices=False)
    Z_k = Z @ U_B[:, :k]
    
    return Z_k


# =============================================================================
# ERROR METRICS (from Musco & Musco 2015)
# =============================================================================

def compute_frobenius_error(A, Z, true_sv, k):
    """
    Compute Frobenius (weak) error metric.
    
    Error = ||A - ZZ^T A||_F / ||A - A_k||_F - 1
    
    where ||A - A_k||_F = sqrt(sum(σ_i^2, i > k))
    
    Uses identity: ||A - ZZ^T A||_F^2 = ||A||_F^2 - ||Z^T A||_F^2
    to avoid forming dense residual for sparse matrices.
    """
    # Optimal Frobenius error
    opt_frob = np.sqrt(np.sum(true_sv[k:]**2))
    
    # Compute ||A||_F^2
    if sp.issparse(A):
        A_frob_sq = sp.linalg.norm(A, 'fro')**2
    else:
        A_frob_sq = np.linalg.norm(A, 'fro')**2
    
    # Compute ||Z^T A||_F^2
    ZtA = Z.T @ A
    ZtA_frob_sq = np.linalg.norm(ZtA, 'fro')**2
    
    # Residual norm
    residual_frob = np.sqrt(max(A_frob_sq - ZtA_frob_sq, 0))
    
    if opt_frob > 1e-14:
        return max(residual_frob / opt_frob - 1, 1e-10)
    return 1e-10


def compute_spectral_error(A, Z, true_sv, k, n_iter=20):
    """
    Compute Spectral (strong) error metric.
    
    Error = ||A - ZZ^T A||_2 / ||A - A_k||_2 - 1
    
    where ||A - A_k||_2 = σ_{k+1}
    
    Uses power iteration to estimate spectral norm without forming dense residual.
    """
    opt_spec = true_sv[k] if k < len(true_sv) else 1e-14
    
    # Cache Z^T A for efficiency
    ZtA = Z.T @ A
    
    def residual_matvec(v):
        """Compute (A - ZZ^T A) @ v"""
        Av = A @ v
        return Av - Z @ (Z.T @ Av)
    
    def residual_rmatvec(u):
        """Compute (A - ZZ^T A)^T @ u"""
        if sp.issparse(A):
            Atu = A.T @ u
            return Atu - ZtA.T @ (Z.T @ u)
        else:
            Atu = A.T @ u
            return Atu - A.T @ (Z @ (Z.T @ u))
    
    # Power iteration for spectral norm
    np.random.seed(999)
    v = np.random.randn(A.shape[1])
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        u = residual_matvec(v)
        v = residual_rmatvec(u)
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            break
        v = v / nv
    
    residual_spec = np.linalg.norm(residual_matvec(v))
    
    if opt_spec > 1e-14:
        return max(residual_spec / opt_spec - 1, 1e-10)
    return 1e-10


def compute_pervector_error(A, Z, true_sv, k):
    """
    Compute Per-Vector (strongest) error metric.
    
    Error = max_i |σ_i^2 - ||A^T z_i||^2| / σ_{k+1}^2
    
    This ensures EACH singular vector captures the correct variance.
    """
    sigma_kplus1_sq = true_sv[k]**2 if k < len(true_sv) else 1e-14
    
    ZtA = Z.T @ A
    
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        # ||A^T z_i||^2 = ||ZtA[i, :]||^2
        captured_var = np.linalg.norm(ZtA[i, :])**2
        true_var = true_sv[i]**2
        err = abs(true_var - captured_var) / sigma_kplus1_sq
        pervec_errors.append(err)
    
    return max(max(pervec_errors), 1e-10) if pervec_errors else 1e-10


def compute_all_errors(A, Z, true_sv, k):
    """
    Compute all three error metrics from Musco & Musco (2015).
    
    Returns:
        tuple: (frobenius_error, spectral_error, pervector_error)
    """
    frob = compute_frobenius_error(A, Z, true_sv, k)
    spec = compute_spectral_error(A, Z, true_sv, k)
    pervec = compute_pervector_error(A, Z, true_sv, k)
    return frob, spec, pervec


# =============================================================================
# MATRIX GENERATION
# =============================================================================

def create_slow_decay_matrix(m, n, decay_rate=0.5, seed=42):
    """
    Create matrix with slowly decaying singular values (hard case for SVD).
    
    σ_i = 1 / i^decay_rate
    
    Args:
        m: Number of rows
        n: Number of columns
        decay_rate: Controls spectral decay (smaller = harder)
            0.5: very slow decay, tiny gaps - HARD
            1.0: Zipf decay - moderate
            2.0: fast decay - easy
        seed: Random seed
    
    Returns:
        A: Matrix (m × n)
        singular_values: Array of true singular values
    """
    np.random.seed(seed)
    
    min_dim = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, min_dim))
    V, _ = np.linalg.qr(np.random.randn(n, min_dim))
    
    singular_values = 1.0 / (np.arange(1, min_dim + 1) ** decay_rate)
    
    A = U @ np.diag(singular_values) @ V.T
    
    return A, singular_values


def load_20newsgroups(max_features=15000):
    """
    Load 20 Newsgroups dataset as sparse TF-IDF matrix.
    
    Args:
        max_features: Maximum vocabulary size
    
    Returns:
        A: Sparse TF-IDF matrix (n_docs × max_features)
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    A = vectorizer.fit_transform(newsgroups.data)
    
    return A
