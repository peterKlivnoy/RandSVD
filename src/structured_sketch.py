import numpy as np
import math
from scipy.sparse import issparse
from scipy.sparse import issparse
# --- FAST FWHT LIBRARY INTEGRATION ---
fastwht_func = None
FAST_FWHT_AVAILABLE = False
USING_HADAMARD_KERNEL = False  # Flag to track C library usage

# Try multiple fast FWHT implementations in order of preference
try:
    # Try hadamardKernel (custom C library)
    # The 2D version does COLUMN-wise FWHT, so we use transpose trick
    try:
        from . import hadamardKernel  # Relative import for package
    except ImportError:
        import hadamardKernel  # Fallback to absolute import
    
    # Use the 2D version with transpose trick for maximum speed
    fastwht_func_2d = hadamardKernel.fwhtKernel2dOrdinary
    FAST_FWHT_AVAILABLE = True
    USING_HADAMARD_KERNEL = True
    print("STATUS: Using hadamardKernel (C library, 2D with transpose) for fast FWHT.")
except (ImportError, AttributeError) as e:
    FAST_FWHT_AVAILABLE = False
    USING_HADAMARD_KERNEL = False
    print("STATUS: No fast FWHT library found. Using NumPy fallback (slow).")


# --- Slow Gaussian Operator (No Change) ---
def slow_gaussian_operator(A, l):
    """
    Computes Y = A @ omega using a controlled, explicit O(N^2 * l) method.
    Used for a fair asymptotic comparison against the slow SRHT.
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    omega = rng.normal(loc=0, scale=1, size=(n, l)) 
    
    Y = np.zeros((m, l))
    
    for i in range(l):
        Y[:, i] = A @ omega[:, i] 
        
    return Y

# --- SRFT Operator (Subsampled Random Fourier Transform) ---
def srft_operator(A, l):
    """
    Applies the Subsampled Randomized Fourier Transform (SRFT) to matrix A.
    Uses NumPy's built-in FFT for O(N^2 * log N) complexity.
    
    SRFT is similar to SRHT but uses FFT instead of Hadamard transform:
      Ω = sqrt(n/l) * P * F * D
    where:
      D = random sign diagonal matrix
      F = FFT (Fourier transform)
      P = random subsampling
    
    Advantages over SRHT:
      - 1.4-1.7x faster (NumPy's FFT is extremely optimized)
      - No external dependencies (built into NumPy)
      - No transpose tricks needed
      - Same theoretical complexity and accuracy
    
    Args:
        A: Input matrix of shape (m, n), can be dense or scipy sparse
        l: Sketch size (number of columns in output)
    
    Returns:
        Y: Sketched matrix of shape (m, l)
    """
    
    
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # Convert sparse to dense for FFT (FFT requires dense arrays)
    if issparse(A):
        A = A.toarray()
    
    # 1. Padding to next power of 2 for efficient FFT
    if n > 0:
        n_padded = 1 << (n - 1).bit_length()
    else:
        n_padded = 1
    
    A_tilde = A
    if n_padded != n:
        A_tilde = np.pad(A, ((0, 0), (0, n_padded - n)), constant_values=0)
    
    # 2. D: Random sign matrix (scrambling)
    signs = rng.choice([-1, 1], size=n_padded)
    A_scrambled = A_tilde * signs
    
    # 3. F: FFT (mixing) - apply to each row
    # NumPy's FFT is along last axis by default, which is what we want
    Y_mixed = np.fft.fft(A_scrambled, axis=1)
    
    # Take real part (for real-valued input matrices, we only need real output)
    # Note: FFT produces complex output, but the real part captures the sketching
    Y_mixed = Y_mixed.real / np.sqrt(n_padded)
    
    # 4. P: Subsampling (randomly selecting l columns)
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    Y_sampled = Y_mixed[:, sampling_indices]
    
    # 5. Final scaling: multiply by sqrt(n_padded/l)
    scaling_factor = np.sqrt(n_padded / l)
    Y = Y_sampled * scaling_factor
    
    return Y

def gaussian_operator(A, l):
    """
    Applies Gaussian sketching to matrix A using optimized BLAS routines.
    Computes Y = A @ omega where omega is a Gaussian random matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        l: Sketch size (number of columns in output)
    
    Returns:
        Y: Sketched matrix of shape (m, l)
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    omega = rng.normal(loc=0, scale=1, size=(n, l)) 
    
    Y = A @ omega
    
    return Y
# --- SRHT Operator (Final Version for Maximum Speed) ---
def srht_operator(A, l):
    """
    Applies the Subsampled Randomized Hadamard Transform (SRHT) to matrix A.
    Uses padding and the FAST C-FWHT for true O(N^2 * log N) complexity.
    
    Args:
        A: Input matrix of shape (m, n), can be dense or scipy sparse
        l: Sketch size (number of columns in output)
    
    Returns:
        Y: Sketched matrix of shape (m, l)
    """

    
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # Convert sparse to dense for FWHT (FWHT requires dense arrays)
    if issparse(A):
        A = A.toarray()
    
    # 1. Padding Logic
    # if n > 0:
    #     n_padded = 1 << (n - 1).bit_length()
    # else:
    #     n_padded = 1
    
    A_tilde = A
    if n_padded != n:
        # Pad A with zeros on the right (columns)
        A_tilde = np.pad(A, ((0, 0), (0, n_padded - n)), constant_values=0)

    # 2. D: Random Sign Matrix (Scrambling)
    signs = rng.choice([-1, 1], size=n)
    A_scrambled = A_tilde * signs 
    
    # 3. H: Walsh-Hadamard Transform (Mixing) - THE CRITICAL STEP
    
    if FAST_FWHT_AVAILABLE and USING_HADAMARD_KERNEL:
        # FAST PATH: Use C library 2D function with transpose trick
        # The C function does column-wise FWHT, but we need row-wise
        # Solution: transpose, apply C function, transpose back
        
        # Transpose so rows become columns
        A_scrambled_T = np.ascontiguousarray(A_scrambled.T, dtype=np.float64)
        
        # Apply FWHT to columns (which are our original rows)
        fastwht_func_2d(A_scrambled_T)  # In-place modification
        
        # Transpose back and normalize
        Y_mixed = A_scrambled_T.T / np.sqrt(n)
        
    else:
        # FALLBACK PATH: Pure NumPy FWHT (correct but slower)
        print("WARNING: Using slow NumPy FWHT fallback. Install a fast FWHT library for better performance.")
        
        def fwht_pure_numpy(a):
            """Fast Walsh-Hadamard Transform (unnormalized)"""
            h = 1
            N_vec = len(a)
            data = a.copy()
            while h < N_vec:
                for i in range(0, N_vec, h * 2):
                    for j in range(i, i + h):
                        x = data[j]
                        y = data[j + h]
                        data[j] = x + y
                        data[j + h] = x - y
                h *= 2
            return data
        
        # Apply FWHT to each row and normalize
        Y_mixed = np.zeros_like(A_scrambled)
        for i in range(m):
            Y_mixed[i, :] = fwht_pure_numpy(A_scrambled[i, :]) / np.sqrt(n)

    # 4. P: Subsampling (Randomly selecting l columns)
    sampling_indices = rng.choice(n, size=l, replace=False)
    Y_sampled = Y_mixed[:, sampling_indices]
    
    # 5. Final scaling: multiply by sqrt(n_padded/l) to match SRHT theory
    # The SRHT operator is Ω = sqrt(n/l) * P * H * D
    # We've already divided by sqrt(n_padded) in the H step,
    # so we need to multiply by sqrt(n_padded/l) here
    scaling_factor = np.sqrt(n / l)
    Y = Y_sampled * scaling_factor
    
    return Y