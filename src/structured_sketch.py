import numpy as np
import math

# --- FAST FWHT LIBRARY INTEGRATION ---
fastwht_func = None
FAST_FWHT_AVAILABLE = False
USING_HADAMARD_KERNEL = False  # Flag to track C library usage

# Try multiple fast FWHT implementations in order of preference
try:
    # Option 1: Try hadamardKernel (custom C library)
    # Use relative import since it's in the same directory
    try:
        from . import hadamardKernel  # Relative import for package
    except ImportError:
        import hadamardKernel  # Fallback to absolute import
    
    fastwht_func = hadamardKernel.fwhtKernel2dOrdinary
    FAST_FWHT_AVAILABLE = True
    USING_HADAMARD_KERNEL = True
    print("STATUS: Using hadamardKernel (C library) for fast FWHT.")
except (ImportError, AttributeError) as e:
    try:
        # Option 2: Try scipy's fht (Fast Hadamard Transform)
        from scipy.fft import fht
        # Wrap scipy's fht to work with 2D arrays row-wise
        def scipy_fht_2d(A):
            """Apply FHT to each row of A"""
            result = np.zeros_like(A)
            for i in range(A.shape[0]):
                result[i, :] = fht(A[i, :])
            return result
        fastwht_func = scipy_fht_2d
        FAST_FWHT_AVAILABLE = True
        print("STATUS: Using SciPy's fht for fast FWHT.")
    except ImportError:
        FAST_FWHT_AVAILABLE = False
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

# --- SRHT Operator (Final Version for Maximum Speed) ---
def srht_operator(A, l):
    """
    Applies the Subsampled Randomized Hadamard Transform (SRHT) to matrix A.
    Uses padding and the FAST C-FWHT for true O(N^2 * log N) complexity.
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # 1. Padding Logic
    if n > 0:
        n_padded = 1 << (n - 1).bit_length()
    else:
        n_padded = 1
    
    A_tilde = A
    if n_padded != n:
        # Pad A with zeros on the right (columns)
        A_tilde = np.pad(A, ((0, 0), (0, n_padded - n)), constant_values=0)

    # 2. D: Random Sign Matrix (Scrambling)
    signs = rng.choice([-1, 1], size=n_padded)
    A_scrambled = A_tilde * signs 
    
    # 3. H: Walsh-Hadamard Transform (Mixing) - THE CRITICAL STEP
    
    if FAST_FWHT_AVAILABLE:
        # FAST PATH: Use C-backed module or scipy
        if USING_HADAMARD_KERNEL:
            # Using the custom C library (in-place modification)
            # The function expects data in row-major order (standard NumPy layout)
            # and modifies the array in-place
            A_scrambled_copy = np.ascontiguousarray(A_scrambled, dtype=np.float64)
            
            # Apply FWHT in-place (returns None)
            fastwht_func(A_scrambled_copy)
            
            # Normalize
            Y_mixed = A_scrambled_copy / np.sqrt(n_padded)
        else:
            # Using scipy's fht (returns new array)
            Y_mixed = fastwht_func(A_scrambled)

    else:
        # FALLBACK PATH (Slow, but correct complexity scaling)
        def fwht_pure_numpy(a):
            h = 1; N_vec = len(a); data = a.copy() 
            while h < N_vec:
                for i in range(0, N_vec, h * 2):
                    for j in range(i, i + h):
                        x = data[j]; y = data[j + h]
                        data[j] = x + y; data[j + h] = x - y
                h *= 2
            return data
            
        fwht_normalized = lambda row: fwht_pure_numpy(row) / np.sqrt(n_padded)
        Y_mixed = np.apply_along_axis(fwht_normalized, axis=1, arr=A_scrambled)

    # 4. P: Subsampling (Randomly selecting l columns)
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    Y = Y_mixed[:, sampling_indices]
    
    return Y