"""
Naive (pure Python loop) implementations of all sketching methods.
These are intentionally slow to demonstrate algorithmic complexity differences.

Use these for educational purposes and complexity comparisons.
DO NOT use in production - use the optimized versions in structured_sketch.py.
"""
import numpy as np


def naive_gaussian_sketch(A, l, seed=0):
    """
    Naive O(mnl) Gaussian sketching using explicit triple loop.
    Y[i,j] = sum_k A[i,k] * Omega[k,j]
    
    Args:
        A: (m, n) input matrix
        l: sketch size
    
    Returns:
        Y: (m, l) sketched matrix
    
    Complexity: O(m * n * l)
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=seed)
    
    # Generate random Gaussian matrix
    Omega = rng.normal(0, 1, size=(n, l))
    
    # Explicit triple loop to compute Y = A @ Omega
    Y = np.zeros((m, l))
    for i in range(m):          # O(m)
        for j in range(l):      # O(l)
            for k in range(n):  # O(n)
                Y[i, j] += A[i, k] * Omega[k, j]
    
    return Y


def naive_fft_1d(x):
    """
    Pure Python FFT implementation - O(N log N) Cooley-Tukey algorithm.
    
    This implements the Fast Fourier Transform using the recursive
    Cooley-Tukey divide-and-conquer algorithm, achieving O(N log N)
    complexity WITHOUT using NumPy's optimized C implementation.
    
    Algorithm:
    - Split into even/odd indices: x[0::2], x[1::2]
    - Recursively compute FFT on each half
    - Combine using twiddle factors: W_N^k = exp(-2πi k/N)
    
    Complexity: O(N log N) algorithmic, but pure Python (no C optimizations)
    
    This demonstrates the algorithmic speedup of FFT over DFT while still
    being much slower than NumPy's highly-optimized C/Fortran FFT.
    
    Args:
        x: 1D array of length N
    
    Returns:
        X: DFT of x
    
    Complexity: O(N log N)
    """
    N = len(x)
    
    # Base case: single element
    if N <= 1:
        return x
    
    # Ensure N is power of 2 by padding
    if N & (N - 1) != 0:  # Check if not a power of 2
        next_pow2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = next_pow2
    
    # Divide: split into even and odd indices
    even = naive_fft_1d(x[0::2])
    odd = naive_fft_1d(x[1::2])
    
    # Conquer: combine with twiddle factors
    # Compute twiddle factors: W_N^k = exp(-2πi k/N)
    T = np.array([np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)])
    
    # Combine:
    # X[k] = E[k] + W_N^k * O[k]           for k < N/2
    # X[k+N/2] = E[k] - W_N^k * O[k]       for k < N/2
    return np.concatenate([even + T, even - T])


def naive_fwht_1d(x):
    """
    Pure Python Fast Walsh-Hadamard Transform - O(N log N) recursive algorithm.
    
    This implements the Fast WHT using recursive butterfly operations,
    achieving O(N log N) complexity WITHOUT forming the full Hadamard matrix
    and WITHOUT using optimized C implementations.
    
    Algorithm (in-place butterfly):
    - Split into two halves: left, right
    - Recursively compute FWHT on each half
    - Butterfly combine: [left + right, left - right]
    
    Complexity: O(N log N) algorithmic, but pure Python (no C optimizations)
    
    This demonstrates the algorithmic speedup of Fast WHT over naive matrix
    multiplication while still being slower than optimized C implementations.
    
    Args:
        x: 1D array of length N (padded to power of 2 if needed)
    
    Returns:
        y: WHT of x
    
    Complexity: O(N log N)
    """
    N = len(x)
    
    # Ensure N is power of 2 by padding
    if N & (N - 1) != 0:  # Check if not a power of 2
        next_pow2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = next_pow2
    
    # Copy to avoid modifying input
    x = x.copy()
    
    # Recursive FWHT implementation
    def fwht_recursive(arr):
        n = len(arr)
        if n <= 1:
            return arr
        
        # Split into two halves
        half = n // 2
        left = arr[:half]
        right = arr[half:]
        
        # Recursively compute FWHT on each half
        h_left = fwht_recursive(left)
        h_right = fwht_recursive(right)
        
        # Butterfly combination
        # First half: H_left + H_right
        # Second half: H_left - H_right
        result = np.zeros(n, dtype=arr.dtype)
        result[:half] = h_left + h_right
        result[half:] = h_left - h_right
        
        return result
    
    result = fwht_recursive(x)
    
    # No normalization needed for our purposes
    return result


def naive_srft_sketch(A, l, seed=0):
    """
    Pure Python SRFT sketching using Python FFT - O(mn log n) complexity.
    This shows the algorithmic speedup WITHOUT NumPy's C optimizations.
    
    SRFT structure: Y = A @ (sqrt(n/l) * P * F * D)
    where:
        D = random diagonal sign matrix (n×n)
        F = DFT matrix (n×n) [FAST: O(n log n) per row using Cooley-Tukey]
        P = random sampling (n×l)
    
    Args:
        A: (m, n) input matrix
        l: sketch size
    
    Returns:
        Y: (m, l) sketched matrix
    
    Complexity: O(m * n * log n) algorithmic, but pure Python implementation
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=seed)
    
    # Pad to power of 2
    n_padded = 1 << (n - 1).bit_length() if n > 0 else 1
    A_padded = A if n_padded == n else np.pad(A, ((0, 0), (0, n_padded - n)))
    
    # 1. D: Random signs
    signs = rng.choice([-1, 1], size=n_padded)
    A_scrambled = A_padded * signs
    
    # 2. F: Apply Python FFT to each row [O(m * n * log n)]
    Y_mixed = np.zeros((m, n_padded), dtype=complex)
    for i in range(m):                          # O(m)
        Y_mixed[i, :] = naive_fft_1d(A_scrambled[i, :])  # O(n log n) per row!
    
    Y_mixed = Y_mixed.real / np.sqrt(n_padded)
    
    # 3. P: Random sampling
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    Y_sampled = Y_mixed[:, sampling_indices]
    
    # 4. Scaling
    Y = Y_sampled * np.sqrt(n_padded / l)
    
    return Y


def naive_srht_sketch(A, l, seed=0):
    """
    Pure Python SRHT sketching using Python FWHT - O(mn log n) complexity.
    This shows the algorithmic speedup WITHOUT optimized C implementations.
    
    SRHT structure: Y = A @ (sqrt(n/l) * P * H * D)
    where:
        D = random diagonal sign matrix (n×n)
        H = Hadamard matrix (n×n) [FAST: O(n log n) per row using recursive butterflies]
        P = random sampling (n×l)
    
    Args:
        A: (m, n) input matrix
        l: sketch size
    
    Returns:
        Y: (m, l) sketched matrix
    
    Complexity: O(m * n * log n) algorithmic, but pure Python implementation
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=seed)
    
    # Pad to power of 2
    n_padded = 1 << (n - 1).bit_length() if n > 0 else 1
    A_padded = A if n_padded == n else np.pad(A, ((0, 0), (0, n_padded - n)))
    
    # 1. D: Random signs
    signs = rng.choice([-1, 1], size=n_padded)
    A_scrambled = A_padded * signs
    
    # 2. H: Apply Python FWHT to each row [O(m * n * log n)]
    Y_mixed = np.zeros((m, n_padded))
    for i in range(m):                              # O(m)
        Y_mixed[i, :] = naive_fwht_1d(A_scrambled[i, :])  # O(n log n) per row!
    
    Y_mixed = Y_mixed / np.sqrt(n_padded)
    
    # 3. P: Random sampling
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    Y_sampled = Y_mixed[:, sampling_indices]
    
    # 4. Scaling
    Y = Y_sampled * np.sqrt(n_padded / l)
    
    return Y


# Summary of complexities:
# 
# naive_gaussian_sketch:  O(m * n * l)      - Triple loop in pure Python
# naive_srft_sketch:      O(m * n * log n)  - Cooley-Tukey FFT in pure Python
# naive_srht_sketch:      O(m * n * log n)  - Recursive FWHT in pure Python
# 
# Optimized versions (from structured_sketch.py):
# gaussian (BLAS):        O(m * n * l)      - Same complexity, but BLAS is ~1000-5000x faster
# srft_operator (FFT):    O(m * n * log n)  - NumPy's optimized C/Fortran FFT (~50-100x faster)
# srht_operator (FWHT):   O(m * n * log n)  - C library FWHT (~10-50x faster)
# 
# Key insights:
# - Pure Python FFT/FWHT achieve optimal O(n log n) algorithmic complexity
# - NumPy/BLAS/C libraries add massive constant factor improvements
# - BLAS optimization is so good that Gaussian @ O(mnl) beats Python FFT @ O(mn log n)
# - For production: use optimized libraries (NumPy FFT is incredibly fast!)
# - For education: pure Python shows the algorithmic improvement clearly
