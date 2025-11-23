"""
Test SRFT (Subsampled Random Fourier Transform) vs SRHT
Compare speed and accuracy using NumPy's built-in FFT
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

def srft_operator(A, l):
    """
    Applies the Subsampled Randomized Fourier Transform (SRFT) to matrix A.
    Uses NumPy's built-in FFT for O(N^2 * log N) complexity.
    """
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # 1. Padding to next power of 2
    n_padded = 1 << (n - 1).bit_length() if n > 0 else 1
    A_tilde = A if n_padded == n else np.pad(A, ((0, 0), (0, n_padded - n)))
    
    # 2. D: Random sign matrix (scrambling)
    signs = rng.choice([-1, 1], size=n_padded)
    A_scrambled = A_tilde * signs
    
    # 3. F: FFT (mixing) - apply to each row
    # NumPy's FFT is along last axis by default
    Y_mixed = np.fft.fft(A_scrambled, axis=1) / np.sqrt(n_padded)
    
    # Take real part only (for real input, we only need real output)
    # In theory SRFT uses complex, but for comparison we take real part
    Y_mixed = Y_mixed.real
    
    # 4. P: Subsampling
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    Y_sampled = Y_mixed[:, sampling_indices]
    
    # 5. Final scaling
    scaling_factor = np.sqrt(n_padded / l)
    Y = Y_sampled * scaling_factor
    
    return Y

# Import SRHT for comparison
from src.structured_sketch import srht_operator

print("="*70)
print("SRFT vs SRHT Speed and Accuracy Comparison")
print("="*70)

# Test parameters
N = 2048
l = 100
k = 50
num_trials = 5

print(f"\nMatrix: {N}×{N}, Sketch size: l={l}, Target rank: k={k}")

# Create test matrix
A = np.random.randn(N, N)
U_true, s_true, Vt_true = np.linalg.svd(A, full_matrices=False)

print("\n" + "="*70)
print("SPEED TEST")
print("="*70)

# Test SRHT speed
start = time.perf_counter()
for _ in range(num_trials):
    Y_srht = srht_operator(A, l)
end = time.perf_counter()
time_srht = (end - start) / num_trials
print(f"SRHT:  {time_srht:.4f}s")

# Test SRFT speed
start = time.perf_counter()
for _ in range(num_trials):
    Y_srft = srft_operator(A, l)
end = time.perf_counter()
time_srft = (end - start) / num_trials
print(f"SRFT:  {time_srft:.4f}s")

print(f"\nSpeedup: {time_srht/time_srft:.2f}x")
if time_srft < time_srht:
    print("✓ SRFT is FASTER!")
else:
    print("✗ SRHT is faster")

print("\n" + "="*70)
print("ACCURACY TEST")
print("="*70)

# Compute approximations using both methods
from src.randsvd_algorithm import randSVD

# SRHT approximation
Y_srht = srht_operator(A, l)
Q_srht, _ = np.linalg.qr(Y_srht)
B_srht = Q_srht.T @ A
U_tilde_srht, s_srht, Vt_srht = np.linalg.svd(B_srht, full_matrices=False)
U_srht = Q_srht @ U_tilde_srht
s_srht_k = s_srht[:k]

# SRFT approximation
Y_srft = srft_operator(A, l)
Q_srft, _ = np.linalg.qr(Y_srft)
B_srft = Q_srft.T @ A
U_tilde_srft, s_srft, Vt_srft = np.linalg.svd(B_srft, full_matrices=False)
U_srft = Q_srft @ U_tilde_srft
s_srft_k = s_srft[:k]

# True singular values
s_true_k = s_true[:k]

# Compute errors
error_srht = np.max(np.abs(s_true_k - s_srht_k))
error_srft = np.max(np.abs(s_true_k - s_srft_k))

print(f"SRHT max error: {error_srht:.4e}")
print(f"SRFT max error: {error_srft:.4e}")
print(f"Error ratio (SRFT/SRHT): {error_srft/error_srht:.3f}")

if error_srft < error_srht * 1.1:  # Within 10%
    print("✓ SRFT accuracy comparable to SRHT")
else:
    print("✗ SRFT accuracy worse than SRHT")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"SRFT uses NumPy's built-in FFT (no external library needed)")
print(f"Speed:    SRFT is {time_srht/time_srft:.2f}x {'faster' if time_srft < time_srht else 'slower'}")
print(f"Accuracy: SRFT error is {error_srft/error_srht:.2f}x that of SRHT")
print(f"\nConclusion: {'SRFT is a good alternative!' if time_srft < time_srht and error_srft < error_srht * 1.5 else 'SRHT is still better overall'}")
