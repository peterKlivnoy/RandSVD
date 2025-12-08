"""
Test: SRFT Timing Breakdown

This test analyzes which parts of the SRFT (Subsampled Random Fourier Transform)
take the most time. We break down:

1. Sparse-to-dense conversion (if applicable)
2. Padding to power of 2
3. Random sign generation + application (D matrix)
4. FFT (F matrix) - the main transform
5. Real part extraction + normalization
6. Random index generation + subsampling (P matrix)
7. Final scaling

This helps identify bottlenecks and optimization opportunities.
"""

import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def srft_operator_timed(A, l, verbose=True):
    """
    SRFT operator with detailed timing for each step.
    
    Returns:
        Y: Sketched matrix
        timings: Dict with timing for each step
    """
    from scipy.sparse import issparse
    
    timings = {}
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # Step 0: Sparse conversion (if needed)
    t0 = time.perf_counter()
    if issparse(A):
        A_dense = A.toarray()
        timings['sparse_to_dense'] = time.perf_counter() - t0
    else:
        A_dense = A
        timings['sparse_to_dense'] = 0.0
    
    # Step 1: Padding to next power of 2
    t0 = time.perf_counter()
    if n > 0:
        n_padded = 1 << (n - 1).bit_length()
    else:
        n_padded = 1
    
    if n_padded != n:
        A_tilde = np.pad(A_dense, ((0, 0), (0, n_padded - n)), constant_values=0)
    else:
        A_tilde = A_dense
    timings['padding'] = time.perf_counter() - t0
    
    # Step 2: Random sign generation
    t0 = time.perf_counter()
    signs = rng.choice([-1, 1], size=n_padded)
    timings['sign_generation'] = time.perf_counter() - t0
    
    # Step 3: Sign application (D matrix)
    t0 = time.perf_counter()
    A_scrambled = A_tilde * signs
    timings['sign_application'] = time.perf_counter() - t0
    
    # Step 4: FFT (F matrix) - the main transform
    t0 = time.perf_counter()
    Y_mixed_complex = np.fft.fft(A_scrambled, axis=1)
    timings['fft'] = time.perf_counter() - t0
    
    # Step 5: Real part extraction + normalization
    t0 = time.perf_counter()
    Y_mixed = Y_mixed_complex.real / np.sqrt(n_padded)
    timings['real_and_normalize'] = time.perf_counter() - t0
    
    # Step 6: Random index generation for subsampling
    t0 = time.perf_counter()
    sampling_indices = rng.choice(n_padded, size=l, replace=False)
    timings['index_generation'] = time.perf_counter() - t0
    
    # Step 7: Subsampling (P matrix)
    t0 = time.perf_counter()
    Y_sampled = Y_mixed[:, sampling_indices]
    timings['subsampling'] = time.perf_counter() - t0
    
    # Step 8: Final scaling
    t0 = time.perf_counter()
    scaling_factor = np.sqrt(n_padded / l)
    Y = Y_sampled * scaling_factor
    timings['final_scaling'] = time.perf_counter() - t0
    
    # Total
    timings['total'] = sum(timings.values())
    
    if verbose:
        print(f"\n  SRFT Timing Breakdown (m={m}, n={n}, l={l}):")
        print(f"  " + "-"*50)
        for step, t in timings.items():
            if step != 'total':
                pct = 100 * t / timings['total'] if timings['total'] > 0 else 0
                print(f"    {step:20s}: {t*1000:8.2f} ms ({pct:5.1f}%)")
        print(f"  " + "-"*50)
        print(f"    {'TOTAL':20s}: {timings['total']*1000:8.2f} ms")
    
    return Y, timings


def srht_operator_timed(A, l, verbose=True):
    """
    SRHT operator with detailed timing for each step.
    
    Returns:
        Y: Sketched matrix
        timings: Dict with timing for each step
    """
    from scipy.sparse import issparse
    
    # Import FWHT function
    try:
        from src import hadamardKernel
        fastwht_func_2d = hadamardKernel.fwhtKernel2dOrdinary
        FAST_FWHT_AVAILABLE = True
    except ImportError:
        FAST_FWHT_AVAILABLE = False
    
    timings = {}
    m, n = A.shape
    rng = np.random.default_rng(seed=0)
    
    # Step 0: Sparse conversion
    t0 = time.perf_counter()
    if issparse(A):
        A_dense = A.toarray()
        timings['sparse_to_dense'] = time.perf_counter() - t0
    else:
        A_dense = A
        timings['sparse_to_dense'] = 0.0
    
    # Step 1: No padding in current SRHT (uses n directly)
    timings['padding'] = 0.0
    A_tilde = A_dense
    
    # Step 2: Random sign generation
    t0 = time.perf_counter()
    signs = rng.choice([-1, 1], size=n)
    timings['sign_generation'] = time.perf_counter() - t0
    
    # Step 3: Sign application (D matrix)
    t0 = time.perf_counter()
    A_scrambled = A_tilde * signs
    timings['sign_application'] = time.perf_counter() - t0
    
    # Step 4: FWHT (H matrix) - includes transpose overhead for C library
    t0 = time.perf_counter()
    if FAST_FWHT_AVAILABLE:
        # Transpose trick for C library
        A_scrambled_T = np.ascontiguousarray(A_scrambled.T, dtype=np.float64)
        fastwht_func_2d(A_scrambled_T)
        Y_mixed = A_scrambled_T.T / np.sqrt(n)
    else:
        # Fallback - would be slow
        raise RuntimeError("Fast FWHT not available")
    timings['fwht'] = time.perf_counter() - t0
    
    # Step 5: No real part extraction needed for Hadamard (already real)
    timings['real_and_normalize'] = 0.0
    
    # Step 6: Random index generation
    t0 = time.perf_counter()
    sampling_indices = rng.choice(n, size=l, replace=False)
    timings['index_generation'] = time.perf_counter() - t0
    
    # Step 7: Subsampling (P matrix)
    t0 = time.perf_counter()
    Y_sampled = Y_mixed[:, sampling_indices]
    timings['subsampling'] = time.perf_counter() - t0
    
    # Step 8: Final scaling
    t0 = time.perf_counter()
    scaling_factor = np.sqrt(n / l)
    Y = Y_sampled * scaling_factor
    timings['final_scaling'] = time.perf_counter() - t0
    
    # Total
    timings['total'] = sum(timings.values())
    
    if verbose:
        print(f"\n  SRHT Timing Breakdown (m={m}, n={n}, l={l}):")
        print(f"  " + "-"*50)
        for step, t in timings.items():
            if step != 'total':
                pct = 100 * t / timings['total'] if timings['total'] > 0 else 0
                print(f"    {step:20s}: {t*1000:8.2f} ms ({pct:5.1f}%)")
        print(f"  " + "-"*50)
        print(f"    {'TOTAL':20s}: {timings['total']*1000:8.2f} ms")
    
    return Y, timings


def run_timing_test(sizes, l_ratio=0.1, num_trials=3):
    """
    Run timing breakdown tests for various matrix sizes.
    
    Args:
        sizes: List of (m, n) tuples
        l_ratio: Sketch size as ratio of n
        num_trials: Number of trials to average
    """
    print("="*70)
    print("SRFT vs SRHT Timing Breakdown Analysis")
    print("="*70)
    
    for m, n in sizes:
        l = max(int(n * l_ratio), 50)
        
        print(f"\n{'='*70}")
        print(f"Matrix size: {m} × {n}, sketch size l={l}")
        print("="*70)
        
        # Create random matrix
        np.random.seed(42)
        A = np.random.randn(m, n)
        
        # Warm-up run
        _ = srft_operator_timed(A, l, verbose=False)
        _ = srht_operator_timed(A, l, verbose=False)
        
        # SRFT timing
        print("\n[SRFT - Subsampled Random Fourier Transform]")
        srft_timings_list = []
        for trial in range(num_trials):
            _, timings = srft_operator_timed(A, l, verbose=False)
            srft_timings_list.append(timings)
        
        # Average timings
        avg_srft = {}
        for key in srft_timings_list[0]:
            avg_srft[key] = np.median([t[key] for t in srft_timings_list])
        
        print(f"\n  SRFT Timing Breakdown (median of {num_trials} trials):")
        print(f"  " + "-"*50)
        for step, t in avg_srft.items():
            if step != 'total':
                pct = 100 * t / avg_srft['total'] if avg_srft['total'] > 0 else 0
                print(f"    {step:20s}: {t*1000:8.2f} ms ({pct:5.1f}%)")
        print(f"  " + "-"*50)
        print(f"    {'TOTAL':20s}: {avg_srft['total']*1000:8.2f} ms")
        
        # SRHT timing
        print("\n[SRHT - Subsampled Random Hadamard Transform]")
        srht_timings_list = []
        for trial in range(num_trials):
            _, timings = srht_operator_timed(A, l, verbose=False)
            srht_timings_list.append(timings)
        
        # Average timings
        avg_srht = {}
        for key in srht_timings_list[0]:
            avg_srht[key] = np.median([t[key] for t in srht_timings_list])
        
        print(f"\n  SRHT Timing Breakdown (median of {num_trials} trials):")
        print(f"  " + "-"*50)
        for step, t in avg_srht.items():
            if step != 'total':
                pct = 100 * t / avg_srht['total'] if avg_srht['total'] > 0 else 0
                print(f"    {step:20s}: {t*1000:8.2f} ms ({pct:5.1f}%)")
        print(f"  " + "-"*50)
        print(f"    {'TOTAL':20s}: {avg_srht['total']*1000:8.2f} ms")
        
        # Summary comparison
        print(f"\n  Summary:")
        print(f"    SRFT total: {avg_srft['total']*1000:.2f} ms")
        print(f"    SRHT total: {avg_srht['total']*1000:.2f} ms")
        speedup = avg_srht['total'] / avg_srft['total'] if avg_srft['total'] > 0 else 0
        if speedup > 1:
            print(f"    SRFT is {speedup:.2f}× faster than SRHT")
        else:
            print(f"    SRHT is {1/speedup:.2f}× faster than SRFT")


def main():
    # Test various sizes
    sizes = [
        (1000, 1000),
        (2000, 2000),
        (2048, 2048),  # Power of 2
        (4096, 4096),  # Larger power of 2
        (2000, 4000),  # Non-square
        (5000, 10000), # Large non-square
    ]
    
    run_timing_test(sizes, l_ratio=0.1, num_trials=5)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
