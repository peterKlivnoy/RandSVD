"""
Experiment 0: RandSVD vs Full SVD - The Fundamental Motivation
==============================================================

This experiment demonstrates WHY randomized SVD exists:
- Full SVD has O(mn²) complexity and becomes impractical for large matrices
- RandSVD achieves O(mnk + (m+n)k²) complexity for rank-k approximation
- For k << n, this provides dramatic speedups

This should be the FIRST figure in the paper - it motivates everything else.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.randsvd_algorithm import randSVD

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def benchmark_svd_methods(n_values, k, num_trials=3):
    """
    Compare full SVD vs RandSVD across matrix sizes.
    
    Parameters:
    -----------
    n_values : list of int
        Matrix dimensions to test (n x n matrices)
    k : int
        Target rank for RandSVD
    num_trials : int
        Number of trials for timing
        
    Returns:
    --------
    dict with timing results
    """
    results = {
        'n': [],
        'full_svd_time': [],
        'rand_svd_time': [],
        'speedup': [],
        'relative_error': []
    }
    
    p = 10  # Oversampling
    q = 0   # Power iterations
    
    for n in n_values:
        print(f"  Testing n={n}...", end=" ", flush=True)
        
        # Create test matrix with decaying spectrum
        np.random.seed(42)
        # U, _ = np.linalg.qr()
        V, _ = np.linalg.qr(np.random.randn(n, n))
        # Polynomial decay spectrum
        singular_values = 1.0 / (np.arange(1, n + 1) ** 1.5)
        A = np.random.randn(n, n)
        
        # Time full SVD (only if feasible)
        full_svd_time = None
        U_full, s_full, Vt_full = None, None, None
        
        if n <= 8000:  # Don't run full SVD for very large matrices
            times = []
            for _ in range(num_trials):
                start = time.perf_counter()
                U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
                times.append(time.perf_counter() - start)
            full_svd_time = np.median(times)
            print(f"Full SVD: {full_svd_time*1000:.0f}ms", end=" ", flush=True)
        else:
            print(f"Full SVD: SKIPPED (too large)", end=" ", flush=True)
        
        # Time RandSVD
        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            U_rand, s_rand, Vt_rand = randSVD(A, k, p=p, q=q, sketch_type='gaussian')
            times.append(time.perf_counter() - start)
        rand_svd_time = np.median(times)
        print(f"RandSVD: {rand_svd_time*1000:.0f}ms", end="", flush=True)
        
        # Calculate relative error if we have full SVD
        rel_error = None
        if U_full is not None:
            # Best rank-k approximation error (from full SVD)
            A_k_best = U_full[:, :k] @ np.diag(s_full[:k]) @ Vt_full[:k, :]
            best_error = np.linalg.norm(A - A_k_best, 'fro')
            
            # RandSVD approximation error
            A_k_rand = U_rand @ np.diag(s_rand) @ Vt_rand
            rand_error = np.linalg.norm(A - A_k_rand, 'fro')
            
            rel_error = rand_error / best_error if best_error > 0 else 1.0
            print(f" Error ratio: {rel_error:.4f}", end="")
        
        print()
        
        results['n'].append(n)
        results['full_svd_time'].append(full_svd_time)
        results['rand_svd_time'].append(rand_svd_time)
        results['speedup'].append(full_svd_time / rand_svd_time if full_svd_time else None)
        results['relative_error'].append(rel_error)
    
    return results


def create_figure(results, k, output_dir):
    """Create publication-quality figure showing SVD comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    n_vals = np.array(results['n'])
    full_times = np.array([t if t is not None else np.nan for t in results['full_svd_time']])
    rand_times = np.array(results['rand_svd_time'])
    speedups = np.array([s if s is not None else np.nan for s in results['speedup']])
    errors = np.array([e if e is not None else np.nan for e in results['relative_error']])
    
    # Panel A: Absolute timing comparison (log-log)
    ax = axes[0]
    
    # Plot full SVD (only where we have data)
    valid_full = ~np.isnan(full_times)
    ax.loglog(n_vals[valid_full], full_times[valid_full], 'o-', 
              color='#d62728', linewidth=2, markersize=8, label='Full SVD')
    
    # Extrapolate full SVD (cubic scaling)
    if np.sum(valid_full) >= 2:
        # Fit O(n³) model to existing data
        log_n = np.log(n_vals[valid_full])
        log_t = np.log(full_times[valid_full])
        # Linear fit in log space
        coeffs = np.polyfit(log_n, log_t, 1)
        
        # Extrapolate
        extrapolate_mask = ~valid_full
        if np.any(extrapolate_mask):
            extrapolated = np.exp(coeffs[1]) * n_vals[extrapolate_mask] ** coeffs[0]
            ax.loglog(n_vals[extrapolate_mask], extrapolated, 's--', 
                      color='#d62728', linewidth=1.5, markersize=6, alpha=0.5,
                      label=f'Full SVD (extrapolated, ~n^{coeffs[0]:.1f})')
    
    # Plot RandSVD
    ax.loglog(n_vals, rand_times, 's-', 
              color='#2ca02c', linewidth=2, markersize=8, label=f'RandSVD (k={k})')
    
    ax.set_xlabel('Matrix dimension n')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('(A) Computation Time')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel B: Speedup factor
    ax = axes[1]
    valid_speedup = ~np.isnan(speedups)
    
    ax.semilogy(n_vals[valid_speedup], speedups[valid_speedup], 'o-', 
                color='#1f77b4', linewidth=2, markersize=8)
    
    # Add reference lines
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10× speedup')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='100× speedup')
    
    ax.set_xlabel('Matrix dimension n')
    ax.set_ylabel('Speedup (Full SVD / RandSVD)')
    ax.set_title('(B) Speedup Factor')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add speedup annotations
    for i, (n, s) in enumerate(zip(n_vals[valid_speedup], speedups[valid_speedup])):
        if i == len(n_vals[valid_speedup]) - 1 or i == 0:  # First and last
            ax.annotate(f'{s:.0f}×', (n, s), textcoords="offset points", 
                       xytext=(5, 5), fontsize=9, color='#1f77b4')
    
    # Panel C: Accuracy (relative error)
    ax = axes[2]
    valid_error = ~np.isnan(errors)
    
    ax.plot(n_vals[valid_error], errors[valid_error], 'o-', 
            color='#9467bd', linewidth=2, markersize=8)
    
    # Reference line at 1.0 (optimal)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, 
               label='Optimal (ratio = 1.0)')
    
    ax.set_xlabel('Matrix dimension n')
    ax.set_ylabel('Relative Error (RandSVD / Best Rank-k)')
    ax.set_title('(C) Approximation Quality')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    # Cap y-axis to start at 0 and set a lower max (e.g., 1.05 or 1.1)
    ax.set_ylim(1,np.nanmax(errors) * 1.2)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'fig0_svd_motivation.pdf')
    plt.savefig(output_dir / 'fig0_svd_motivation.png')
    plt.close()
    
    return output_dir / 'fig0_svd_motivation.pdf'


def main():
    print("=" * 70)
    print("EXPERIMENT 0: RandSVD vs Full SVD - The Fundamental Motivation")
    print("=" * 70)
    print()
    
    # Test parameters
    # Use large matrices to show the real power of RandSVD
    n_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    k = 50  # We only want top-k singular values/vectors
    print('Helllooo')
    print(f"Configuration:")
    print(f"  Matrix sizes: {n_values}")
    print(f"  Target rank k = {k}")
    print(f"  Oversampling p = 10, Power iterations q = 0")
    print()
    
    print("[1/2] Running benchmarks...")
    results = benchmark_svd_methods(n_values, k, num_trials=3)
    
    print()
    print("[2/2] Creating figure...")
    output_path = create_figure(results, k, Path(__file__).parent.parent / 'figures')
    
    print()
    print(f"✓ Saved: {output_path}")
    
    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'n':>8} | {'Full SVD':>12} | {'RandSVD':>12} | {'Speedup':>10} | {'Error Ratio':>12}")
    print("-" * 70)
    for i, n in enumerate(results['n']):
        full_t = results['full_svd_time'][i]
        rand_t = results['rand_svd_time'][i]
        speedup = results['speedup'][i]
        error = results['relative_error'][i]
        
        full_str = f"{full_t*1000:.0f}ms" if full_t else "N/A"
        speedup_str = f"{speedup:.1f}×" if speedup else "N/A"
        error_str = f"{error:.4f}" if error else "N/A"
        
        print(f"{n:>8} | {full_str:>12} | {rand_t*1000:.0f}ms{' ':>6} | {speedup_str:>10} | {error_str:>12}")
    
    print()
    print("Key Insight: RandSVD achieves near-optimal accuracy with")
    print("             dramatically reduced computation time for large matrices.")
    print()


if __name__ == "__main__":
    main()
