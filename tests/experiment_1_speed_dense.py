"""
Experiment 1: Speed Comparison on Dense Matrices

This experiment compares sketching methods on dense matrices:
  - Gaussian (optimized with BLAS)
  - SRFT (FFT-based)
  - SRHT (Hadamard-based)
  - Full SVD (for reference)

We show both optimized and naive (pure-Python) implementations to
disentangle algorithmic complexity from library optimization.

Figures produced:
  - Fig 1: Runtime comparison (optimized vs naive implementations)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.structured_sketch import srft_operator, srht_operator
from src.naive_sketching import (
    naive_gaussian_sketch,
    naive_srft_sketch,
    naive_srht_sketch
)
from src.randsvd_algorithm import randSVD

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'gaussian': '#0072B2',  # Blue
    'srft': '#D55E00',      # Orange
    'srht': '#009E73',      # Green
}
MARKERS = {'gaussian': 'o', 'srft': 's', 'srht': '^'}


def benchmark_optimized(n, sketch_sizes, num_trials=5):
    """Benchmark optimized implementations for a single matrix size."""
    results = {'gaussian': [], 'srft': [], 'srht': [], 'full_svd': None, 'randsvd': []}
    
    print(f"\n  Matrix size: {n}×{n}")
    
    # Create random matrix
    np.random.seed(42)
    A = np.random.randn(n, n)
    
    # Benchmark full SVD (once, doesn't depend on l)
    times = []
    for trial in range(num_trials):
        t0 = time.perf_counter()
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        times.append(time.perf_counter() - t0)
    results['full_svd'] = np.median(times)
    print(f"    Full SVD: {results['full_svd']*1000:.1f}ms")
    
    for l in sketch_sizes:
        if l > n:
            for method in ['gaussian', 'srft', 'srht', 'randsvd']:
                results[method].append(np.nan)
            continue
            
        print(f"    l={l}: ", end="", flush=True)
        
        # Gaussian sketch
        times = []
        for trial in range(num_trials):
            np.random.seed(trial)
            Omega = np.random.randn(n, l)
            t0 = time.perf_counter()
            Y = A @ Omega
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"G={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # SRFT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srft_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # SRHT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srht_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # RandSVD (Gaussian)
        k = l - 10  # k < l because l = k + p
        if k > 0:
            times = []
            for trial in range(num_trials):
                t0 = time.perf_counter()
                U, s, Vt = randSVD(A, k, p=10, q=1, sketch_type='gaussian')
                times.append(time.perf_counter() - t0)
            results['randsvd'].append(np.median(times))
            print(f"RandSVD={np.median(times)*1000:.1f}ms")
        else:
            results['randsvd'].append(np.nan)
            print()
    
    return results


def benchmark_naive(n, sketch_sizes, num_trials=3):
    """Benchmark naive pure-Python implementations."""
    results = {'gaussian': [], 'srft': [], 'srht': []}
    
    print(f"\n  Matrix size: {n}×{n}")
    
    np.random.seed(42)
    A = np.random.randn(n, n)
    
    for l in sketch_sizes:
        if l > n:
            for method in results:
                results[method].append(np.nan)
            continue
            
        print(f"    l={l}: ", end="", flush=True)
        
        # Naive Gaussian
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_gaussian_sketch(A, l, seed=trial)
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"G={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # Naive SRFT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_srft_sketch(A, l, seed=trial)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # Naive SRHT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_srht_sketch(A, l, seed=trial)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.1f}ms")
    
    return results


def create_figure_1(results_opt, results_naive, sketch_sizes_opt, sketch_sizes_naive, n_opt, n_naive, output_dir):
    """Create publication-quality Figure 1: Speed comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # ===== Panel A: Optimized implementations =====
    ax = axes[0]
    
    for method in ['gaussian', 'srft', 'srht']:
        times_ms = np.array(results_opt[method]) * 1000
        valid = ~np.isnan(times_ms)
        ax.plot(np.array(sketch_sizes_opt)[valid], times_ms[valid], 
                marker=MARKERS[method], color=COLORS[method],
                linewidth=2, markersize=8, label=f'{method.upper()} sketch')
    
    # Add RandSVD
    times_randsvd = np.array(results_opt['randsvd']) * 1000
    valid_rs = ~np.isnan(times_randsvd)
    ax.plot(np.array(sketch_sizes_opt)[valid_rs], times_randsvd[valid_rs], 
            marker='D', color='#E69F00', linewidth=2, markersize=8, 
            linestyle='--', label='RandSVD (full)')
    
    # Add Full SVD as horizontal reference line
    full_svd_ms = results_opt['full_svd'] * 1000
    ax.axhline(y=full_svd_ms, color='#CC79A7', linestyle=':', linewidth=2.5, 
               label=f'Full SVD ({full_svd_ms:.0f}ms)')
    
    ax.set_xlabel('Sketch size ($\\ell$)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(a) Optimized Implementations\n$n = {n_opt}$', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('log')
    
    # ===== Panel B: Naive implementations =====
    ax = axes[1]
    
    times_g = np.array(results_naive['gaussian']) * 1000
    times_srft = np.array(results_naive['srft']) * 1000
    times_srht = np.array(results_naive['srht']) * 1000
    
    valid_g = ~np.isnan(times_g)
    valid_srft = ~np.isnan(times_srft)
    valid_srht = ~np.isnan(times_srht)
    
    ax.plot(np.array(sketch_sizes_naive)[valid_g], times_g[valid_g], 
            marker='o', color=COLORS['gaussian'],
            linewidth=2, markersize=8, label='Gaussian')
    ax.plot(np.array(sketch_sizes_naive)[valid_srft], times_srft[valid_srft], 
            marker='s', color=COLORS['srft'],
            linewidth=2, markersize=8, label='SRFT (naive)')
    ax.plot(np.array(sketch_sizes_naive)[valid_srht], times_srht[valid_srht], 
            marker='^', color=COLORS['srht'],
            linewidth=2, markersize=8, label='SRHT (naive)')
    
    ax.set_xlabel('Sketch size ($\\ell$)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(b) Naive Pure-Python\n$n = {n_naive}$', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    
    # ===== Panel C: Speedup factors =====
    ax = axes[2]
    
    # Optimized speedup: Gaussian / Structured
    speedup_srft_opt = np.array(results_opt['gaussian']) / np.array(results_opt['srft'])
    speedup_srht_opt = np.array(results_opt['gaussian']) / np.array(results_opt['srht'])
    
    valid_srft = ~np.isnan(speedup_srft_opt)
    valid_srht = ~np.isnan(speedup_srht_opt)
    
    ax.plot(np.array(sketch_sizes_opt)[valid_srft], speedup_srft_opt[valid_srft], 
            marker='s', color=COLORS['srft'],
            linewidth=2, markersize=8, linestyle='-', label='SRFT (optimized)')
    ax.plot(np.array(sketch_sizes_opt)[valid_srht], speedup_srht_opt[valid_srht], 
            marker='^', color=COLORS['srht'],
            linewidth=2, markersize=8, linestyle='-', label='SRHT (optimized)')
    
    # Naive speedup
    speedup_srft_naive = np.array(results_naive['gaussian']) / np.array(results_naive['srft'])
    speedup_srht_naive = np.array(results_naive['gaussian']) / np.array(results_naive['srht'])
    
    valid_srft_n = ~np.isnan(speedup_srft_naive)
    valid_srht_n = ~np.isnan(speedup_srht_naive)
    
    ax.plot(np.array(sketch_sizes_naive)[valid_srft_n], speedup_srft_naive[valid_srft_n], 
            marker='s', color=COLORS['srft'],
            linewidth=2, markersize=8, linestyle='--', label='SRFT (naive)')
    ax.plot(np.array(sketch_sizes_naive)[valid_srht_n], speedup_srht_naive[valid_srht_n], 
            marker='^', color=COLORS['srht'],
            linewidth=2, markersize=8, linestyle='--', label='SRHT (naive)')
    
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, label='Breakeven')
    ax.set_xlabel('Sketch size ($\\ell$)')
    ax.set_ylabel('Speedup (Gaussian / Structured)')
    ax.set_title('(c) Speedup Factors', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig1_speed_dense.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig1_speed_dense.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 1: Speed Comparison on Dense Matrices")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    n_opt = 2048
    sketch_sizes_opt = [20, 50, 100, 200, 400]
    
    n_naive = 512
    sketch_sizes_naive = [20, 50, 100, 200]
    
    # Run optimized benchmarks
    print("\n[1/2] Benchmarking OPTIMIZED implementations...")
    results_opt = benchmark_optimized(n_opt, sketch_sizes_opt)
    
    # Run naive benchmarks (smaller sizes due to slowness)
    print("\n[2/2] Benchmarking NAIVE (pure-Python) implementations...")
    results_naive = benchmark_naive(n_naive, sketch_sizes_naive)
    
    # Create figure
    print("\n[3/3] Creating Figure 1...")
    fig = create_figure_1(
        results_opt, results_naive,
        sketch_sizes_opt, sketch_sizes_naive,
        n_opt, n_naive,
        output_dir
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. OPTIMIZED: BLAS-accelerated Gaussian is competitive with SRFT/SRHT
   for practical problem sizes due to highly optimized DGEMM.

2. NAIVE: Pure-Python comparison reveals true algorithmic complexity:
   - Gaussian: O(n²ℓ) — linear in ℓ
   - SRFT/SRHT: O(n² log n) — nearly constant in ℓ
   
3. The algorithmic advantage of structured sketches is real but often
   hidden by library optimization constants.
""")


if __name__ == "__main__":
    main()
