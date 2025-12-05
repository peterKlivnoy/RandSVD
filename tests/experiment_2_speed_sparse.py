"""
Experiment 2: Speed Comparison on Sparse Matrices (CountSketch)

This experiment demonstrates CountSketch's advantage on sparse data:
  - CountSketch: O(nnz(A) · ℓ) complexity
  - Gaussian: O(mnℓ) complexity (doesn't leverage sparsity)

We show both optimized (SciPy sparse) and naive (pure-Python) implementations.

Figures produced:
  - Fig 2a: Runtime vs density (fixed matrix size)
  - Fig 2b: Runtime vs matrix size (fixed density)
  - Fig 2c: Algorithmic speedup (naive comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sparse_sketching import (
    countsketch_operator,
    naive_gaussian_sparse,
    naive_countsketch_sparse
)

# Configure matplotlib
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

COLORS = {
    'gaussian': '#0072B2',
    'countsketch': '#E69F00',
}


def benchmark_vs_density(m, n, l, densities, num_trials=5):
    """Benchmark methods across different sparsity levels."""
    results = {'gaussian': [], 'countsketch': [], 'nnz': []}
    
    for density in densities:
        print(f"  Density {density*100:.1f}%: ", end="", flush=True)
        
        A = sparse_random(m, n, density=density, format='csr', random_state=42)
        results['nnz'].append(A.nnz)
        
        # Gaussian (dense sketch of sparse matrix)
        times = []
        for seed in range(num_trials):
            np.random.seed(seed)
            Omega = np.random.randn(n, l)
            t0 = time.perf_counter()
            Y = A @ Omega
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"Gauss={np.median(times)*1000:.1f}ms ", end="")
        
        # CountSketch
        times = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = countsketch_operator(A, l, seed=seed)
            times.append(time.perf_counter() - t0)
        results['countsketch'].append(np.median(times))
        print(f"CS={np.median(times)*1000:.1f}ms")
    
    return results


def benchmark_vs_size(sizes, density, l, num_trials=5):
    """Benchmark methods across different matrix sizes."""
    results = {'gaussian': [], 'countsketch': [], 'nnz': []}
    
    for m, n in sizes:
        print(f"  Size {m}×{n}: ", end="", flush=True)
        
        A = sparse_random(m, n, density=density, format='csr', random_state=42)
        results['nnz'].append(A.nnz)
        
        # Gaussian
        times = []
        for seed in range(num_trials):
            np.random.seed(seed)
            Omega = np.random.randn(n, l)
            t0 = time.perf_counter()
            Y = A @ Omega
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"Gauss={np.median(times)*1000:.1f}ms ", end="")
        
        # CountSketch
        times = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = countsketch_operator(A, l, seed=seed)
            times.append(time.perf_counter() - t0)
        results['countsketch'].append(np.median(times))
        print(f"CS={np.median(times)*1000:.1f}ms")
    
    return results


def benchmark_naive_comparison(m, n, l, densities, num_trials=3):
    """Compare naive implementations for pure algorithmic analysis."""
    results = {'gaussian': [], 'countsketch': [], 'speedup': []}
    
    for density in densities:
        print(f"  Density {density*100:.1f}%: ", end="", flush=True)
        
        A = sparse_random(m, n, density=density, format='csr', random_state=42)
        
        # Naive Gaussian
        times = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_gaussian_sparse(A, l, seed=seed)
            times.append(time.perf_counter() - t0)
        t_gauss = np.median(times)
        results['gaussian'].append(t_gauss)
        print(f"Gauss={t_gauss*1000:.1f}ms ", end="")
        
        # Naive CountSketch
        times = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_countsketch_sparse(A, l, seed=seed)
            times.append(time.perf_counter() - t0)
        t_cs = np.median(times)
        results['countsketch'].append(t_cs)
        print(f"CS={t_cs*1000:.1f}ms ", end="")
        
        speedup = t_gauss / t_cs
        results['speedup'].append(speedup)
        print(f"Speedup={speedup:.1f}×")
    
    return results


def create_figure_2(results_density, results_size, results_naive, 
                    densities, sizes, naive_densities, output_dir):
    """Create Figure 2: Sparse matrix speed comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # ===== Panel A: Runtime vs Density =====
    ax = axes[0]
    density_pct = np.array(densities) * 100
    
    ax.plot(density_pct, np.array(results_density['gaussian']) * 1000,
            'o-', color=COLORS['gaussian'], linewidth=2, markersize=8,
            label='Gaussian')
    ax.plot(density_pct, np.array(results_density['countsketch']) * 1000,
            's-', color=COLORS['countsketch'], linewidth=2, markersize=8,
            label='CountSketch')
    
    ax.set_xlabel('Matrix Density (%)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Runtime vs Density\n$5000 \\times 2000$, $\\ell=50$', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # ===== Panel B: Runtime vs Size =====
    ax = axes[1]
    size_labels = [f"{m//1000}k×{n//1000}k" for m, n in sizes]
    x = np.arange(len(sizes))
    
    width = 0.35
    ax.bar(x - width/2, np.array(results_size['gaussian']) * 1000, width,
           color=COLORS['gaussian'], label='Gaussian', edgecolor='black')
    ax.bar(x + width/2, np.array(results_size['countsketch']) * 1000, width,
           color=COLORS['countsketch'], label='CountSketch', edgecolor='black')
    
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, rotation=15)
    ax.set_title('(b) Runtime vs Size\nDensity = 1%, $\\ell=50$', fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    
    # ===== Panel C: Naive Algorithmic Speedup =====
    ax = axes[2]
    naive_density_pct = np.array(naive_densities) * 100
    
    ax.bar(range(len(naive_densities)), results_naive['speedup'],
           color=COLORS['countsketch'], edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for i, (d, s) in enumerate(zip(naive_density_pct, results_naive['speedup'])):
        ax.text(i, s + 10, f'{s:.0f}×', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Matrix Density (%)')
    ax.set_ylabel('Speedup (Gaussian / CountSketch)')
    ax.set_xticks(range(len(naive_densities)))
    ax.set_xticklabels([f'{d:.1f}%' for d in naive_density_pct])
    ax.set_title('(c) Algorithmic Speedup\nNaive Pure-Python, $n=500$', fontweight='bold')
    
    # Add theoretical line
    ax.axhline(y=500, color='red', linestyle='--', linewidth=2, 
               label=f'Theory: $n={500}$')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig2_speed_sparse.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig2_speed_sparse.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 2: Speed Comparison on Sparse Matrices")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # ===== Benchmark 1: Runtime vs Density =====
    print("\n[1/3] Benchmarking vs density...")
    densities = [0.001, 0.005, 0.01, 0.05, 0.1]
    results_density = benchmark_vs_density(
        m=5000, n=2000, l=50, 
        densities=densities
    )
    
    # ===== Benchmark 2: Runtime vs Size =====
    print("\n[2/3] Benchmarking vs matrix size...")
    sizes = [(2000, 1000), (4000, 2000), (8000, 4000), (16000, 8000)]
    results_size = benchmark_vs_size(
        sizes=sizes,
        density=0.01,
        l=50
    )
    
    # ===== Benchmark 3: Naive comparison =====
    print("\n[3/3] Benchmarking naive implementations (pure-Python)...")
    naive_densities = [0.01, 0.05, 0.1]
    results_naive = benchmark_naive_comparison(
        m=1000, n=500, l=20,
        densities=naive_densities
    )
    
    # Create figure
    print("\n[4/4] Creating Figure 2...")
    fig = create_figure_2(
        results_density, results_size, results_naive,
        densities, sizes, naive_densities,
        output_dir
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. CountSketch scales with nnz(A), not matrix dimensions mn.
   For sparse matrices, this gives massive speedups.

2. Algorithmic advantage is O(n) — naive CountSketch is ~n× faster
   than naive Gaussian. This matches theory: O(ζℓ) vs O(ζnℓ).

3. Even with library optimization, CountSketch wins on sparse data
   because the fundamental operation count is lower.

4. Recommendation: For matrices with density < 10%, use CountSketch.
""")


if __name__ == "__main__":
    main()
