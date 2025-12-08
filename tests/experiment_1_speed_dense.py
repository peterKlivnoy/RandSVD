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
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.structured_sketch import gaussian_operator, srft_operator, srht_operator
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
    # for trial in range(num_trials):
    #     t0 = time.perf_counter()
    #     U, s, Vt = np.linalg.svd(A, full_matrices=False)
    #     times.append(time.perf_counter() - t0)
    # results['full_svd'] = np.median(times)
    # print(f"    Full SVD: {results['full_svd']*1000:.1f}ms")
    
    for l in sketch_sizes:
        if l > n:
            for method in ['gaussian', 'srht', 'randsvd']:
                results[method].append(np.nan)
            continue
            
        print(f"    l={l}: ", end="", flush=True)
        
        # Gaussian sketch
        times = []
        for trial in range(num_trials):
            np.random.seed(trial)
            
            t0 = time.perf_counter()
        
            Y = gaussian_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"G={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        
        # SRHT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srht_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # SRFT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srft_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.1f}ms ", end="", flush=True)
 
    
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
        
        # Naive SRHT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_srht_sketch(A, l, seed=trial)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.1f}ms ", end="", flush=True)
        
        # Naive SRFT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = naive_srft_sketch(A, l, seed=trial)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.1f}ms")
    
    return results


def create_figure_1(results_opt, results_naive, sketch_sizes_opt, sketch_sizes_naive, n_opt, n_naive, output_dir):
    """Create publication-quality Figure 1: Speed comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # ===== Panel A: Optimized implementations =====
    ax = axes[0]
    
    for method in ['gaussian', 'srht', 'srft']:
        times_ms = np.array(results_opt[method]) * 1000
        valid = ~np.isnan(times_ms)
        ax.plot(np.array(sketch_sizes_opt)[valid], times_ms[valid], 
                marker=MARKERS[method], color=COLORS[method],
                linewidth=2, markersize=8, label=f'{method.upper()} sketch')
    
    # Add theoretical complexity lines
    l_vals = np.array(sketch_sizes_opt)
    valid_l = l_vals <= n_opt
    l_theory = l_vals[valid_l]
    
    # Extend x-axis for theoretical lines (3 extra points beyond data)
    l_extended = np.concatenate([l_theory, [400, 450, 500]])
    
    # Scale theoretical lines to match data at first point
    t_gauss_data = np.array(results_opt['gaussian'])[0] * 1000
    t_srht_data = np.array(results_opt['srht'])[0] * 1000
    
    # Gaussian: O(n^2 * l) -> scale by l
    gauss_theory = t_gauss_data * (l_extended / l_theory[0])
    # SRHT: O(n^2 * log(n)) - independent of l, so flat line
    srht_theory = t_srht_data * (np.log(n_opt) / np.log(n_opt)) * np.ones_like(l_extended)
    
    ax.plot(l_extended, gauss_theory, '--', color=COLORS['gaussian'], alpha=0.5, 
            linewidth=1.5, label=r'$O(n^2 \ell)$')
    ax.plot(l_extended, srht_theory, '--', color=COLORS['srht'], alpha=0.5, 
            linewidth=1.5, label=r'$O(n^2 \log n)$ (SRHT)')
    # SRFT theoretical line (same complexity as SRHT)
    t_srft_data = np.array(results_opt['srft'])[0] * 1000
    srft_theory = t_srft_data * np.ones_like(l_extended)
    ax.plot(l_extended, srft_theory, '--', color=COLORS['srft'], alpha=0.5, 
            linewidth=1.5, label=r'$O(n^2 \log n)$ (SRFT)')
    
    # Set x-axis to include extended range
    ax.set_xlim(l_theory[0] - 20, l_extended[-1] + 20)
    
    ax.set_xlabel('Sketch size ($\\ell$)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(a) Optimized Implementations\n$n = {n_opt}$', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    
    # Increase y-axis limits to give room for legend (top) and space at bottom
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin / 2, ymax * 10)
    
    # ===== Panel B: Naive implementations =====
    ax = axes[1]
    
    times_g = np.array(results_naive['gaussian']) * 1000
    times_srht = np.array(results_naive['srht']) * 1000
    times_srft = np.array(results_naive['srft']) * 1000
    
    valid_g = ~np.isnan(times_g)
    valid_srht = ~np.isnan(times_srht)
    valid_srft = ~np.isnan(times_srft)
    
    ax.plot(np.array(sketch_sizes_naive)[valid_g], times_g[valid_g], 
            marker='o', color=COLORS['gaussian'],
            linewidth=2, markersize=8, label='Gaussian')

    ax.plot(np.array(sketch_sizes_naive)[valid_srht], times_srht[valid_srht], 
            marker='^', color=COLORS['srht'],
            linewidth=2, markersize=8, label='SRHT (naive)')
    
    ax.plot(np.array(sketch_sizes_naive)[valid_srft], times_srft[valid_srft], 
            marker='s', color=COLORS['srft'],
            linewidth=2, markersize=8, label='SRFT (naive)')
    
    # Add theoretical complexity lines for naive panel
    l_naive = np.array(sketch_sizes_naive)
    valid_naive = ~np.isnan(times_g)
    l_theory_naive = l_naive[valid_naive]
    
    # Extend x-axis for theoretical lines (3 extra points beyond data)
    l_extended_naive = np.concatenate([l_theory_naive, [400, 450, 500]])
    
    # Anchor theoretical lines to their respective data lines, shifted down a bit
    t_gauss_naive = times_g[valid_naive][0] * 0.8
    t_srht_naive = times_srht[valid_naive][0] * 0.8
    t_srft_naive = times_srft[valid_naive][0] * 0.8
    
    # Gaussian: O(n^2 * l)
    gauss_theory_naive = t_gauss_naive * (l_extended_naive / l_theory_naive[0])
    # SRHT: O(n^2 * log(n)) - flat line
    srht_theory_naive = t_srht_naive * np.ones_like(l_extended_naive)
    # SRFT: O(n^2 * log(n)) - flat line
    srft_theory_naive = t_srft_naive * np.ones_like(l_extended_naive)
    
    ax.plot(l_extended_naive, gauss_theory_naive, '--', color=COLORS['gaussian'], 
            alpha=0.5, linewidth=1.5, label=r'$O(n^2 \ell)$')
    ax.plot(l_extended_naive, srht_theory_naive, '--', color=COLORS['srht'], 
            alpha=0.5, linewidth=1.5, label=r'$O(n^2 \log n)$ (SRHT)')
    # SRFT theoretical line
    ax.plot(l_extended_naive, srft_theory_naive, '--', color=COLORS['srft'], 
            alpha=0.5, linewidth=1.5, label=r'$O(n^2 \log n)$ (SRFT)')
    
    # Set x-axis to include extended range
    ax.set_xlim(l_theory_naive[0] - 5, l_extended_naive[-1] + 5)
    
    ax.set_xlabel('Sketch size ($\\ell$)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(b) Naive Pure-Python\n$n = {n_naive}$', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    
    # Increase y-axis limits to give room for legend (top) and space at bottom
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin / 2, ymax * 10)
    
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig1_speed_dense.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig1_speed_dense.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def plot_from_json():
    """Load results from JSON and regenerate the figure without running benchmarks."""
    output_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    json_path = data_dir / 'experiment_1_results.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    sketch_sizes = data['sketch_sizes']
    results_opt = data['optimized']
    results_naive = data['naive']
    
    print("Creating Figure 1 from saved data...")
    fig = create_figure_1(
        results_opt, results_naive,
        sketch_sizes, sketch_sizes,
        n, n,
        output_dir
    )
    return fig


def run_loglog_experiment(n_power=14, l_base=5, l_powers=(2, 3, 4, 5), num_trials=5):
    """
    Run optimized sketching experiment with log-log plot.
    
    Args:
        n_power: Matrix size n = 2^n_power (default 14 -> n=16384)
        l_base: Base for sketch sizes (default 5)
        l_powers: Tuple of powers for l = l_base^i
        num_trials: Number of trials per configuration
    
    Returns:
        results: Dict with timing results
    """
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    n = 2**n_power
    sketch_sizes = [l_base**i for i in l_powers]
    
    print("="*70)
    print("EXPERIMENT 1: Speed Comparison (Log-Log Scale)")
    print("="*70)
    print(f"\nn = 2^{n_power} = {n}")
    print(f"l = {l_base}^i for i in {l_powers}: {sketch_sizes}")
    
    # Create random matrix
    print(f"\nCreating {n}×{n} random matrix...")
    np.random.seed(42)
    A = np.random.randn(n, n)
    print("  Done.")
    
    results = {'gaussian': [], 'srft': [], 'srht': []}
    
    print(f"\nBenchmarking ({num_trials} trials each)...")
    for l in sketch_sizes:
        print(f"\n  l={l}:", end=" ", flush=True)
        
        # Gaussian
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = gaussian_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"G={np.median(times)*1000:.1f}ms", end=" ", flush=True)
        
        # SRHT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srht_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.1f}ms", end=" ", flush=True)
        
        # SRFT
        times = []
        for trial in range(num_trials):
            t0 = time.perf_counter()
            Y = srft_operator(A, l)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.1f}ms", end="", flush=True)
    
    print("\n")
    
    # Create log-log plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    l_arr = np.array(sketch_sizes)
    
    # Plot data
    ax.loglog(l_arr, np.array(results['gaussian'])*1000, 'o-', color=COLORS['gaussian'],
              linewidth=2, markersize=8, label='Gaussian')
    ax.loglog(l_arr, np.array(results['srht'])*1000, '^-', color=COLORS['srht'],
              linewidth=2, markersize=8, label='SRHT')
    ax.loglog(l_arr, np.array(results['srft'])*1000, 's-', color=COLORS['srft'],
              linewidth=2, markersize=8, label='SRFT')
    
    # Add theoretical complexity lines
    l_theory = np.array([sketch_sizes[0], sketch_sizes[-1]])
    
    # Gaussian: O(n^2 * l) -> scales linearly with l
    t_gauss_ref = results['gaussian'][0] * 1000
    gauss_theory = t_gauss_ref * (l_theory / l_theory[0])
    ax.loglog(l_theory, gauss_theory, '--', color=COLORS['gaussian'], alpha=0.5, 
              linewidth=1.5, label=r'$O(n^2 \ell)$')
    
    # SRHT/SRFT: O(n^2 log n) -> constant in l (flat line)
    t_srht_ref = results['srht'][0] * 1000
    srht_theory = np.ones_like(l_theory) * t_srht_ref
    ax.loglog(l_theory, srht_theory, '--', color=COLORS['srht'], alpha=0.5, 
              linewidth=1.5, label=r'$O(n^2 \log n)$')
    
    ax.set_xlabel('Sketch size $\\ell$')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Sketching Time vs. Sketch Size (Log-Log)\n$n = 2^{{{n_power}}} = {n}$', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    
    # Set x-ticks to show base^i values
    ax.set_xticks(sketch_sizes)
    ax.set_xticklabels([f'${l_base}^{i}$' for i in l_powers])
    
    # Increase y-axis limits to give room for legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig1_speed_dense_loglog.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig1_speed_dense_loglog.png')
    print(f"✓ Saved: {output_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'l':>6} | {'Gaussian':>10} | {'SRHT':>10} | {'SRFT':>10}")
    print("-"*45)
    for i, l in enumerate(sketch_sizes):
        print(f"{l:>6} | {results['gaussian'][i]*1000:>9.1f}ms | {results['srht'][i]*1000:>9.1f}ms | {results['srft'][i]*1000:>9.1f}ms")
    
    return results


def main():
    print("="*70)
    print("EXPERIMENT 1: Speed Comparison on Dense Matrices")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Parameters - same n and l's for both
    n = 2048
    sketch_sizes = [50*i for i in range(1, 8)]  # 50 to 350
    
    # Run optimized benchmarks
    print("\n[1/2] Benchmarking OPTIMIZED implementations...")
    results_opt = benchmark_optimized(n, sketch_sizes)
    
    # Run naive benchmarks (same sizes now)
    print("\n[2/2] Benchmarking NAIVE (pure-Python) implementations...")
    results_naive = benchmark_naive(n, sketch_sizes)
    
    # Save results to JSON for later use
    results_data = {
        'n': n,
        'sketch_sizes': sketch_sizes,
        'optimized': {
            'gaussian': results_opt['gaussian'],
            'srht': results_opt['srht'],
            'srft': results_opt['srft'],
        },
        'naive': {
            'gaussian': results_naive['gaussian'],
            'srht': results_naive['srht'],
            'srft': results_naive['srft'],
        }
    }
    
    json_path = data_dir / 'experiment_1_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Saved results to: {json_path}")
    
    # Create figure
    print("\n[3/3] Creating Figure 1...")
    fig = create_figure_1(
        results_opt, results_naive,
        sketch_sizes, sketch_sizes,
        n, n,
        output_dir
    )
    


if __name__ == "__main__":
    main()
