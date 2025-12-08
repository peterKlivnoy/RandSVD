"""
Experiment 2: Speed Comparison on Sparse Matrices

This experiment compares sketching methods on sparse matrices:
  - Gaussian sketch: O(nnz · ℓ) - scales with both nnz and sketch size
  - CountSketch: O(nnz) - independent of sketch size ℓ!

We plot runtime vs nnz for multiple values of ℓ to show the scaling difference.

Figures produced:
  - Fig 2: Runtime comparison on sparse matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from pathlib import Path
import time
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sparse_sketching import countsketch_operator

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

# Color scheme for different ℓ values
COLORS_L = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']  # Blue, Green, Orange, Pink
MARKERS = ['o', 's', '^', 'D']


def benchmark_multi_l(m, n, sketch_sizes, densities, num_trials=5):
    """Benchmark methods across different ℓ and density combinations."""
    results = {
        'gaussian': {l: [] for l in sketch_sizes},
        'countsketch': {l: [] for l in sketch_sizes},
        'nnz': []
    }
    
    print(f"\n  Matrix size: {m}×{n}")
    
    for density in densities:
        A = sparse_random(m, n, density=density, format='csr', random_state=42)
        nnz = A.nnz
        results['nnz'].append(nnz)
        
        print(f"    Density {density*100:.2f}% (nnz={nnz:,}):")
        
        for l in sketch_sizes:
            # Gaussian (sparse @ dense)
            times_g = []
            for seed in range(num_trials):
                np.random.seed(seed)
                Omega = np.random.randn(n, l)
                t0 = time.perf_counter()
                Y = A @ Omega
                times_g.append(time.perf_counter() - t0)
            results['gaussian'][l].append(np.median(times_g))
            
            # CountSketch
            times_cs = []
            for seed in range(num_trials):
                t0 = time.perf_counter()
                Y = countsketch_operator(A, l, seed=seed)
                times_cs.append(time.perf_counter() - t0)
            results['countsketch'][l].append(np.median(times_cs))
            
            print(f"      l={l}: G={np.median(times_g)*1000:.1f}ms, CS={np.median(times_cs)*1000:.1f}ms")
    
    return results


def create_figure_2(results, sketch_sizes, m, n, output_dir):
    """Create publication-quality Figure 2: Speed comparison on sparse matrices."""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    nnz_vals = np.array(results['nnz'])
    
    # ===== Panel A: Gaussian - shows ℓ dependence =====
    ax = axes[0]
    
    for i, l in enumerate(sketch_sizes):
        times_ms = np.array(results['gaussian'][l]) * 1000
        ax.plot(nnz_vals, times_ms, 
                marker=MARKERS[i], color=COLORS_L[i],
                linewidth=2, markersize=7, label=f'$\\ell={l}$')
    
    # Theoretical lines (extend beyond data)
    nnz_extended = np.concatenate([nnz_vals, [nnz_vals[-1] * 2, nnz_vals[-1] * 3]])
    
    # Show O(nnz · ℓ) scaling - lines should be proportional to ℓ
    for i, l in enumerate(sketch_sizes):
        t_ref = np.array(results['gaussian'][l])[0] * 1000
        theory = t_ref * (nnz_extended / nnz_vals[0]) * 0.8
        ax.plot(nnz_extended, theory, '--', color=COLORS_L[i], alpha=0.4, linewidth=1.5)
    
    ax.set_xlabel('Number of nonzeros (nnz)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(a) Gaussian Sketch: $O(\\mathrm{{nnz}} \\cdot \\ell)$\n${m} \\times {n}$ matrix', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, title='Sketch size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin / 2, ymax * 4)
    
    # ===== Panel B: CountSketch - shows ℓ independence =====
    ax = axes[1]
    
    for i, l in enumerate(sketch_sizes):
        times_ms = np.array(results['countsketch'][l]) * 1000
        ax.plot(nnz_vals, times_ms, 
                marker=MARKERS[i], color=COLORS_L[i],
                linewidth=2, markersize=7, label=f'$\\ell={l}$')
    
    # Theoretical line - all ℓ values should collapse to same line!
    t_ref = np.array(results['countsketch'][sketch_sizes[0]])[0] * 1000
    theory_cs = t_ref * (nnz_extended / nnz_vals[0]) * 0.8
    ax.plot(nnz_extended, theory_cs, '--', color='black', alpha=0.5, linewidth=2,
            label=r'$O(\mathrm{nnz})$')
    
    ax.set_xlabel('Number of nonzeros (nnz)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'(b) CountSketch: $O(\\mathrm{{nnz}})$\n${m} \\times {n}$ matrix', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, title='Sketch size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin / 2, ymax * 4)
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig2_speed_sparse.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig2_speed_sparse.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def plot_from_json():
    """Load results from JSON and regenerate the figure without running benchmarks."""
    output_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    json_path = data_dir / 'experiment_2_results.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to int for sketch_sizes
    results = {
        'nnz': data['results']['nnz'],
        'gaussian': {int(k): v for k, v in data['results']['gaussian'].items()},
        'countsketch': {int(k): v for k, v in data['results']['countsketch'].items()},
    }
    
    print("Creating Figure 2 from saved data...")
    fig = create_figure_2(
        results, data['sketch_sizes'],
        data['m'], data['n'],
        output_dir
    )
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 2: Speed Comparison on Sparse Matrices")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Parameters
    m, n = 20000, 10000
    sketch_sizes = [50, 100, 200, 400]  # Multiple ℓ values to show scaling
    densities = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    
    print("\n[1/2] Benchmarking with multiple sketch sizes...")
    results = benchmark_multi_l(m, n, sketch_sizes, densities)
    
    # Save results to JSON
    # Convert int keys to strings for JSON compatibility
    results_json = {
        'nnz': results['nnz'],
        'gaussian': {str(k): v for k, v in results['gaussian'].items()},
        'countsketch': {str(k): v for k, v in results['countsketch'].items()},
    }
    
    results_data = {
        'm': m,
        'n': n,
        'sketch_sizes': sketch_sizes,
        'densities': densities,
        'results': results_json,
    }
    
    json_path = data_dir / 'experiment_2_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Saved results to: {json_path}")
    
    # Create figure
    print("\n[2/2] Creating Figure 2...")
    fig = create_figure_2(results, sketch_sizes, m, n, output_dir)


if __name__ == "__main__":
    main()
