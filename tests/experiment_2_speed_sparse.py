"""
Experiment 2: Sparse Sketching Speed Comparison

Compares sketching methods for sparse matrices:
1. Gaussian (dense) - O(n^2) baseline
2. CountSketch - O(nnz) input sparsity time
3. Sparse Sign Embedding - O(s * nnz) tunable

Figures produced:
  - Fig 2: Speed vs sparsity for different sketching methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from pathlib import Path
import time
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sparse_sketching import countsketch_operator, sparse_sign_embedding

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
    'sparse_sign': '#009E73',
}
MARKERS = {'gaussian': 'o', 'countsketch': 's', 'sparse_sign': '^'}


def experiment_speed_limit(m, n, densities, l, sparsity_s=4, num_trials=5):
    results = {
        'densities': densities,
        'nnz': [],
        'gaussian': [],
        'countsketch': [],
        'sparse_sign': [],
    }
    
    print(f"\n  Matrix size: {m}x{n}, sketch size l={l}, sparse_sign s={sparsity_s}")
    
    for density in densities:
        A_sparse = sparse_random(m, n, density=density, format='csr', random_state=42)
        nnz = A_sparse.nnz
        results['nnz'].append(nnz)
        
        A_dense = A_sparse.toarray()
        
        print(f"    Density {density*100:.2f}% (nnz={nnz:,}): ", end="", flush=True)
        
        # Gaussian with DENSE multiplication
        times_g = []
        for seed in range(num_trials):
            np.random.seed(seed)
            Omega = np.random.randn(n, l)
            t0 = time.perf_counter()
            Y = A_dense @ Omega
            times_g.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times_g))
        print(f"G={np.median(times_g)*1000:.1f}ms ", end="")
        
        # CountSketch
        times_cs = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = countsketch_operator(A_sparse, l, seed=seed)
            times_cs.append(time.perf_counter() - t0)
        results['countsketch'].append(np.median(times_cs))
        print(f"CS={np.median(times_cs)*1000:.1f}ms ", end="")
        
        # Sparse Sign Embedding
        times_ss = []
        for seed in range(num_trials):
            t0 = time.perf_counter()
            Y = sparse_sign_embedding(A_sparse, l, sparsity=sparsity_s, seed=seed)
            times_ss.append(time.perf_counter() - t0)
        results['sparse_sign'].append(np.median(times_ss))
        print(f"SS={np.median(times_ss)*1000:.1f}ms")
    
    return results


def create_figure_2(results, m, n, l, sparsity_s, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    
    nnz_vals = np.array(results['nnz'])
    
    ax.plot(nnz_vals, np.array(results['gaussian']) * 1000,
            marker=MARKERS['gaussian'], color=COLORS['gaussian'],
            linewidth=2, markersize=8, label='Gaussian (dense)')
    
    ax.plot(nnz_vals, np.array(results['countsketch']) * 1000,
            marker=MARKERS['countsketch'], color=COLORS['countsketch'],
            linewidth=2, markersize=8, label='CountSketch')
    
    ax.plot(nnz_vals, np.array(results['sparse_sign']) * 1000,
            marker=MARKERS['sparse_sign'], color=COLORS['sparse_sign'],
            linewidth=2, markersize=8, label=f'Sparse Sign (s={sparsity_s})')
    
    # Theoretical: Gaussian flat - anchor at 80% of mean
    g_flat = np.array(results['gaussian']).mean() * 1000 * 0.8
    ax.axhline(y=g_flat, color=COLORS['gaussian'], linestyle='--', 
               alpha=0.5, linewidth=1.5, label=r'$O(mn)$ bound')
    
    # Theoretical: CountSketch linear - anchor at 80% of first point
    nnz_theory = np.array([nnz_vals[0], nnz_vals[-1] * 1.5])
    cs_ref = np.array(results['countsketch'])[0] * 1000
    cs_theory = cs_ref * (nnz_theory / nnz_vals[0]) * 0.8
    ax.plot(nnz_theory, cs_theory, '--', color=COLORS['countsketch'], 
            alpha=0.5, linewidth=1.5, label=r'$O(\mathrm{nnz})$')
    
    # Theoretical: Sparse Sign linear
    ss_ref = np.array(results['sparse_sign'])[0] * 1000
    ss_theory = ss_ref * (nnz_theory / nnz_vals[0]) * 0.8
    ax.plot(nnz_theory, ss_theory, '--', color=COLORS['sparse_sign'], 
            alpha=0.5, linewidth=1.5, label=r'$O(s \cdot \mathrm{nnz})$')
    
    ax.set_xlabel('Number of nonzeros (nnz)')
    ax.set_ylabel('Sketch Time (ms)')
    ax.set_title(f'Sparse Sketching Speed Comparison\n${m} \\times {n}$ matrix, $\\ell={l}$', fontweight='bold')
    
    # Legend in lower right to avoid blocking Gaussian line at top
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set reasonable y limits - expand upward to give Gaussian breathing room
    ymin = min(np.array(results['countsketch']).min(), 
               np.array(results['sparse_sign']).min()) * 1000 * 0.3
    ymax = np.array(results['gaussian']).max() * 1000 * 4
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig2_speed_sparse.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig2_speed_sparse.png')
    print(f"\n  Saved: {output_path}")
    
    return fig


def plot_from_json():
    output_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    json_path = data_dir / 'experiment_2_results.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("Creating Figure 2 from saved data...")
    fig = create_figure_2(data['results'], data['m'], data['n'], data['l'], data['sparsity_s'], output_dir)
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 2: Sparse Sketching Speed Comparison")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Larger rectangular matrix to avoid fixed-cost floor at low density
    m = 20000
    n = 30000
    densities = [0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]  # Stop at 0.1% to avoid fixed costs
    l = 100
    sparsity_s = 4
    
    print("\nHypothesis: CountSketch achieves O(nnz), Gaussian stays O(mn)")
    
    results = experiment_speed_limit(m, n, densities, l, sparsity_s)
    
    save_data = {'results': results, 'm': m, 'n': n, 'l': l, 'sparsity_s': sparsity_s}
    json_path = data_dir / 'experiment_2_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved results to: {json_path}")
    
    print("\n" + "="*70)
    print("CREATING FIGURE")
    print("="*70)
    
    fig = create_figure_2(results, m, n, l, sparsity_s, output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    g_times = np.array(results['gaussian']) * 1000
    cs_times = np.array(results['countsketch']) * 1000
    ss_times = np.array(results['sparse_sign']) * 1000
    
    print(f"\nAt lowest density (density={densities[-1]*100:.2f}%):")
    print(f"  Gaussian:     {g_times[-1]:.1f} ms")
    print(f"  CountSketch:  {cs_times[-1]:.1f} ms  ({g_times[-1]/cs_times[-1]:.0f}x faster)")
    print(f"  Sparse Sign:  {ss_times[-1]:.1f} ms  ({g_times[-1]/ss_times[-1]:.0f}x faster)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        plot_from_json()
    else:
        main()
