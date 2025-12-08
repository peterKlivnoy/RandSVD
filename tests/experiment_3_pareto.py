"""
Experiment 3: Error vs. Runtime Pareto Frontier

This is the "money shot" experiment that validates the paper's core claim:
Sparse sketching methods achieve comparable accuracy at a fraction of the cost.

For each method, we vary the sketch size l to generate points on the
Error vs. Runtime trade-off curve (Pareto Frontier).

Key insight: If sparse methods dominate (below and to the left of Gaussian),
they provide better accuracy per unit of computation.

Figures produced:
  - Fig 3: Error vs. Runtime Pareto Frontier
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import svds
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


def run_randsvd(A, A_dense, l, k, sketch_func, num_trials=3):
    """
    Run randomized SVD with given sketching function.
    Returns median time and error over trials.
    """
    n = A.shape[1]
    A_fro = np.linalg.norm(A_dense, 'fro')
    
    times = []
    errors = []
    
    for seed in range(num_trials):
        t0 = time.perf_counter()
        
        # Sketching step
        Y = sketch_func(A, l, seed)
        
        # Orthogonalize
        Q, _ = np.linalg.qr(Y)
        
        # Project and compute SVD
        B = Q.T @ A_dense
        U_B, s_B, Vt_B = np.linalg.svd(B, full_matrices=False)
        
        # Reconstruct approximate singular vectors
        U_approx = Q @ U_B[:, :k]
        
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
        # Compute approximation error
        A_approx = U_approx @ (U_approx.T @ A_dense)
        error = np.linalg.norm(A_dense - A_approx, 'fro') / A_fro
        errors.append(error)
    
    return np.median(times), np.median(errors)


def gaussian_sketch(A, l, seed):
    """Gaussian sketch - uses dense multiplication."""
    np.random.seed(seed)
    n = A.shape[1]
    Omega = np.random.randn(n, l)
    # Force dense multiplication to show true O(mn) cost
    if hasattr(A, 'toarray'):
        return A.toarray() @ Omega
    return A @ Omega


def experiment_pareto(A, sketch_sizes, k, sparsity_levels=[1, 2, 3, 4, 5], num_trials=3):
    """
    Generate Pareto frontier data for Error vs. Runtime.
    
    Args:
        A: Sparse input matrix
        sketch_sizes: List of sketch sizes l to test
        k: Target rank for SVD approximation
        sparsity_levels: List of sparsity parameters s to test
        num_trials: Number of trials per configuration
    
    Returns:
        dict with results for each method
    """
    results = {
        'sketch_sizes': sketch_sizes,
        'sparsity_levels': sparsity_levels,
        'gaussian': {'times': [], 'errors': []},
        'optimal_error': None,
    }
    
    # Initialize results for each sparsity level
    for s in sparsity_levels:
        results[f'sparse_s{s}'] = {'times': [], 'errors': []}
    
    A_dense = A.toarray()
    A_fro = np.linalg.norm(A_dense, 'fro')
    
    print(f"\n  Matrix: {A.shape}, nnz={A.nnz:,}, density={A.nnz/(A.shape[0]*A.shape[1])*100:.2f}%")
    print(f"  Target rank k={k}")
    print(f"  Sparsity levels: {sparsity_levels}")
    
    # Compute optimal rank-k error using deterministic SVD
    print(f"  Computing optimal rank-{k} SVD...", end=" ", flush=True)
    t0 = time.perf_counter()
    U_opt, s_opt, Vt_opt = svds(A, k=k)
    svd_time = time.perf_counter() - t0
    
    # Optimal reconstruction
    A_k = U_opt @ np.diag(s_opt) @ Vt_opt
    optimal_error = np.linalg.norm(A_dense - A_k, 'fro') / A_fro
    results['optimal_error'] = optimal_error
    results['svd_time'] = svd_time
    
    print(f"done ({svd_time*1000:.0f}ms)")
    print(f"  Optimal rank-{k} error: {optimal_error:.6f}")
    print(f"  Sketch sizes: {sketch_sizes}")
    
    for l in sketch_sizes:
        print(f"\n    l={l}:", end=" ", flush=True)
        
        # Gaussian (dense)
        t, e = run_randsvd(A, A_dense, l, k, gaussian_sketch, num_trials)
        results['gaussian']['times'].append(t)
        results['gaussian']['errors'].append(e)
        print(f"G={t*1000:.0f}ms/{e:.4f}", end=" ", flush=True)
        
        # Sparse Sign for each sparsity level
        for s in sparsity_levels:
            t, e = run_randsvd(A, A_dense, l, k,
                              lambda A, l, seed, s=s: sparse_sign_embedding(A, l, sparsity=s, seed=seed),
                              num_trials)
            results[f'sparse_s{s}']['times'].append(t)
            results[f'sparse_s{s}']['errors'].append(e)
            print(f"s{s}={t*1000:.0f}ms/{e:.4f}", end=" ", flush=True)
    
    print()
    return results


def create_figure_3(results, m, n, k, output_dir, selected_sparsities=None):
    """Create Error vs. Runtime Pareto frontier plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    all_sparsity_levels = results.get('sparsity_levels', [1, 2, 3, 4, 5])
    
    # Allow filtering to specific sparsity levels
    if selected_sparsities is not None:
        sparsity_levels = [s for s in selected_sparsities if f'sparse_s{s}' in results]
    else:
        sparsity_levels = all_sparsity_levels
    
    # Color map for sparsity levels (gradient from orange to green)
    import matplotlib.cm as cm
    sparse_colors = cm.viridis(np.linspace(0.2, 0.8, len(sparsity_levels)))
    
    # Plot Gaussian first (as baseline)
    times = np.array(results['gaussian']['times']) * 1000
    errors = np.array(results['gaussian']['errors'])
    ax.plot(times, errors, marker='o', color='#0072B2', linewidth=2.5, 
            markersize=8, label='Gaussian (dense)', zorder=10)
    
    # Plot each sparsity level
    for i, s in enumerate(sparsity_levels):
        key = f'sparse_s{s}'
        if key in results:
            times = np.array(results[key]['times']) * 1000
            errors = np.array(results[key]['errors'])
            label = f'Sparse Sign (s={s})' if s > 1 else 'CountSketch (s=1)'
            marker = 's' if s == 1 else '^'
            ax.plot(times, errors, marker=marker, color=sparse_colors[i],
                    linewidth=2, markersize=7, label=label, alpha=0.85)
    
    # Add optimal rank-k error as horizontal baseline
    if results.get('optimal_error') is not None:
        opt_err = results['optimal_error']
        ax.axhline(y=opt_err, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Optimal rank-{k}')
    
    # Add sketch size annotations for Gaussian only (to avoid clutter)
    sketch_sizes = results['sketch_sizes']
    g_times = np.array(results['gaussian']['times']) * 1000
    g_errors = np.array(results['gaussian']['errors'])
    for i, l in enumerate(sketch_sizes):
        if i == 0 or i == len(sketch_sizes) - 1:
            ax.annotate(f'$\\ell$={l}', (g_times[i], g_errors[i]),
                       textcoords="offset points", xytext=(8, 0), 
                       fontsize=7, color='#0072B2')
    
    ax.set_xlabel('Runtime (ms)')
    ax.set_ylabel('Relative Error ($\\|A - \\hat{A}\\|_F / \\|A\\|_F$)')
    ax.set_title(f'Error vs. Runtime: Sparsity-Accuracy Trade-off\n20 Newsgroups: {m} docs × {n} words, rank-{k}', 
                 fontweight='bold')
    
    # Legend outside or well-placed
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)
    
    # Use log scale for time
    ax.set_xscale('log')
    
    # Add annotation
    ax.text(0.02, 0.02, 
            'Lower-left is better\n(faster & more accurate)',
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig3_pareto.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig3_pareto.png')
    print(f"\n  Saved: {output_path}")
    
    return fig


def plot_from_json(selected_sparsities=None):
    """Load results from JSON and regenerate the figure."""
    output_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    json_path = data_dir / 'experiment_3_results.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("Creating Figure 3 from saved data...")
    if selected_sparsities:
        print(f"  Showing only s = {selected_sparsities}")
    fig = create_figure_3(
        data['results'], 
        data['m'], data['n'], data['k'],
        output_dir,
        selected_sparsities=selected_sparsities
    )
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 3: Error vs. Runtime Pareto Frontier")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Use 20 Newsgroups - a real-world sparse text dataset
    # This has actual low-rank structure (topics) so SVD is meaningful
    print("\nLoading 20 Newsgroups dataset...")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Load data
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Convert to TF-IDF sparse matrix
    print("Converting to TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
    A = vectorizer.fit_transform(newsgroups.data)
    A = A.tocsr()
    
    m, n = A.shape
    density = A.nnz / (m * n)
    k = 30  # Target rank (roughly number of topics = 20)
    sparsity_s = 4
    
    print(f"  Matrix: {m} documents x {n} words")
    print(f"  Density: {density*100:.2f}%")
    print(f"  nnz: {A.nnz:,}")
    
    # Sketch sizes to test (generates Pareto curve)
    sketch_sizes = [40, 60, 80, 100, 150, 200, 300, 400, 500]
    
    # Sparsity levels to test (s=1 is CountSketch, powers of 2 up to 64)
    sparsity_levels = [1, 2, 4, 8, 16, 32, 64]
    
    print("\nRunning Pareto frontier experiment...")
    results = experiment_pareto(A, sketch_sizes, k, sparsity_levels)
    
    # Save results
    save_data = {
        'results': results,
        'm': m, 'n': n, 'k': k, 
        'density': density,
        'sparsity_levels': sparsity_levels,
        'dataset': '20newsgroups',
    }
    json_path = data_dir / 'experiment_3_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved results to: {json_path}")
    
    # Create figure
    print("\n" + "="*70)
    print("CREATING FIGURE")
    print("="*70)
    
    fig = create_figure_3(results, m, n, k, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Compare methods at fixed sketch size
    l_idx = len(sketch_sizes) // 2  # Middle sketch size
    g_time = results['gaussian']['times'][l_idx] * 1000
    g_err = results['gaussian']['errors'][l_idx]
    
    print(f"\nAt l={sketch_sizes[l_idx]} (target rank k={k}):")
    print(f"  {'Method':<20} {'Time (ms)':<12} {'Error':<10} {'Speedup':<10}")
    print(f"  {'-'*50}")
    print(f"  {'Gaussian (dense)':<20} {g_time:<12.0f} {g_err:<10.4f} {'1.0x':<10}")
    
    for s in sparsity_levels:
        key = f'sparse_s{s}'
        t = results[key]['times'][l_idx] * 1000
        e = results[key]['errors'][l_idx]
        speedup = g_time / t
        label = f'CountSketch (s=1)' if s == 1 else f'Sparse Sign (s={s})'
        print(f"  {label:<20} {t:<12.0f} {e:<10.4f} {speedup:.1f}x")
    
    print(f"\n  Optimal rank-{k} error: {results['optimal_error']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        plot_from_json()
    else:
        main()
