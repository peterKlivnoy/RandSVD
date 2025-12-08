"""
Experiment 4: Block Krylov vs Simultaneous Iteration - Runtime Comparison

This experiment compares:
- Simultaneous Iteration (standard power method)
- Block Krylov Iteration (keeps full Krylov subspace)

Using three error metrics from Musco & Musco (2015):
1. Frobenius Error (weak): ||A - ZZ^T A||_F / ||A - A_k||_F - 1
2. Spectral Error (strong): ||A - ZZ^T A||_2 / ||A - A_k||_2 - 1  
3. Per-Vector Error (strongest): max_i |σ_i² - ||A^T z_i||²| / σ_{k+1}²

Figures produced:
  - fig4_runtime_vs_error.pdf: Error vs Runtime on synthetic data
  - fig4_runtime_newsgroups.pdf: Error vs Runtime on 20 Newsgroups
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    simultaneous_iteration,
    block_krylov_iteration,
    compute_all_errors,
    create_slow_decay_matrix,
    load_20newsgroups,
)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors matching reference style
GREEN = '#2ca02c'  # Block Krylov
BLUE = '#1f77b4'   # Simultaneous Iteration


def run_iteration_timed(A, k, q, method='simultaneous', seed=0):
    """
    Run an iteration method and time the FULL algorithm.
    
    Times everything: random Ω, A@Ω, iterations, QR, truncation.
    Does NOT time error computation.
    
    Args:
        A: Input matrix
        k: Target rank
        q: Number of iterations
        method: 'simultaneous' or 'krylov'
        seed: Random seed
    
    Returns:
        Z: Orthonormal basis (m × k)
        elapsed: Wall-clock time in seconds
    """
    t0 = time.perf_counter()
    
    if method == 'simultaneous':
        Z = simultaneous_iteration(A, k, q, seed=seed)
    elif method == 'krylov':
        Z = block_krylov_iteration(A, k, q, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time.perf_counter() - t0
    return Z, elapsed


def run_experiment(A, true_sv, k, q_values, num_trials=3, verbose=True):
    """
    Run full experiment comparing Simultaneous Iteration vs Block Krylov.
    
    Args:
        A: Input matrix (dense or sparse)
        true_sv: True singular values (at least k+1 values)
        k: Target rank
        q_values: List of iteration counts to test
        num_trials: Number of trials per configuration
        verbose: Print progress
    
    Returns:
        results: Dict with times and errors for each method
    """
    results = {
        'q_values': q_values,
        'simult': {'times': [], 'frob': [], 'spec': [], 'pervec': []},
        'krylov': {'times': [], 'frob': [], 'spec': [], 'pervec': []},
    }
    
    for q in q_values:
        if verbose:
            print(f"  q={q:2d}", end=" ", flush=True)
        
        # Simultaneous iteration
        times_s, frob_s, spec_s, pervec_s = [], [], [], []
        for trial in range(num_trials):
            Z, t = run_iteration_timed(A, k, q, method='simultaneous', seed=trial)
            times_s.append(t)
            f, s, p = compute_all_errors(A, Z, true_sv, k)
            frob_s.append(f)
            spec_s.append(s)
            pervec_s.append(p)
        
        results['simult']['times'].append(np.median(times_s))
        results['simult']['frob'].append(np.median(frob_s))
        results['simult']['spec'].append(np.median(spec_s))
        results['simult']['pervec'].append(np.median(pervec_s))
        
        # Block Krylov
        times_k, frob_k, spec_k, pervec_k = [], [], [], []
        for trial in range(num_trials):
            Z, t = run_iteration_timed(A, k, q, method='krylov', seed=trial)
            times_k.append(t)
            f, s, p = compute_all_errors(A, Z, true_sv, k)
            frob_k.append(f)
            spec_k.append(s)
            pervec_k.append(p)
        
        results['krylov']['times'].append(np.median(times_k))
        results['krylov']['frob'].append(np.median(frob_k))
        results['krylov']['spec'].append(np.median(spec_k))
        results['krylov']['pervec'].append(np.median(pervec_k))
        
        if verbose:
            print()
    
    return results


def create_figure(results, title, output_path):
    """
    Create publication-quality figure with all 6 lines.
    
    Args:
        results: Dict from run_experiment()
        title: Plot title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    times_s = np.array(results['simult']['times']) * 1000  # Convert to ms
    times_k = np.array(results['krylov']['times']) * 1000
    
    # Block Krylov - green with different markers
    ax.plot(times_k, results['krylov']['frob'], '-+', color=GREEN, linewidth=2, markersize=8,
            label='Block Krylov - Frobenius Error')
    ax.plot(times_k, results['krylov']['spec'], '-o', color=GREEN, linewidth=2, markersize=6,
            label='Block Krylov - Spectral Error')
    ax.plot(times_k, results['krylov']['pervec'], '-^', color=GREEN, linewidth=2, markersize=6,
            label='Block Krylov - Per Vector Error')
    
    # Simultaneous Iteration - blue with different markers
    ax.plot(times_s, results['simult']['frob'], '-+', color=BLUE, linewidth=2, markersize=8,
            label='Simult. Iter. - Frobenius Error')
    ax.plot(times_s, results['simult']['spec'], '-o', color=BLUE, linewidth=2, markersize=6,
            label='Simult. Iter. - Spectral Error')
    ax.plot(times_s, results['simult']['pervec'], '-^', color=BLUE, linewidth=2, markersize=6,
            label='Simult. Iter. - Per Vector Error')
    
    ax.set_xlabel('Runtime (ms)')
    ax.set_ylabel('Error ε')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 0.35)
    # Cap x-axis at the smaller of the two final times
    x_max = min(times_s[-1], times_k[-1])
    ax.set_xlim(0, x_max * 1.02)  # Small margin
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"  Saved: {output_path}")
    
    return fig


def create_figure_iterations(results, title, output_path):
    """
    Create figure with Error vs Iterations (q).
    
    Args:
        results: Dict from run_experiment()
        title: Plot title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    q_values = results['q_values']
    
    # Block Krylov - green with different markers
    ax.plot(q_values, results['krylov']['frob'], '-+', color=GREEN, linewidth=2, markersize=8,
            label='Block Krylov - Frobenius Error')
    ax.plot(q_values, results['krylov']['spec'], '-o', color=GREEN, linewidth=2, markersize=6,
            label='Block Krylov - Spectral Error')
    ax.plot(q_values, results['krylov']['pervec'], '-^', color=GREEN, linewidth=2, markersize=6,
            label='Block Krylov - Per Vector Error')
    
    # Simultaneous Iteration - blue with different markers
    ax.plot(q_values, results['simult']['frob'], '-+', color=BLUE, linewidth=2, markersize=8,
            label='Simult. Iter. - Frobenius Error')
    ax.plot(q_values, results['simult']['spec'], '-o', color=BLUE, linewidth=2, markersize=6,
            label='Simult. Iter. - Spectral Error')
    ax.plot(q_values, results['simult']['pervec'], '-^', color=BLUE, linewidth=2, markersize=6,
            label='Simult. Iter. - Per Vector Error')
    
    ax.set_xlabel('Iterations q')
    ax.set_ylabel('Error ε')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 0.35)
    ax.set_xlim(0, max(q_values))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"  Saved: {output_path}")
    
    return fig


def experiment_synthetic(m=5000, n=10000, k=20, decay_rate=0.5, q_max=25):
    """
    Run experiment on synthetic matrix with slow spectral decay.
    """
    print("="*70)
    print(f"EXPERIMENT: Synthetic Matrix ({m}×{n})")
    print("="*70)
    
    print(f"\nCreating matrix with decay_rate={decay_rate}...")
    A, true_sv = create_slow_decay_matrix(m, n, decay_rate=decay_rate)
    
    gap = true_sv[k-1] / true_sv[k]
    print(f"  Spectral gap: σ_k/σ_(k+1) = {gap:.4f}")
    
    q_values = list(range(0, q_max + 1))
    
    print(f"\nRunning experiments (q=0 to {q_max})...")
    results = run_experiment(A, true_sv, k, q_values, num_trials=3)
    
    # Add metadata
    results['m'] = m
    results['n'] = n
    results['k'] = k
    results['decay_rate'] = decay_rate
    results['spectral_gap'] = gap
    
    return results


def experiment_newsgroups(k=30, q_max=25):
    """
    Run experiment on 20 Newsgroups TF-IDF matrix.
    """
    from scipy.sparse.linalg import svds
    import scipy.sparse as sp
    
    print("="*70)
    print("EXPERIMENT: 20 Newsgroups Dataset")
    print("="*70)
    
    print("\nLoading dataset...")
    A = load_20newsgroups(max_features=15000)
    m, n = A.shape
    print(f"  Matrix: {m}×{n}, nnz={A.nnz:,}, density={A.nnz/(m*n):.4f}")
    
    print(f"\nComputing reference SVD (top {k+10} singular values)...")
    _, true_sv, _ = svds(A, k=k+10, which='LM')
    true_sv = true_sv[::-1]  # svds returns ascending order
    
    gap = true_sv[k-1] / true_sv[k]
    print(f"  Spectral gap: σ_k/σ_(k+1) = {gap:.4f}")
    
    q_values = list(range(0, q_max + 1))
    
    print(f"\nRunning experiments (q=0 to {q_max})...")
    results = run_experiment(A, true_sv, k, q_values, num_trials=3)
    
    # Add metadata
    results['m'] = m
    results['n'] = n
    results['k'] = k
    results['nnz'] = A.nnz
    results['spectral_gap'] = gap
    
    return results


def main():
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Experiment 1: Synthetic matrix
    results_synth = experiment_synthetic(m=5000, n=10000, k=20, decay_rate=0.5, q_max=25)
    
    title_synth = f"Synthetic ({results_synth['m']}×{results_synth['n']}), k={results_synth['k']}"
    create_figure(results_synth, title_synth, output_dir / 'fig4_runtime_vs_error.pdf')
    
    # Save results
    with open(data_dir / 'experiment_4_synthetic_results.json', 'w') as f:
        json.dump(results_synth, f, indent=2)
    
    # Experiment 2: 20 Newsgroups
    results_news = experiment_newsgroups(k=30, q_max=25)
    
    title_news = f"20 Newsgroups ({results_news['m']}×{results_news['n']}), k={results_news['k']}"
    create_figure(results_news, title_news, output_dir / 'fig4_runtime_newsgroups.pdf')
    
    # Save results
    with open(data_dir / 'experiment_4_newsgroups_results.json', 'w') as f:
        json.dump(results_news, f, indent=2)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


def plot_from_json(dataset='synthetic', plot_type='runtime'):
    """
    Regenerate figures from saved JSON data.
    
    Args:
        dataset: 'synthetic' or 'newsgroups'
        plot_type: 'runtime' (error vs time) or 'iterations' (error vs q)
    """
    output_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    if dataset == 'synthetic':
        json_path = data_dir / 'experiment_4_synthetic_results.json'
        if plot_type == 'runtime':
            output_path = output_dir / 'fig4_runtime_vs_error.pdf'
        else:
            output_path = output_dir / 'fig4_iterations_synthetic.pdf'
    else:
        json_path = data_dir / 'experiment_4_newsgroups_results.json'
        if plot_type == 'runtime':
            output_path = output_dir / 'fig4_runtime_newsgroups.pdf'
        else:
            output_path = output_dir / 'fig4_iterations_newsgroups.pdf'
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    title = f"{dataset.replace('newsgroups', '20 Newsgroups').replace('synthetic', 'Synthetic')} ({results['m']}×{results['n']}), k={results['k']}"
    
    if plot_type == 'runtime':
        create_figure(results, title, output_path)
    else:
        create_figure_iterations(results, title, output_path)


if __name__ == "__main__":
    main()
