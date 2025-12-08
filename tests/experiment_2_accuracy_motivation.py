"""
Experiment 2: RandSVD Accuracy - Motivating Power Iterations

Demonstrates that basic RandSVD (q=0) accuracy depends heavily on singular value decay.
This motivates the need for power iterations in harder cases.

Three test matrices:
1. Exponential  - σ_i = exp(-0.1i)   → Fast decay, easy
2. Polynomial   - σ_i = 1/i          → Moderate decay  
3. Slow         - σ_i = 1/sqrt(i)    → Slow decay, hard

Figure produced:
  - fig2_accuracy_motivation.pdf: Bar chart showing error ratio for three decay types
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.randsvd_algorithm import randSVD

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
    'optimal': '#000000',
}


def create_test_matrix(n, decay_type='exponential', rank=None):
    """Create test matrix with specified spectral decay."""
    if rank is None:
        rank = n
    
    # Create singular values with specified decay
    # Key: smaller spectral gap at rank k → harder for RandSVD
    if decay_type == 'exponential':
        # Fast decay: σ_i = 0.9^i - easy for RandSVD
        # Gap: σ_k/σ_{k+1} = 1/0.9 ≈ 1.11 (constant multiplicative gap)
        sigma = 0.9 ** np.arange(rank)
    elif decay_type == 'polynomial':
        # Polynomial decay: σ_i = 1/(1+i) - moderate
        # Gap: σ_k/σ_{k+1} = (k+2)/(k+1) → 1 as k grows
        sigma = 1.0 / (1 + np.arange(rank))
    elif decay_type == 'slow':
        # Slow decay: σ_i = 1/sqrt(1+i) - harder for RandSVD  
        # Gap: σ_k/σ_{k+1} = sqrt((k+2)/(k+1)) → 1 faster
        sigma = 1.0 / np.sqrt(1 + np.arange(rank))
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    
    # Create matrix A = U @ diag(sigma) @ V^T
    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(n, rank))
    V, _ = np.linalg.qr(np.random.randn(n, rank))
    A = U @ np.diag(sigma) @ V.T
    
    return A, sigma


def compute_optimal_error(sigma, k):
    """Compute optimal rank-k approximation error (Eckart-Young)."""
    tail = sigma[k:]
    return np.sqrt(np.sum(tail**2))


def benchmark_accuracy_vs_spectrum(A_dict, sigma_dict, k, p, q, num_trials=5):
    """Benchmark Gaussian accuracy across spectrum types."""
    results = {}
    optimal_errors = {}
    
    for decay_type in A_dict.keys():
        print(f"  {decay_type}: ", end="", flush=True)
        A = A_dict[decay_type]
        sigma = sigma_dict[decay_type]
        optimal_error = compute_optimal_error(sigma, k)
        optimal_errors[decay_type] = optimal_error
        
        errors = []
        for trial in range(num_trials):
            np.random.seed(trial)
            U, S, Vt = randSVD(A, k, p=p, q=q, sketch_type='gaussian')
            A_approx = U @ np.diag(S) @ Vt
            error = np.linalg.norm(A - A_approx, 'fro')
            errors.append(error)
        
        median_error = np.median(errors)
        ratio = median_error / optimal_error
        results[decay_type] = ratio
        print(f"ratio={ratio:.3f}")
    
    return results, optimal_errors


def create_figure(results, decay_types, k, p, q, output_dir):
    """Create Figure 2: Accuracy motivation bar chart."""
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    x = np.arange(len(decay_types))
    width = 0.5
    
    # Get errors for each decay type
    errors = [results[d] for d in decay_types]
    
    # Create bars
    bars = ax.bar(x, errors, width, color=COLORS['gaussian'], 
                  edgecolor='black', linewidth=1.5, label='Gaussian RandSVD')
    
    # Optimal line at 1.0
    ax.axhline(y=1, color='black', linestyle='-', linewidth=2, label='Optimal (truncated SVD)')
    
    ax.set_xlabel('Spectral Decay Type')
    ax.set_ylabel(r'Error Ratio ($\|A - \tilde{A}_k\| / \|A - A_k^*\|$)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in decay_types])
    ax.set_title(f'RandSVD Accuracy by Spectrum Type\n$k={k}$, $p={p}$, $q={q}$ (no power iterations)', 
                 fontweight='bold')
    ax.legend(loc='upper left')
    
    # Y-axis: start at 1, end at max + margin
    max_error = max(errors)
    margin = (max_error - 1) * 0.15  # 15% margin above highest bar
    ax.set_ylim(1.0, max_error + margin)
    
    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.annotate(f'{err:.2f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig2_accuracy_motivation.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig2_accuracy_motivation.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def main():
    print("=" * 70)
    print("EXPERIMENT 2: RandSVD Accuracy - Motivating Power Iterations")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Parameters
    n = 1024
    k = 50
    p = 20
    q = 0  # No power iterations - this is the key!
    
    # Create test matrices
    print("\n[1/2] Creating test matrices...")
    decay_types = ['exponential', 'polynomial', 'slow']
    A_dict = {}
    sigma_dict = {}
    for decay_type in decay_types:
        A, sigma = create_test_matrix(n, decay_type)
        A_dict[decay_type] = A
        sigma_dict[decay_type] = sigma
        print(f"  {decay_type}: n={n}, σ_1={sigma[0]:.3f}, σ_k={sigma[k]:.3e}, σ_n={sigma[-1]:.3e}")
    
    # Benchmark accuracy
    print("\n[2/2] Benchmarking Gaussian RandSVD (q=0)...")
    results, optimal_errors = benchmark_accuracy_vs_spectrum(
        A_dict, sigma_dict, k, p=p, q=q
    )
    
    # Save results
    save_data = {
        'results': results,
        'n': n, 'k': k, 'p': p, 'q': q,
        'decay_types': decay_types,
    }
    json_path = data_dir / 'experiment_2_accuracy_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✓ Saved results to: {json_path}")
    
    # Create figure
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)
    fig = create_figure(results, decay_types, k, p, q, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nRandSVD with q=0 (no power iterations), k={k}, p={p}:")
    for decay_type in decay_types:
        print(f"  {decay_type.capitalize():12s}: {results[decay_type]:.2f}× optimal")
    print("\nKey finding: Slow-decay matrices produce significantly larger errors")
    print("without power iterations → motivates using q > 0 for hard problems.")


if __name__ == "__main__":
    main()
