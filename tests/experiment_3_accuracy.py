"""
Experiment 3: Accuracy Analysis

This experiment studies approximation accuracy:
  - All sketching methods achieve similar accuracy (theory)
  - Power iterations (q) dramatically improve accuracy for slow decay
  - Oversampling (p) provides additional accuracy buffer

Figures produced:
  - Fig 3: Accuracy comparison across spectral decay types
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.randsvd_algorithm import randSVD

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
    'srft': '#D55E00',
    'srht': '#009E73',
    'optimal': '#000000',
}


def create_test_matrix(n, decay_type='exponential', rank=None):
    """Create test matrix with specified spectral decay."""
    if rank is None:
        rank = n
    
    # Create singular values with specified decay
    if decay_type == 'exponential':
        # Fast decay: σ_i = exp(-0.1 * i)
        sigma = np.exp(-0.1 * np.arange(rank))
    elif decay_type == 'polynomial':
        # Moderate decay: σ_i = 1 / i
        sigma = 1.0 / (1 + np.arange(rank))
    elif decay_type == 'slow':
        # Slow decay: σ_i = 1 / sqrt(i)
        sigma = 1.0 / np.sqrt(1 + np.arange(rank))
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    
    # Create matrix A = U @ diag(sigma) @ V^T
    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(n, rank))
    V, _ = np.linalg.qr(np.random.randn(n, rank))
    A = U @ np.diag(sigma) @ V.T
    
    return A, sigma


def compute_optimal_error(sigma, k, norm='fro'):
    """Compute optimal rank-k approximation error (Eckart-Young)."""
    tail = sigma[k:]
    if norm == 'fro':
        return np.sqrt(np.sum(tail**2))
    else:  # spectral
        return tail[0] if len(tail) > 0 else 0


def benchmark_accuracy_vs_oversampling(A, sigma, k, p_values, methods, num_trials=5):
    """Benchmark accuracy across oversampling values."""
    results = {m: [] for m in methods}
    
    optimal_error = compute_optimal_error(sigma, k)
    
    for p in p_values:
        print(f"  p={p}: ", end="", flush=True)
        
        for method in methods:
            errors = []
            for trial in range(num_trials):
                np.random.seed(trial)
                U, S, Vt = randSVD(A, k, p=p, q=1, sketch_type=method)
                A_approx = U @ np.diag(S) @ Vt
                error = np.linalg.norm(A - A_approx, 'fro')
                errors.append(error)
            
            median_error = np.median(errors)
            results[method].append(median_error)
            ratio = median_error / optimal_error
            print(f"{method}={ratio:.3f} ", end="")
        print()
    
    return results, optimal_error


def benchmark_accuracy_vs_power_iterations(A, sigma, k, p, q_values, methods, num_trials=5):
    """Benchmark accuracy across power iteration counts."""
    results = {m: [] for m in methods}
    
    optimal_error = compute_optimal_error(sigma, k)
    
    for q in q_values:
        print(f"  q={q}: ", end="", flush=True)
        
        for method in methods:
            errors = []
            for trial in range(num_trials):
                np.random.seed(trial)
                U, S, Vt = randSVD(A, k, p=p, q=q, sketch_type=method)
                A_approx = U @ np.diag(S) @ Vt
                error = np.linalg.norm(A - A_approx, 'fro')
                errors.append(error)
            
            median_error = np.median(errors)
            results[method].append(median_error)
            ratio = median_error / optimal_error
            print(f"{method}={ratio:.3f} ", end="")
        print()
    
    return results, optimal_error


def benchmark_accuracy_vs_spectrum(A_dict, sigma_dict, k, p, q, methods, num_trials=5):
    """Benchmark accuracy across spectrum types."""
    results = {decay: {m: 0 for m in methods} for decay in A_dict.keys()}
    optimal_errors = {}
    
    for decay_type in A_dict.keys():
        print(f"  {decay_type}: ", end="", flush=True)
        A = A_dict[decay_type]
        sigma = sigma_dict[decay_type]
        optimal_error = compute_optimal_error(sigma, k)
        optimal_errors[decay_type] = optimal_error
        
        for method in methods:
            errors = []
            for trial in range(num_trials):
                np.random.seed(trial)
                U, S, Vt = randSVD(A, k, p=p, q=q, sketch_type=method)
                A_approx = U @ np.diag(S) @ Vt
                error = np.linalg.norm(A - A_approx, 'fro')
                errors.append(error)
            
            median_error = np.median(errors)
            results[decay_type][method] = median_error / optimal_error
            print(f"{method}={median_error/optimal_error:.3f} ", end="")
        print()
    
    return results, optimal_errors


def create_figure_3(results_spectrum, results_power, results_oversample,
                    decay_types, q_values, p_values,
                    optimal_power, optimal_oversample, output_dir):
    """Create Figure 3: Accuracy analysis."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    methods = ['gaussian', 'srft', 'srht']
    
    # ===== Panel A: Error by spectrum type =====
    ax = axes[0]
    
    x = np.arange(len(decay_types))
    width = 0.25
    
    for i, method in enumerate(methods):
        errors = [results_spectrum[d][method] for d in decay_types]
        ax.bar(x + (i - 1) * width, errors, width, 
               color=COLORS[method], label=method.upper(), edgecolor='black')
    
    ax.axhline(y=1, color='black', linestyle='-', linewidth=2, label='Optimal')
    ax.set_xlabel('Spectral Decay Type')
    ax.set_ylabel('Error Ratio ($\\|A - \\tilde{A}_k\\| / \\|A - A_k^*\\|$)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in decay_types])
    ax.set_title('(a) Error by Spectrum Type\n$k=50$, $p=20$, $q=1$', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_ylim([0.9, 1.5])
    
    # ===== Panel B: Error vs Power Iterations =====
    ax = axes[1]
    
    for method in methods:
        errors = np.array(results_power[method])
        ratio = errors / optimal_power
        ax.plot(q_values, ratio, 'o-', color=COLORS[method],
                linewidth=2, markersize=8, label=method.upper())
    
    ax.axhline(y=1, color='black', linestyle='-', linewidth=2, label='Optimal')
    ax.set_xlabel('Power Iterations ($q$)')
    ax.set_ylabel('Error Ratio')
    ax.set_title('(b) Effect of Power Iterations\nSlow decay, $k=50$, $p=20$', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xticks(q_values)
    
    # ===== Panel C: Error vs Oversampling =====
    ax = axes[2]
    
    for method in methods:
        errors = np.array(results_oversample[method])
        ratio = errors / optimal_oversample
        ax.plot(p_values, ratio, 'o-', color=COLORS[method],
                linewidth=2, markersize=8, label=method.upper())
    
    ax.axhline(y=1, color='black', linestyle='-', linewidth=2, label='Optimal')
    ax.set_xlabel('Oversampling ($p$)')
    ax.set_ylabel('Error Ratio')
    ax.set_title('(c) Effect of Oversampling\nPolynomial decay, $k=50$, $q=1$', fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = output_dir / 'fig3_accuracy.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fig3_accuracy.png')
    print(f"\n✓ Saved: {output_path}")
    
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 3: Accuracy Analysis")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    n = 1024
    k = 50
    methods = ['gaussian', 'srft', 'srht']
    
    # ===== Create test matrices =====
    print("\n[1/4] Creating test matrices...")
    decay_types = ['exponential', 'polynomial', 'slow']
    A_dict = {}
    sigma_dict = {}
    for decay_type in decay_types:
        A, sigma = create_test_matrix(n, decay_type)
        A_dict[decay_type] = A
        sigma_dict[decay_type] = sigma
    
    # ===== Benchmark 1: Error by spectrum type =====
    print("\n[2/4] Benchmarking error by spectrum type...")
    results_spectrum, _ = benchmark_accuracy_vs_spectrum(
        A_dict, sigma_dict, k, p=20, q=1, methods=methods
    )
    
    # ===== Benchmark 2: Error vs Power Iterations =====
    print("\n[3/4] Benchmarking error vs power iterations...")
    A, sigma = A_dict['slow'], sigma_dict['slow']
    q_values = [0, 1, 2, 3, 4]
    results_power, optimal_power = benchmark_accuracy_vs_power_iterations(
        A, sigma, k, p=20, q_values=q_values, methods=methods
    )
    
    # ===== Benchmark 3: Error vs Oversampling =====
    print("\n[4/4] Benchmarking error vs oversampling...")
    A, sigma = A_dict['polynomial'], sigma_dict['polynomial']
    p_values = [5, 10, 15, 20, 30, 50]
    results_oversample, optimal_oversample = benchmark_accuracy_vs_oversampling(
        A, sigma, k, p_values=p_values, methods=methods
    )
    
    # Create figure
    print("\n[5/5] Creating Figure 3...")
    fig = create_figure_3(
        results_spectrum, results_power, results_oversample,
        decay_types, q_values, p_values,
        optimal_power, optimal_oversample, output_dir
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. ALL METHODS ACHIEVE SIMILAR ACCURACY — Gaussian, SRFT, and SRHT
   produce nearly identical approximation errors for the same parameters.

2. SPECTRAL DECAY MATTERS — Fast decay (exponential) is easy;
   slow decay requires power iterations or larger oversampling.

3. POWER ITERATIONS (q) — Most effective for slow-decaying spectra.
   - q=0: May have large error for slow decay
   - q=1-2: Usually sufficient for practical problems
   - q>2: Diminishing returns

4. OVERSAMPLING (p) — Provides accuracy buffer.
   - p=5-10: Good default choice
   - Larger p helps for slow decay but increases computation

5. PRACTICAL GUIDELINE: Choose sketching method for speed,
   then tune q and p for accuracy based on spectral decay.
""")


if __name__ == "__main__":
    main()
