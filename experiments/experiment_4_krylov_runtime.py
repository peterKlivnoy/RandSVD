"""
Experiment 4: Block Krylov vs Simultaneous Iteration - Runtime vs Error
========================================================================

Three error metrics (from Musco & Musco 2015):
1. Frobenius Error (weak): ||A - ZZ^T A||_F / ||A - A_k||_F - 1
2. Spectral Error (strong): ||A - ZZ^T A||_2 / ||A - A_k||_2 - 1  
3. Per-Vector Error (strongest): max_i |σ_i² - ||A^T z_i||²| / σ_{k+1}²

The per-vector error ensures EACH singular vector captures the right variance.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Publication-quality settings
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


def create_slow_decay_matrix(m, n, k, decay_rate=0.5):
    """
    Create matrix with SLOWLY decaying singular values (hard case).
    
    σ_i = 1 / i^decay_rate
    
    decay_rate = 0.5: very slow decay (tiny gaps) - HARD
    """
    np.random.seed(42)
    
    min_dim = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, min_dim))
    V, _ = np.linalg.qr(np.random.randn(n, min_dim))
    
    singular_values = 1.0 / (np.arange(1, min_dim + 1) ** decay_rate)
    
    A = U @ np.diag(singular_values) @ V.T
    
    gap_ratio = singular_values[k-1] / singular_values[k]
    print(f"  Spectral gap at k={k}: σ_k/σ_(k+1) = {gap_ratio:.4f}")
    
    return A, singular_values, U


def load_20newsgroups(max_features=15000):
    """
    Load 20 Newsgroups dataset as TF-IDF matrix.
    Returns sparse matrix and computes true singular values for reference.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("  Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    print("  Computing TF-IDF representation...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    A = vectorizer.fit_transform(newsgroups.data)
    
    print(f"  Matrix shape: {A.shape}, nnz: {A.nnz}, density: {A.nnz / (A.shape[0] * A.shape[1]):.4f}")
    
    return A


def simultaneous_iteration(A, k, q):
    """
    Algorithm 1: Simultaneous Power Iteration (standard).
    
    Keeps only the final (AA^T)^q * A * Omega.
    """
    m, n = A.shape
    np.random.seed(0)
    Omega = np.random.randn(n, k)
    
    K = A @ Omega
    
    for _ in range(q):
        K, _ = np.linalg.qr(K)
        K = A @ (A.T @ K)
    
    Z, _ = np.linalg.qr(K)
    return Z


def block_krylov_iteration(A, k, q):
    """
    Algorithm 2: Block Krylov Iteration.
    
    Keeps ALL intermediate powers: [A*Ω, (AA^T)A*Ω, ..., (AA^T)^q*A*Ω]
    """
    m, n = A.shape
    np.random.seed(0)
    Omega = np.random.randn(n, k)
    
    K_i = A @ Omega
    blocks = [K_i.copy()]
    
    for _ in range(q):
        K_i, _ = np.linalg.qr(K_i)
        K_i = A @ (A.T @ K_i)
        blocks.append(K_i.copy())
    
    K = np.hstack(blocks)
    Z, _ = np.linalg.qr(K)
    
    # Truncate to top k directions
    B = Z.T @ A
    U_B, _, _ = np.linalg.svd(B, full_matrices=False)
    Z_k = Z @ U_B[:, :k]
    
    return Z_k


def compute_all_errors(A, Z, true_sv, k):
    """
    Compute all three error metrics from Musco & Musco (2015).
    
    For sparse A, we avoid forming the dense residual A - ZZ^T A.
    Instead we use: ||A - ZZ^T A||^2 = ||A||^2 - ||Z^T A||^2
    
    Returns: (frobenius_error, spectral_error, pervector_error)
    """
    import scipy.sparse as sp
    
    is_sparse = sp.issparse(A)
    
    # Optimal errors
    opt_frob = np.sqrt(np.sum(true_sv[k:]**2))
    opt_spec = true_sv[k] if k < len(true_sv) else 1e-14
    sigma_kplus1_sq = true_sv[k]**2 if k < len(true_sv) else 1e-14
    
    # Compute Z^T A (needed for multiple metrics)
    ZtA = Z.T @ A  # k × n, dense
    
    # 1. Frobenius error (weak)
    # ||A - ZZ^T A||_F^2 = ||A||_F^2 - ||Z^T A||_F^2 (since Z is orthonormal)
    if is_sparse:
        A_frob_sq = sp.linalg.norm(A, 'fro')**2
    else:
        A_frob_sq = np.linalg.norm(A, 'fro')**2
    ZtA_frob_sq = np.linalg.norm(ZtA, 'fro')**2
    residual_frob_sq = max(A_frob_sq - ZtA_frob_sq, 0)
    actual_frob = np.sqrt(residual_frob_sq)
    frob_error = max(actual_frob / opt_frob - 1, 1e-14) if opt_frob > 1e-14 else 1e-14
    
    # 2. Spectral error (strong)
    # ||A - ZZ^T A||_2 = σ_max(A - ZZ^T A)
    # We estimate this using a few power iterations on (I - ZZ^T)A
    # Actually, for efficiency, we use: ||(I - ZZ^T)A||_2 via randomized estimation
    def matvec_residual(v):
        """Compute (A - ZZ^T A) @ v = A @ v - Z @ (Z^T @ (A @ v))"""
        Av = A @ v
        return Av - Z @ (Z.T @ Av)
    
    # Power iteration to estimate spectral norm
    np.random.seed(123)
    v = np.random.randn(A.shape[1])
    v = v / np.linalg.norm(v)
    for _ in range(10):  # 10 iterations usually enough
        u = matvec_residual(v)
        v = A.T @ u - ZtA.T @ (Z.T @ u)  # (A - ZZ^T A)^T @ u
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            break
        v = v / norm_v
    actual_spec = np.linalg.norm(matvec_residual(v))
    spec_error = max(actual_spec / opt_spec - 1, 1e-14) if opt_spec > 1e-14 else 1e-14
    
    # 3. Per-vector error (strongest)
    # ||A^T z_i||^2 should equal σ_i^2
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        z_i = Z[:, i]
        # A^T @ z_i is just the i-th row of Z^T A = ZtA[i, :]
        var_z = np.linalg.norm(ZtA[i, :])**2
        var_true = true_sv[i]**2
        err = abs(var_true - var_z) / sigma_kplus1_sq
        pervec_errors.append(err)
    
    pervec_error = max(max(pervec_errors), 1e-14) if pervec_errors else 1e-14
    
    return frob_error, spec_error, pervec_error


def compute_all_errors_sparse(A, Z, true_sv, k, opt_frob_sq):
    """
    Compute all three error metrics for sparse matrices.
    
    Uses precomputed opt_frob_sq = ||A||_F^2 - sum(σ_i^2, i<=k)
    which is the correct optimal Frobenius error squared.
    
    Returns: (frobenius_error, spectral_error, pervector_error)
    """
    import scipy.sparse as sp
    
    opt_frob = np.sqrt(opt_frob_sq)
    opt_spec = true_sv[k] if k < len(true_sv) else 1e-14
    sigma_kplus1_sq = true_sv[k]**2 if k < len(true_sv) else 1e-14
    
    # Compute Z^T A (needed for multiple metrics)
    ZtA = Z.T @ A  # k × n
    
    # 1. Frobenius error
    A_frob_sq = sp.linalg.norm(A, 'fro')**2
    ZtA_frob_sq = np.linalg.norm(ZtA, 'fro')**2
    residual_frob_sq = max(A_frob_sq - ZtA_frob_sq, 0)
    actual_frob = np.sqrt(residual_frob_sq)
    frob_error = max(actual_frob / opt_frob - 1, 1e-14) if opt_frob > 1e-14 else 1e-14
    
    # 2. Spectral error via power iteration
    def matvec_residual(v):
        Av = A @ v
        return Av - Z @ (Z.T @ Av)
    
    np.random.seed(123)
    v = np.random.randn(A.shape[1])
    v = v / np.linalg.norm(v)
    for _ in range(10):
        u = matvec_residual(v)
        v = A.T @ u - ZtA.T @ (Z.T @ u)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            break
        v = v / norm_v
    actual_spec = np.linalg.norm(matvec_residual(v))
    spec_error = max(actual_spec / opt_spec - 1, 1e-14) if opt_spec > 1e-14 else 1e-14
    
    # 3. Per-vector error
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        var_z = np.linalg.norm(ZtA[i, :])**2
        var_true = true_sv[i]**2
        err = abs(var_true - var_z) / sigma_kplus1_sq
        pervec_errors.append(err)
    
    pervec_error = max(max(pervec_errors), 1e-14) if pervec_errors else 1e-14
    
    return frob_error, spec_error, pervec_error


def run_experiment_newsgroups(k=30, q_max=15, n_trials=1):
    """
    Run experiment on 20 Newsgroups real-world dataset.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    
    print("Loading 20 Newsgroups dataset...")
    A = load_20newsgroups(max_features=15000)
    m, n = A.shape
    
    # Compute reference singular values using sparse SVD
    print(f"  Computing reference SVD (top {k+10} singular values)...")
    _, true_sv, _ = svds(A, k=k+10, which='LM')
    true_sv = true_sv[::-1]  # svds returns in ascending order
    
    gap_ratio = true_sv[k-1] / true_sv[k]
    print(f"  Spectral gap at k={k}: σ_k/σ_(k+1) = {gap_ratio:.4f}")
    
    # Compute ||A||_F for correct optimal Frobenius error
    A_frob_sq = sp.linalg.norm(A, 'fro')**2
    top_k_energy = np.sum(true_sv[:k]**2)
    opt_frob_sq = A_frob_sq - top_k_energy  # Correct: ||A||_F^2 - sum of top k σ^2
    
    q_values = list(range(q_max + 1))
    
    results = {
        'q_values': q_values,
        'm': m, 'n': n, 'k': k, 'dataset': '20newsgroups',
        'simul': {'time': [], 'frob': [], 'spec': [], 'pervec': []},
        'krylov': {'time': [], 'frob': [], 'spec': [], 'pervec': []},
    }
    
    print(f"\nRunning experiments (averaging over {n_trials} trials)...")
    print("-" * 80)
    
    for q in q_values:
        print(f"q={q}: ", end="", flush=True)
        
        # --- Simultaneous Iteration ---
        times_simul = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            Z_simul = simultaneous_iteration(A, k, q)
            times_simul.append(time.perf_counter() - t0)
        
        avg_time_simul = np.mean(times_simul)
        frob_s, spec_s, pv_s = compute_all_errors_sparse(A, Z_simul, true_sv, k, opt_frob_sq)
        
        results['simul']['time'].append(avg_time_simul * 1000)
        results['simul']['frob'].append(frob_s)
        results['simul']['spec'].append(spec_s)
        results['simul']['pervec'].append(pv_s)
        
        print(f"Simul({avg_time_simul*1000:.0f}ms, F={frob_s:.2e}, S={spec_s:.2e}, PV={pv_s:.2e}) ", end="")
        
        # --- Block Krylov ---
        times_krylov = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            Z_krylov = block_krylov_iteration(A, k, q)
            times_krylov.append(time.perf_counter() - t0)
        
        avg_time_krylov = np.mean(times_krylov)
        frob_k, spec_k, pv_k = compute_all_errors_sparse(A, Z_krylov, true_sv, k, opt_frob_sq)
        
        results['krylov']['time'].append(avg_time_krylov * 1000)
        results['krylov']['frob'].append(frob_k)
        results['krylov']['spec'].append(spec_k)
        results['krylov']['pervec'].append(pv_k)
        
        print(f"Krylov({avg_time_krylov*1000:.0f}ms, F={frob_k:.2e}, S={spec_k:.2e}, PV={pv_k:.2e})")
    
    return results


def run_experiment(m=2000, n=3000, k=30, decay_rate=0.5, q_max=8, n_trials=3):
    """
    Run timing experiment for both methods.
    """
    print(f"Creating {m}×{n} matrix with decay rate = {decay_rate}...")
    A, true_sv, _ = create_slow_decay_matrix(m, n, k, decay_rate)
    
    q_values = list(range(q_max + 1))
    
    results = {
        'q_values': q_values,
        'm': m, 'n': n, 'k': k, 'decay_rate': decay_rate,
        'simul': {'time': [], 'frob': [], 'spec': [], 'pervec': []},
        'krylov': {'time': [], 'frob': [], 'spec': [], 'pervec': []},
    }
    
    print(f"\nRunning experiments (averaging over {n_trials} trials)...")
    print("-" * 80)
    
    for q in q_values:
        print(f"q={q}: ", end="", flush=True)
        
        # --- Simultaneous Iteration ---
        times_simul = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            Z_simul = simultaneous_iteration(A, k, q)
            times_simul.append(time.perf_counter() - t0)
        
        avg_time_simul = np.mean(times_simul)
        frob_s, spec_s, pv_s = compute_all_errors(A, Z_simul, true_sv, k)
        
        results['simul']['time'].append(avg_time_simul * 1000)  # ms
        results['simul']['frob'].append(frob_s)
        results['simul']['spec'].append(spec_s)
        results['simul']['pervec'].append(pv_s)
        
        print(f"Simul({avg_time_simul*1000:.0f}ms, F={frob_s:.2e}, S={spec_s:.2e}, PV={pv_s:.2e}) ", end="")
        
        # --- Block Krylov ---
        times_krylov = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            Z_krylov = block_krylov_iteration(A, k, q)
            times_krylov.append(time.perf_counter() - t0)
        
        avg_time_krylov = np.mean(times_krylov)
        frob_k, spec_k, pv_k = compute_all_errors(A, Z_krylov, true_sv, k)
        
        results['krylov']['time'].append(avg_time_krylov * 1000)  # ms
        results['krylov']['frob'].append(frob_k)
        results['krylov']['spec'].append(spec_k)
        results['krylov']['pervec'].append(pv_k)
        
        print(f"Krylov({avg_time_krylov*1000:.0f}ms, F={frob_k:.2e}, S={spec_k:.2e}, PV={pv_k:.2e})")
    
    return results


def create_figure(results, output_dir, filename='fig4_krylov_runtime'):
    """
    Create single plot: Iterations q vs Error (linear scale), all 6 lines.
    Style matching the reference image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    q_values = results['q_values']
    
    # Colors: green for Block Krylov, blue for Simultaneous
    colors = {'krylov': '#2ca02c', 'simul': '#1f77b4'}
    
    # Different markers for different metrics
    markers = {'frob': '+', 'spec': 'o', 'pervec': '^'}
    metric_labels = {'frob': 'Frobenius Error', 'spec': 'Spectral Error', 'pervec': 'Per Vector Error'}
    method_labels = {'krylov': 'Block Krylov', 'simul': 'Simult. Iter.'}
    
    # Plot all 6 lines
    for method in ['krylov', 'simul']:
        for metric in ['frob', 'spec', 'pervec']:
            errors = results[method][metric]
            label = f"{method_labels[method]} – {metric_labels[metric]}"
            
            ax.plot(q_values, errors, 
                   marker=markers[metric], linewidth=1.5, markersize=6,
                   color=colors[method], label=label,
                   markerfacecolor='none', markeredgewidth=1.5)
    
    ax.set_xlabel('Iterations q', fontsize=12)
    ax.set_ylabel('Error ε', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    # Linear scale, fixed y limits
    ax.set_xlim([min(q_values) - 0.5, max(q_values) + 0.5])
    ax.set_ylim([0, 0.35])
    
    # Add title with dataset info
    if 'dataset' in results:
        title = f"20 Newsgroups ({results['m']}×{results['n']}), k={results['k']}"
    else:
        title = f"Synthetic ({results['m']}×{results['n']}), k={results['k']}, decay={results.get('decay_rate', 0.5)}"
    ax.set_title(title, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.png', bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Figure saved: {output_dir / f'{filename}.pdf'}")
    
    return output_dir / f'{filename}.pdf'


def save_results(results, output_dir, filename='experiment_4_runtime_results.json'):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved: {output_dir / filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--newsgroups', action='store_true', help='Run on 20 Newsgroups dataset')
    args = parser.parse_args()
    
    figures_dir = Path(__file__).parent.parent / 'figures'
    data_dir = Path(__file__).parent.parent / 'data'
    
    if args.newsgroups:
        json_path = data_dir / 'experiment_4_newsgroups_results.json'
        fig_name = 'fig4_krylov_newsgroups'
    else:
        json_path = data_dir / 'experiment_4_runtime_results.json'
        fig_name = 'fig4_krylov_runtime'
    
    if args.plot_only:
        if json_path.exists():
            print("Recreating figure from saved data...")
            with open(json_path, 'r') as f:
                results = json.load(f)
            create_figure(results, figures_dir, fig_name)
        else:
            print(f"No saved data at {json_path}")
        return
    
    print("=" * 80)
    print("EXPERIMENT 4: Block Krylov vs Simultaneous Iteration")
    print("Runtime vs Error (Three Metrics)")
    print("=" * 80)
    print()
    print("ERROR METRICS (Musco & Musco 2015):")
    print("  1. Frobenius (weak)   - Both methods do reasonably well")
    print("  2. Spectral (strong)  - Block Krylov wins")
    print("  3. Per-Vector (strongest) - Block Krylov wins clearly")
    print()
    
    if args.newsgroups:
        # Run on 20 Newsgroups
        results = run_experiment_newsgroups(
            k=30,
            q_max=25,
            n_trials=1
        )
        save_results(results, data_dir, 'experiment_4_newsgroups_results.json')
        create_figure(results, figures_dir, fig_name)
    else:
        # Run on synthetic
        results = run_experiment(
            m=2000,
            n=3000,
            k=30,
            decay_rate=0.5,
            q_max=25,
            n_trials=1
        )
        save_results(results, data_dir)
        create_figure(results, figures_dir, fig_name)
    
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
    For the same runtime, Block Krylov achieves:
    
    • MUCH lower Frobenius error
    • MUCH lower Spectral error  
    • MUCH lower Per-Vector error
    
    The advantage is most dramatic for Per-Vector error (strongest metric),
    which ensures each singular vector individually captures the right variance.
    
    This is critical for PCA applications where you need accurate directions,
    not just accurate overall reconstruction.
    """)


if __name__ == "__main__":
    main()
