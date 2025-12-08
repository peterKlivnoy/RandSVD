#!/usr/bin/env python3
"""
Experiment 6: Block Krylov vs Simultaneous Iteration
Based on Musco & Musco (2015) - Randomized Block Krylov Methods

This version properly computes the error metrics to match the paper's figures.

Key insight from paper:
- ε = (||A - Π_Z A||_X / ||A - A_k||_X) - 1
- Where X is Frobenius, Spectral, or Per-Vector norm
- The denominator ||A - A_k||_X is the OPTIMAL rank-k error
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from pathlib import Path
import time


def load_20newsgroups():
    """Load 20 Newsgroups TF-IDF matrix."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("  Loading 20 Newsgroups dataset...")
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        
        print("  Computing TF-IDF representation...")
        vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
        A = vectorizer.fit_transform(newsgroups.data)
        
        print(f"  Matrix shape: {A.shape}, nnz: {A.nnz}")
        return A
    except Exception as e:
        print(f"  Could not load: {e}")
        return None


def create_synthetic_matrix(m, n, decay_rate, seed=42):
    """
    Create matrix with singular values σ_i = 1/i^decay_rate.
    """
    k_full = min(m, n)
    singular_values = np.array([1.0 / (i + 1)**decay_rate for i in range(k_full)])
    
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(m, k_full))
    V, _ = np.linalg.qr(rng.randn(n, k_full))
    
    A = U @ np.diag(singular_values) @ V.T
    return A, singular_values


def simultaneous_iteration(A, k, q, seed=42):
    """
    Algorithm 1 from Musco & Musco (2015): Simultaneous Iteration
    
    Input: A ∈ R^{n×d}, error ε ∈ (0,1), rank k ≤ n,d
    
    1. K := (AA^T)^q * A * Π
    2. Q := orthonormalize(K) ∈ R^{n×k}
    3. M := Q^T AA^T Q ∈ R^{k×k}
    4. Ū_k := top k singular vectors of M
    5. return Z = Q Ū_k
    
    Note: We add intermediate orthogonalization for numerical stability.
    """
    m, n = A.shape
    rng = np.random.RandomState(seed)
    Pi = rng.randn(n, k)  # Random projection matrix
    
    # Step 1: K = (AA^T)^q * A * Π  (with stabilization)
    K = A @ Pi  # Start with A*Π
    for _ in range(q):
        K, _ = np.linalg.qr(K)  # Stabilize to prevent overflow
        K = A @ (A.T @ K)  # Apply AA^T
    
    # Step 2: Orthonormalize K to get Q
    Q, _ = np.linalg.qr(K)
    
    # Step 3: M = Q^T AA^T Q
    AtQ = A.T @ Q  # d × k
    if hasattr(AtQ, 'toarray'):
        AtQ = AtQ.toarray()
    M = AtQ.T @ AtQ  # k × k, this is Q^T A A^T Q
    
    # Step 4: Get top k singular vectors of M
    # M is symmetric PSD, so eigendecomposition = SVD
    eigenvalues, U_k = np.linalg.eigh(M)
    # eigh returns ascending order, reverse for descending
    idx = np.argsort(eigenvalues)[::-1]
    U_k = U_k[:, idx[:k]]
    
    # Step 5: Z = Q * Ū_k
    Z = Q @ U_k
    return Z


def block_krylov_iteration(A, k, q, seed=42):
    """
    Algorithm 2 from Musco & Musco (2015): Block Krylov Iteration
    
    Input: A ∈ R^{n×d}, error ε ∈ (0,1), rank k ≤ n,d
    
    1. K := [AΠ, (AA^T)AΠ, ..., (AA^T)^q AΠ]  ← Concatenate ALL blocks
    2. Q := orthonormalize(K) ∈ R^{n×qk}      ← MUCH larger subspace!
    3. M := Q^T AA^T Q ∈ R^{qk×qk}
    4. Ū_k := top k singular vectors of M
    5. return Z = Q Ū_k
    
    Key insight: By keeping all Krylov blocks, we search a LARGER subspace
    of dimension qk instead of just k. This gives gap-independent convergence.
    """
    m, n = A.shape
    rng = np.random.RandomState(seed)
    Pi = rng.randn(n, k)  # Random projection matrix
    
    # Step 1: Build K = [AΠ, (AA^T)AΠ, ..., (AA^T)^q AΠ]
    Y = A @ Pi  # Start with A*Π
    blocks = [Y.copy()]
    
    for _ in range(q):
        Y = A @ (A.T @ Y)  # Apply AA^T
        blocks.append(Y.copy())
    
    # Concatenate all blocks: K is n × k(q+1)
    K = np.hstack(blocks)
    
    # Step 2: Orthonormalize K to get Q ∈ R^{n × min(n, k(q+1))}
    Q, _ = np.linalg.qr(K)
    
    # Step 3: M = Q^T AA^T Q
    AtQ = A.T @ Q  # d × qk
    if hasattr(AtQ, 'toarray'):
        AtQ = AtQ.toarray()
    M = AtQ.T @ AtQ  # qk × qk
    
    # Step 4: Get top k singular vectors of M
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    U_k = eigenvectors[:, idx[:k]]
    
    # Step 5: Z = Q * Ū_k
    Z = Q @ U_k
    return Z


def compute_projection_errors(A, Z, true_singular_values, k):
    """
    Compute error metrics as in Musco & Musco.
    
    Given:
    - A: the matrix
    - Z: orthonormal basis for approximate row space (m × k)
    - true_singular_values: σ_1, ..., σ_min(m,n)
    - k: target rank
    
    Returns: (frob_error, spec_error, pervec_error)
    
    Where error = (actual / optimal) - 1
    """
    # Compute projection Π_Z A = Z @ Z^T @ A
    ZtA = Z.T @ A
    if hasattr(ZtA, 'toarray'):
        ZtA = ZtA.toarray()
    
    # Residual A - Π_Z A
    if sp.issparse(A):
        # For sparse A, compute ||A - ZZ^TA||_F^2 = ||A||_F^2 - ||Z^TA||_F^2
        A_frob_sq = (A.data ** 2).sum()
        ZtA_frob_sq = np.linalg.norm(ZtA, 'fro')**2
        actual_frob_sq = max(0, A_frob_sq - ZtA_frob_sq)
        actual_frob = np.sqrt(actual_frob_sq)
        
        # Spectral norm via power iteration on residual
        actual_spec = estimate_spectral_norm_residual(A, Z, n_iter=30)
    else:
        residual = A - Z @ ZtA
        actual_frob = np.linalg.norm(residual, 'fro')
        actual_spec = np.linalg.norm(residual, 2)
    
    # Optimal errors (Eckart-Young theorem)
    # ||A - A_k||_F = sqrt(σ_{k+1}^2 + σ_{k+2}^2 + ...)
    # ||A - A_k||_2 = σ_{k+1}
    opt_frob = np.sqrt(np.sum(true_singular_values[k:]**2))
    opt_spec = true_singular_values[k] if k < len(true_singular_values) else 1e-14
    
    # Relative errors
    frob_error = (actual_frob / opt_frob - 1) if opt_frob > 1e-14 else 0
    spec_error = (actual_spec / opt_spec - 1) if opt_spec > 1e-14 else 0
    
    # Per-vector error: max_i |σ_i^2 - ||A^T z_i||^2| / σ_i^2
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        z_i = Z[:, i]
        if sp.issparse(A):
            Atz_i = A.T @ z_i
            if hasattr(Atz_i, 'toarray'):
                Atz_i = Atz_i.toarray().flatten()
            captured_var = np.linalg.norm(Atz_i)**2
        else:
            captured_var = np.linalg.norm(A.T @ z_i)**2
        
        true_var = true_singular_values[i]**2
        if true_var > 1e-14:
            rel_err = abs(true_var - captured_var) / true_var
            pervec_errors.append(rel_err)
    
    pervec_error = max(pervec_errors) if pervec_errors else 0
    
    return frob_error, spec_error, pervec_error


def estimate_spectral_norm_residual(A, Z, n_iter=30):
    """
    Estimate ||A - ZZ^T A||_2 using power iteration.
    """
    m, n = A.shape
    rng = np.random.RandomState(123)
    v = rng.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        # u = (A - ZZ^TA) v
        Av = A @ v
        ZZtAv = Z @ (Z.T @ Av)
        u = Av - ZZtAv
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-14:
            return 0.0
        u = u / u_norm
        
        # v = (A - ZZ^TA)^T u = A^T u - A^T Z Z^T u
        Atu = A.T @ u
        ZZtu = Z @ (Z.T @ u)
        AtZZtu = A.T @ ZZtu
        w = Atu - AtZZtu
        if hasattr(w, 'toarray'):
            w = w.toarray().flatten()
        v = w / (np.linalg.norm(w) + 1e-14)
    
    # Final estimate
    Av = A @ v
    ZZtAv = Z @ (Z.T @ Av)
    return np.linalg.norm(Av - ZZtAv)


def run_experiment(A, true_sv, k, q_max, is_sparse=False):
    """
    Run experiment for both methods across q = 0, 1, ..., q_max.
    """
    q_values = list(range(q_max + 1))
    
    results = {
        'simul': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
        'krylov': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
        'q_values': q_values
    }
    
    for q in q_values:
        print(f"  q={q}: ", end="", flush=True)
        
        # Simultaneous Iteration
        t0 = time.perf_counter()
        Z_simul = simultaneous_iteration(A, k, q)
        t_simul = time.perf_counter() - t0
        
        f, s, p = compute_projection_errors(A, Z_simul, true_sv, k)
        results['simul']['frob'].append(f)
        results['simul']['spec'].append(s)
        results['simul']['pervec'].append(p)
        results['simul']['time'].append(t_simul)
        
        # Block Krylov
        t0 = time.perf_counter()
        Z_krylov = block_krylov_iteration(A, k, q)
        t_krylov = time.perf_counter() - t0
        
        f, s, p = compute_projection_errors(A, Z_krylov, true_sv, k)
        results['krylov']['frob'].append(f)
        results['krylov']['spec'].append(s)
        results['krylov']['pervec'].append(p)
        results['krylov']['time'].append(t_krylov)
        
        print(f"Simul(S={results['simul']['spec'][-1]:.3f}, PV={results['simul']['pervec'][-1]:.3f}) "
              f"Krylov(S={results['krylov']['spec'][-1]:.3f}, PV={results['krylov']['pervec'][-1]:.3f})")
    
    return results


def create_paper_style_figure(results_dict, output_dir):
    """
    Create figure matching the style of Musco & Musco (2015) Figure 1.
    
    Layout: 2×2 grid
    - (a) Synthetic matrix with k=30
    - (b) Another synthetic or configuration  
    - (c) 20 Newsgroups, k=20
    - (d) 20 Newsgroups runtime cost
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Style matching paper
    krylov_colors = {
        'frob': '#2ca02c',    # Green
        'spec': '#98df8a',    # Light green  
        'pervec': '#bcbd22',  # Yellow-green
    }
    simul_colors = {
        'frob': '#1f77b4',    # Blue
        'spec': '#aec7e8',    # Light blue
        'pervec': '#17becf',  # Cyan
    }
    
    metric_labels = {
        'frob': 'Frobenius Error',
        'spec': 'Spectral Error',
        'pervec': 'Per Vector Error'
    }
    
    def plot_panel(ax, results, title, show_legend=False):
        q_values = results['q_values']
        
        for metric in ['frob', 'spec', 'pervec']:
            # Block Krylov
            vals = [max(v, 0.001) for v in results['krylov'][metric]]
            ax.plot(q_values, vals, '-o', color=krylov_colors[metric], 
                   markersize=4, linewidth=1.5,
                   label=f'Block Krylov – {metric_labels[metric]}')
            
            # Simultaneous
            vals = [max(v, 0.001) for v in results['simul'][metric]]
            ax.plot(q_values, vals, '-^', color=simul_colors[metric],
                   markersize=4, linewidth=1.5,
                   label=f'Simult. Iter. – {metric_labels[metric]}')
        
        ax.set_xlabel('Iterations q')
        ax.set_ylabel('Error ε')
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        if show_legend:
            ax.legend(loc='upper right', fontsize=7)
    
    # Plot each panel
    configs = list(results_dict.keys())
    
    for idx, (config, results) in enumerate(results_dict.items()):
        row, col = idx // 2, idx % 2
        if idx < 4:
            plot_panel(axes[row, col], results, config, show_legend=(idx == 0))
    
    # If we have a 4th panel for runtime, add it
    if len(configs) >= 3 and 'newsgroups' in configs[2].lower():
        ax = axes[1, 1]
        results = list(results_dict.values())[2]  # 20 Newsgroups
        q_values = results['q_values']
        
        # Cumulative runtime
        for metric in ['frob', 'spec', 'pervec']:
            # Krylov runtime
            times = np.cumsum(results['krylov']['time'])
            vals = [max(v, 0.001) for v in results['krylov'][metric]]
            ax.plot(times, vals, '-o', color=krylov_colors[metric], 
                   markersize=4, linewidth=1.5,
                   label=f'Block Krylov – {metric_labels[metric]}')
            
            # Simul runtime
            times = np.cumsum(results['simul']['time'])
            vals = [max(v, 0.001) for v in results['simul'][metric]]
            ax.plot(times, vals, '-^', color=simul_colors[metric],
                   markersize=4, linewidth=1.5,
                   label=f'Simult. Iter. – {metric_labels[metric]}')
        
        ax.set_xlabel('Runtime (seconds)')
        ax.set_ylabel('Error ε')
        ax.set_title('20 Newsgroups, k=20, runtime cost')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='gray')
    
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'fig6_krylov.pdf')
    plt.savefig(output_dir / 'fig6_krylov.png', dpi=150)
    plt.close()
    
    return output_dir / 'fig6_krylov.pdf'


def main():
    print("=" * 70)
    print("EXPERIMENT 6: Block Krylov vs Simultaneous Iteration")
    print("Based on Musco & Musco (2015)")
    print("=" * 70)
    
    results_dict = {}
    
    # =========================================================================
    # Part 1: Synthetic matrix (like paper's SNAP/AMAZON)
    # =========================================================================
    print("\n[1/3] Synthetic Matrix (1000×2000, decay=1.0, k=30)")
    print("-" * 50)
    
    m, n, k, q_max = 1000, 2000, 30, 25
    A_synth, true_sv_synth = create_synthetic_matrix(m, n, decay_rate=1.0)
    
    print(f"  σ_k={true_sv_synth[k-1]:.4f}, σ_{k+1}={true_sv_synth[k]:.4f}, "
          f"gap={true_sv_synth[k-1]/true_sv_synth[k]:.2f}")
    
    results_dict[f'Synthetic {m}×{n}, k={k}'] = run_experiment(A_synth, true_sv_synth, k, q_max)
    
    # =========================================================================
    # Part 2: Different decay rate
    # =========================================================================
    print("\n[2/3] Synthetic Matrix (1000×2000, decay=0.5, k=10)")
    print("-" * 50)
    
    k2 = 10
    A_synth2, true_sv_synth2 = create_synthetic_matrix(m, n, decay_rate=0.5)
    
    print(f"  σ_k={true_sv_synth2[k2-1]:.4f}, σ_{k2+1}={true_sv_synth2[k2]:.4f}, "
          f"gap={true_sv_synth2[k2-1]/true_sv_synth2[k2]:.2f}")
    
    results_dict[f'Synthetic {m}×{n}, k={k2}'] = run_experiment(A_synth2, true_sv_synth2, k2, q_max)
    
    # =========================================================================
    # Part 3: 20 Newsgroups (real data)
    # =========================================================================
    print("\n[3/3] 20 Newsgroups (real data, k=20)")
    print("-" * 50)
    
    A_news = load_20newsgroups()
    
    if A_news is not None:
        k_news = 20
        
        # Compute reference singular values
        # For sparse, we need MORE singular values to properly compute optimal error
        print("  Computing reference SVD...")
        n_sv = min(100, min(A_news.shape) - 1)
        U_ref, s_ref, Vt_ref = svds(A_news, k=n_sv)
        # svds returns ascending order, reverse it
        s_ref = s_ref[::-1]
        
        print(f"  Top singular values: {s_ref[:5]}")
        print(f"  σ_k={s_ref[k_news-1]:.4f}, σ_{k_news+1}={s_ref[k_news]:.4f}")
        
        results_dict[f'20 Newsgroups, k={k_news}'] = run_experiment(
            A_news, s_ref, k_news, q_max, is_sparse=True)
    
    # =========================================================================
    # Generate Figure
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating figure...")
    print("=" * 70)
    
    output_path = create_paper_style_figure(
        results_dict, 
        Path(__file__).parent.parent / 'figures'
    )
    
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Saved: {output_path.with_suffix('.png')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for config, res in results_dict.items():
        q1_simul_pv = res['simul']['pervec'][1] if len(res['simul']['pervec']) > 1 else 0
        q1_krylov_pv = res['krylov']['pervec'][1] if len(res['krylov']['pervec']) > 1 else 0
        print(f"\n{config}:")
        print(f"  At q=1: Simul PV={q1_simul_pv:.3f}, Krylov PV={q1_krylov_pv:.3f}")
        if q1_krylov_pv > 0:
            print(f"  Krylov advantage: {q1_simul_pv/q1_krylov_pv:.1f}x better")


if __name__ == "__main__":
    main()
