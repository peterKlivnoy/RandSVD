#!/usr/bin/env python3
"""
Experiment 6 - Part 1: Block Krylov vs Simultaneous Iteration on 20 Newsgroups
Based on Musco & Musco (2015)

This runs ONLY the 20 Newsgroups real data experiment and saves results to a pickle file.
"""

import numpy as np
import time
import pickle
from pathlib import Path


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
        
        print(f"  Matrix shape: {A.shape}, nnz: {A.nnz}, density: {A.nnz / (A.shape[0] * A.shape[1]):.4f}")
        
        return A
    except Exception as e:
        print(f"  Could not load 20 Newsgroups: {e}")
        return None


def simultaneous_iteration(A, k, q, seed=0):
    """
    Algorithm 1: Simultaneous Power Iteration (standard randomized SVD).
    """
    m, n = A.shape
    rng = np.random.RandomState(seed)
    Omega = rng.randn(n, k)
    
    Y = A @ Omega
    for _ in range(q):
        Y, _ = np.linalg.qr(Y)
        Y = A @ (A.T @ Y)
    
    Z, _ = np.linalg.qr(Y)
    return Z


def block_krylov_iteration(A, k, q, seed=0):
    """
    Algorithm 2: Block Krylov Iteration.
    Builds Krylov subspace [A*Omega, (AA^T)*A*Omega, ..., (AA^T)^q*A*Omega].
    
    Returns basis for the span of the Krylov blocks, truncated to rank k.
    """
    m, n = A.shape
    rng = np.random.RandomState(seed)
    Omega = rng.randn(n, k)
    
    Y = A @ Omega
    blocks = [Y.copy()]
    
    for _ in range(q):
        Y = A @ (A.T @ Y)
        blocks.append(Y.copy())
    
    # Concatenate all blocks to form Krylov matrix
    K = np.hstack(blocks)  # m × k(q+1) matrix
    
    # Orthonormalize
    Z, _ = np.linalg.qr(K)
    
    # Return top-k columns (or all if fewer)
    # The key: Z spans a space that is k(q+1) dimensional at most
    # We project A onto this space, but only return k vectors
    # Actually for fair comparison with Simultaneous, we should return
    # the best rank-k approximation within this Krylov space
    
    # Compute A projected onto Z's column space, then get top-k
    # This is what makes Krylov powerful: it searches a LARGER space
    B = Z.T @ A  # Project A onto Krylov space
    U_B, s_B, Vt_B = np.linalg.svd(B.toarray() if hasattr(B, 'toarray') else B, full_matrices=False)
    
    # Top-k left singular vectors in original space
    Z_k = Z @ U_B[:, :k]
    
    return Z_k


def compute_metrics_sparse_efficient(A_sparse, Z, ref_sv, k):
    """
    Compute metrics efficiently for sparse matrices.
    
    Metrics from Musco & Musco (2015):
    1. Frobenius: ε where ||A - ZZ^TA||_F ≤ (1+ε)||A - A_k||_F
    2. Spectral:  ε where ||A - ZZ^TA||_2 ≤ (1+ε)||A - A_k||_2
    3. Per-Vector: ε where |σ_i² - ||A^Tz_i||²| ≤ ε·σ_i² for all i ≤ k
    """
    m, n = A_sparse.shape
    
    # 1. Frobenius norm error
    # ||A - ZZ^TA||_F^2 = ||A||_F^2 - ||Z^TA||_F^2 (Z is orthonormal)
    A_frob_sq = (A_sparse.data ** 2).sum()
    ZtA = Z.T @ A_sparse
    proj_frob_sq = np.linalg.norm(ZtA.toarray() if hasattr(ZtA, 'toarray') else ZtA, 'fro')**2
    actual_frob = np.sqrt(max(0, A_frob_sq - proj_frob_sq))
    
    opt_frob = np.sqrt(np.sum(ref_sv[k:]**2))
    frob_error = (actual_frob / opt_frob - 1) if opt_frob > 1e-14 else 0
    
    # 2. Spectral norm via power iteration
    np.random.seed(42)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(20):
        Av = A_sparse @ v
        ZZtAv = Z @ (Z.T @ Av)
        residual_v = Av - ZZtAv
        u = residual_v / (np.linalg.norm(residual_v) + 1e-14)
        
        Atu = A_sparse.T @ u
        ZZtAtu = A_sparse.T @ (Z @ (Z.T @ u))
        w = Atu - ZZtAtu
        v = w / (np.linalg.norm(w) + 1e-14)
    
    Av = A_sparse @ v
    ZZtAv = Z @ (Z.T @ Av)
    actual_spec = np.linalg.norm(Av - ZZtAv)
    
    opt_spec = ref_sv[k] if k < len(ref_sv) else 1e-14
    spec_error = (actual_spec / opt_spec - 1) if opt_spec > 1e-14 else 0
    
    # 3. Per-vector error: max_i |σ_i² - ||A^Tz_i||²| / σ_i²
    # This measures how well each z_i captures the variance of u_i
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        z_i = Z[:, i]
        Atz_i = A_sparse.T @ z_i
        var_z = np.linalg.norm(Atz_i.toarray().flatten() if hasattr(Atz_i, 'toarray') else Atz_i)**2
        var_true = ref_sv[i]**2
        # Relative error normalized by the TRUE variance (not σ_{k+1})
        if var_true > 1e-14:
            err = abs(var_true - var_z) / var_true
            pervec_errors.append(err)
    
    pervec_error = max(pervec_errors) if pervec_errors else 0
    
    return frob_error, spec_error, pervec_error


def run_experiment_newsgroups(A_sparse, k, q_values):
    """
    Run Block Krylov vs Simultaneous Iteration on 20 Newsgroups.
    Uses efficient sparse-aware metrics.
    """
    print("\n  Computing reference singular values using scipy.sparse.linalg.svds...")
    t0 = time.perf_counter()
    
    # Use scipy's sparse SVD for reference (much more accurate for sparse matrices)
    from scipy.sparse.linalg import svds
    # svds returns singular values in ascending order, so we reverse
    U_ref, ref_sv, Vt_ref = svds(A_sparse, k=k+20)
    ref_sv = ref_sv[::-1]  # Descending order
    
    print(f"    Done in {time.perf_counter() - t0:.1f}s")
    print(f"    Reference σ_k={ref_sv[k-1]:.4f}, σ_{k+1}={ref_sv[k]:.4f}")
    
    results = {
        'simul': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
        'krylov': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
    }
    
    for q in q_values:
        print(f"\n  q={q}:", flush=True)
        
        # Simultaneous Iteration
        t0 = time.perf_counter()
        Z_simul = simultaneous_iteration(A_sparse, k, q, seed=42)
        t_simul = time.perf_counter() - t0
        
        print(f"    Computing Simultaneous metrics...", end="", flush=True)
        frob, spec, pervec = compute_metrics_sparse_efficient(A_sparse, Z_simul, ref_sv, k)
        results['simul']['frob'].append(frob)
        results['simul']['spec'].append(spec)
        results['simul']['pervec'].append(pervec)
        results['simul']['time'].append(t_simul)
        print(f" F={frob:.3f}, S={spec:.3f}, PV={pervec:.2f}, t={t_simul:.2f}s")
        
        # Block Krylov
        t0 = time.perf_counter()
        Z_krylov = block_krylov_iteration(A_sparse, k, q, seed=42)
        t_krylov = time.perf_counter() - t0
        
        print(f"    Computing Krylov metrics...", end="", flush=True)
        frob, spec, pervec = compute_metrics_sparse_efficient(A_sparse, Z_krylov, ref_sv, k)
        results['krylov']['frob'].append(frob)
        results['krylov']['spec'].append(spec)
        results['krylov']['pervec'].append(pervec)
        results['krylov']['time'].append(t_krylov)
        print(f" F={frob:.3f}, S={spec:.3f}, PV={pervec:.2f}, t={t_krylov:.2f}s")
    
    return results, ref_sv


def main():
    print("=" * 70)
    print("EXPERIMENT 6 - PART 1: 20 Newsgroups (Real Data)")
    print("Based on Musco & Musco (2015)")
    print("=" * 70)
    
    # Parameters
    k = 20
    q_values = list(range(0, 26))  # 0 to 25, like in the paper
    
    # Load data
    print("\n[1/3] Loading 20 Newsgroups dataset...")
    A = load_20newsgroups()
    
    if A is None:
        print("ERROR: Could not load 20 Newsgroups dataset")
        return
    
    # Run experiment
    print("\n[2/3] Running experiment...")
    results, ref_sv = run_experiment_newsgroups(A, k, q_values)
    
    # Save results
    print("\n[3/3] Saving results...")
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'results': results,
        'ref_sv': ref_sv,
        'q_values': q_values,
        'k': k,
        'matrix_shape': A.shape,
        'config_label': f'20 Newsgroups\n({A.shape[0]}×{A.shape[1]})'
    }
    
    output_file = output_dir / 'exp6_newsgroups_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"  ✓ Saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'q':<4} {'Simul F':<10} {'Simul S':<10} {'Simul PV':<10} {'Krylov F':<10} {'Krylov S':<10} {'Krylov PV':<10}")
    print("-" * 70)
    for i, q in enumerate(q_values):
        sf = results['simul']['frob'][i]
        ss = results['simul']['spec'][i]
        sp = results['simul']['pervec'][i]
        kf = results['krylov']['frob'][i]
        ks = results['krylov']['spec'][i]
        kp = results['krylov']['pervec'][i]
        print(f"{q:<4} {sf:<10.4f} {ss:<10.4f} {sp:<10.2f} {kf:<10.4f} {ks:<10.4f} {kp:<10.2f}")
    
    print("\n✓ Part 1 complete! Run experiment_6_combine.py to generate figure.")


if __name__ == "__main__":
    main()
