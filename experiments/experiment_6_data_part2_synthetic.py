#!/usr/bin/env python3
"""
Experiment 6 - Part 2: Block Krylov vs Simultaneous Iteration on Synthetic Matrices
Based on Musco & Musco (2015)

This runs synthetic matrix experiments and saves results to a pickle file.
"""

import numpy as np
import time
import pickle
from pathlib import Path


def create_heavy_tail_matrix(m, n, decay_rate):
    """Create matrix with singular values σ_i = 1/i^decay_rate."""
    k_full = min(m, n)
    singular_values = np.array([1.0 / (i + 1)**decay_rate for i in range(k_full)])
    
    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(m, k_full))
    V, _ = np.linalg.qr(np.random.randn(n, k_full))
    
    A = U @ np.diag(singular_values) @ V.T
    
    return A, singular_values, U, V


def simultaneous_iteration(A, k, q):
    """Algorithm 1: Simultaneous Power Iteration."""
    m, n = A.shape
    np.random.seed(0)
    Omega = np.random.randn(n, k)
    
    Y = A @ Omega
    for _ in range(q):
        Y, _ = np.linalg.qr(Y)
        Y = A @ (A.T @ Y)
    
    Z, _ = np.linalg.qr(Y)
    return Z


def block_krylov_iteration(A, k, q):
    """
    Algorithm 2: Block Krylov Iteration.
    Builds Krylov subspace and returns best rank-k approximation within it.
    """
    m, n = A.shape
    np.random.seed(0)
    Omega = np.random.randn(n, k)
    
    Y = A @ Omega
    blocks = [Y.copy()]
    
    for _ in range(q):
        Y = A @ (A.T @ Y)
        blocks.append(Y.copy())
    
    # Concatenate all blocks
    K = np.hstack(blocks)  # m × k(q+1)
    Z, _ = np.linalg.qr(K)
    
    # Project A onto Krylov space and find best rank-k approximation
    B = Z.T @ A
    U_B, s_B, Vt_B = np.linalg.svd(B, full_matrices=False)
    
    # Top-k left singular vectors in original space
    Z_k = Z @ U_B[:, :k]
    
    return Z_k


def compute_metrics(A, Z, true_sv, true_U, k):
    """
    Compute the three metrics from Musco & Musco (2015).
    
    1. Frobenius: ε where ||A - ZZ^TA||_F ≤ (1+ε)||A - A_k||_F
    2. Spectral:  ε where ||A - ZZ^TA||_2 ≤ (1+ε)||A - A_k||_2
    3. Per-Vector: ε where |σ_i² - ||A^Tz_i||²| ≤ ε·σ_i² for all i ≤ k
    """
    proj = Z @ (Z.T @ A)
    residual = A - proj
    
    opt_frob = np.sqrt(np.sum(true_sv[k:]**2))
    opt_spec = true_sv[k] if k < len(true_sv) else 0
    
    # 1. Frobenius norm error
    actual_frob = np.linalg.norm(residual, 'fro')
    frob_error = (actual_frob / opt_frob - 1) if opt_frob > 1e-14 else 0
    
    # 2. Spectral norm error
    actual_spec = np.linalg.norm(residual, 2)
    spec_error = (actual_spec / opt_spec - 1) if opt_spec > 1e-14 else 0
    
    # 3. Per-vector error: max_i |σ_i² - ||A^Tz_i||²| / σ_i²
    pervec_errors = []
    for i in range(min(k, Z.shape[1])):
        z_i = Z[:, i]
        var_z = np.linalg.norm(A.T @ z_i)**2
        var_true = true_sv[i]**2
        # Relative error normalized by TRUE variance (not σ_{k+1})
        if var_true > 1e-14:
            err = abs(var_true - var_z) / var_true
            pervec_errors.append(err)
    
    pervec_error = max(pervec_errors) if pervec_errors else 0
    
    return frob_error, spec_error, pervec_error


def run_experiment(A, true_sv, true_U, k, q_values):
    """Run both methods across iteration counts."""
    
    results = {
        'simul': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
        'krylov': {'frob': [], 'spec': [], 'pervec': [], 'time': []},
    }
    
    for q in q_values:
        print(f"  q={q}: ", end="", flush=True)
        
        # Simultaneous Iteration
        t0 = time.perf_counter()
        Z_simul = simultaneous_iteration(A, k, q)
        t_simul = time.perf_counter() - t0
        
        frob, spec, pervec = compute_metrics(A, Z_simul, true_sv, true_U, k)
        results['simul']['frob'].append(frob)
        results['simul']['spec'].append(spec)
        results['simul']['pervec'].append(pervec)
        results['simul']['time'].append(t_simul)
        
        print(f"Simul(F={frob:.3f}, S={spec:.3f}, PV={pervec:.2f}) ", end="")
        
        # Block Krylov
        t0 = time.perf_counter()
        Z_krylov = block_krylov_iteration(A, k, q)
        t_krylov = time.perf_counter() - t0
        
        frob, spec, pervec = compute_metrics(A, Z_krylov, true_sv, true_U, k)
        results['krylov']['frob'].append(frob)
        results['krylov']['spec'].append(spec)
        results['krylov']['pervec'].append(pervec)
        results['krylov']['time'].append(t_krylov)
        
        print(f"Krylov(F={frob:.3f}, S={spec:.3f}, PV={pervec:.2f})")
    
    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 6 - PART 2: Synthetic Matrices")
    print("Based on Musco & Musco (2015)")
    print("=" * 70)
    
    # Parameters
    k = 20
    q_values = list(range(0, 26))  # 0 to 25, like in the paper
    m, n = 1000, 2000  # Moderate size for reasonable speed
    
    all_results = {}
    all_labels = {}
    
    # =========================================================================
    # Experiment A: Heavy tail (decay = 0.5)
    # =========================================================================
    print("\n" + "-" * 70)
    print("PART 2A: Synthetic Matrix - Heavy Tail (decay=0.5)")
    print("-" * 70)
    
    decay = 0.5
    print(f"\nCreating {m}×{n} matrix with σ_i = 1/i^{decay}...")
    A, true_sv, true_U, _ = create_heavy_tail_matrix(m, n, decay)
    
    gap_ratio = true_sv[k-1] / true_sv[k]
    print(f"  σ_{k} = {true_sv[k-1]:.6f}, σ_{k+1} = {true_sv[k]:.6f}, gap ratio = {gap_ratio:.3f}")
    
    print("Running experiments...")
    all_results['synth_heavy'] = run_experiment(A, true_sv, true_U, k, q_values)
    all_labels['synth_heavy'] = f'Synthetic {m}×{n}\n(Heavy tail, gap={gap_ratio:.2f})'
    
    # =========================================================================
    # Experiment B: Zipf decay (decay = 1.0)
    # =========================================================================
    print("\n" + "-" * 70)
    print("PART 2B: Synthetic Matrix - Zipf Decay (decay=1.0)")
    print("-" * 70)
    
    decay = 1.0
    print(f"\nCreating {m}×{n} matrix with σ_i = 1/i^{decay}...")
    A, true_sv, true_U, _ = create_heavy_tail_matrix(m, n, decay)
    
    gap_ratio = true_sv[k-1] / true_sv[k]
    print(f"  σ_{k} = {true_sv[k-1]:.6f}, σ_{k+1} = {true_sv[k]:.6f}, gap ratio = {gap_ratio:.3f}")
    
    print("Running experiments...")
    all_results['synth_zipf'] = run_experiment(A, true_sv, true_U, k, q_values)
    all_labels['synth_zipf'] = f'Synthetic {m}×{n}\n(Zipf decay, gap={gap_ratio:.2f})'
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "-" * 70)
    print("Saving results...")
    print("-" * 70)
    
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'results': all_results,
        'labels': all_labels,
        'q_values': q_values,
        'k': k,
        'matrix_size': (m, n)
    }
    
    output_file = output_dir / 'exp6_synthetic_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"  ✓ Saved to {output_file}")
    
    print("\n✓ Part 2 complete! Run experiment_6_combine.py to generate figure.")


if __name__ == "__main__":
    main()
