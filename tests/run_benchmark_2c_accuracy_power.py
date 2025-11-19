import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Path Hack ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Hack ---

from src.randsvd_algorithm import randSVD

print("Running Benchmark 2c: Accuracy vs. Optimal (Power Method Fix)")
print("Testing RandSVD on a slow-decay matrix with q=2 power iterations.")

# --- 1. Create Test Matrix with SLOW Decaying Spectrum ---
m, n = 2000, 1000
print(f"Creating {m}x{n} test matrix with (slow) polynomial spectrum...")

# Create random orthogonal matrices U and V
U_full, _ = np.linalg.qr(np.random.randn(m, m))
U = U_full[:, :n]

V_full, _ = np.linalg.qr(np.random.randn(n, n))
V_t = V_full.T

# Create a known, POLYNOMIALLY decaying spectrum (sigma_i = 1/i)
S_full = 1.0 / (np.arange(n) + 1)

Sigma_diag = np.diag(S_full)

# Assemble the "ground truth" matrix A = U * Sigma * V^T
A = U @ Sigma_diag @ V_t
print("Test matrix A (slow decay) created.")

# --- 2. Set Benchmark Parameters ---
k_range = np.arange(5, 101, 5)
oversampling_p = 10

# --- THE CRUCIAL FIX ---
power_iterations_q = 2 # This is the key difference to Benchmark 2b
# -----------------------

errors_rand_q0 = []
errors_rand_q2 = []
errors_opt = []

print(f"Testing ranks k={list(k_range)} with q={power_iterations_q}...")

# --- 3. Loop Over Ranks and Calculate Errors ---
for k in k_range:
    # A. Get the OPTIMAL error (Eckart-Young Theorem) 
    opt_error = S_full[k]
    errors_opt.append(opt_error)
    
    # B. Get the BASELINE (q=0) Randomized error (for comparison)
    U_r0, S_r0, V_r0_t = randSVD(A, k, oversampling_p, q=0)
    A_approx_q0 = U_r0[:, :k] @ np.diag(S_r0[:k]) @ V_r0_t[:k, :]
    errors_rand_q0.append(np.linalg.norm(A - A_approx_q0, ord=2))

    # C. Get the IMPROVED (q=2) Randomized error (the fix)
    # The randSVD function MUST be updated with the 'q' parameter for this to work.
    U_r2, S_r2, V_r2_t = randSVD(A, k, oversampling_p, q=power_iterations_q)
    A_approx_q2 = U_r2[:, :k] @ np.diag(S_r2[:k]) @ V_r2_t[:k, :]
    errors_rand_q2.append(np.linalg.norm(A - A_approx_q2, ord=2))
    
    # print(f"  k={k}: Opt={opt_error:.4e}, Rand(q=0)={errors_rand_q0[-1]:.4e}, Rand(q=2)={errors_rand_q2[-1]:.4e}")

print("Benchmark 2c complete (Power Method validated).")

# --- 4. Plot the Results (Log-Linear Plot) ---
plt.figure(figsize=(10, 6))
plt.semilogy(k_range, errors_opt, 'b-', marker='o', label=r'Optimal (Eckart-Young: $\sigma_{k+1}$)') 
plt.semilogy(k_range, errors_rand_q0, 'r:', marker='x', alpha=0.5, label=f'Randomized SVD (q=0, Baseline Failure)')
plt.semilogy(k_range, errors_rand_q2, 'g--', marker='s', label=f'Randomized SVD (q={power_iterations_q}, Subspace Iteration)') # Green line for the fix
plt.xlabel('Target Rank (k)')
plt.ylabel('Approximation Error (Spectral Norm)')
plt.title('Benchmark 2c: Accuracy Fix - Power Iteration vs. Optimal (Slow Decay)')
plt.legend()
plt.grid(True, which="both", ls="--")

# --- 5. Save Figure ---
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
save_path = os.path.join(FIGURE_DIR, 'benchmark_2c_power_fix.png') 
plt.savefig(save_path)
print(f"Saved accuracy plot to {save_path}")
plt.show()