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
m, n = 4000, 2000
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
oversampling_p = 5

# --- THE CRUCIAL FIX ---
power_iterations_q = [0,1,2,3] # This is the key difference to Benchmark 2b
# -----------------------

errors = [[] for _ in power_iterations_q]
errors_opt = []

print(f"Testing ranks k={list(k_range)} with q={power_iterations_q}...")

# --- 3. Loop Over Ranks and Calculate Errors ---
for k in k_range:
    # A. Get the OPTIMAL error (Eckart-Young Theorem) 
    opt_error = S_full[k]
    errors_opt.append(opt_error)
    for q_i in power_iterations_q:
        # The randSVD function MUST be updated with the 'q' parameter for this to work.
        # IMPORTANT: Explicitly specify sketch_type to test different methods
        U_r2, S_r2, V_r2_t = randSVD(A, k, oversampling_p, q=q_i, sketch_type='gaussian')
        A_approx_q2 = U_r2[:, :k] @ np.diag(S_r2[:k]) @ V_r2_t[:k, :]
        errors[q_i].append(np.linalg.norm(A - A_approx_q2, ord=2))
    
    # print(f"  k={k}: Opt={opt_error:.4e}, Rand(q=0)={errors_rand_q0[-1]:.4e}, Rand(q=2)={errors_rand_q2[-1]:.4e}")

print("Benchmark 2c complete (Power Method validated).")

# --- 4. Plot the Results (Log-Linear Plot) ---
color_map = {0: 'r', 1: 'g', 2: 'b', 3: 'm'}
plt.figure(figsize=(10, 6))
for q_i in power_iterations_q:
    plt.semilogy(k_range, errors[q_i], color_map[q_i] + ':', marker='x', alpha=0.5, label=f'Randomized SVD (q={q_i})')

plt.semilogy(k_range, errors_opt, 'b-', marker='o', label=r'Optimal (Eckart-Young: $\sigma_{k+1}$)') 
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