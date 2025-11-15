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

print("Running Benchmark 2b: Accuracy vs. Optimal (Slow Decay)")

# --- 1. Create Test Matrix with SLOW Decaying Spectrum ---
m, n = 2000, 1000
print(f"Creating {m}x{n} test matrix with (slow) polynomial spectrum...")

# Create random orthogonal matrices U and V
U_full, _ = np.linalg.qr(np.random.randn(m, m))
U = U_full[:, :n]

V_full, _ = np.linalg.qr(np.random.randn(n, n))
V_t = V_full.T

# --- THIS IS THE ONLY CHANGE ---
# Create a known, POLYNOMIALLY decaying spectrum (sigma_i = 1/i)
# This is a much harder test for the algorithm 
S_full = 1.0 / (np.arange(n) + 1)
# --- END OF CHANGE ---

Sigma_diag = np.diag(S_full)

# Assemble the "ground truth" matrix A = U * Sigma * V^T [5, 2]
A = U @ Sigma_diag @ V_t
print("Test matrix A (slow decay) created.")

# --- 2. Set Benchmark Parameters ---
k_range = np.arange(5, 101, 5)
oversampling_p = 10
errors_rand = []
errors_opt = []
print(f"Testing ranks k={list(k_range)}...")

# --- 3. Loop Over Ranks and Calculate Errors ---
for k in k_range:
    # A. Get the OPTIMAL error (Eckart-Young Theorem) 
    opt_error = S_full[k]
    errors_opt.append(opt_error)
    
    # B. Get the RANDOMIZED error (Your Function)
    U_r, S_r, V_r_t = randSVD(A, k, oversampling_p)
    
    # Reconstruct the rank-k approximation
    A_approx = U_r[:, :k] @ np.diag(S_r[:k]) @ V_r_t[:k, :]
    
    # Calculate the spectral norm of the error
    rand_error = np.linalg.norm(A - A_approx, ord=2)
    errors_rand.append(rand_error)
    
    print(f"  k={k}: Optimal Error = {opt_error:.4e}, RandSVD Error = {rand_error:.4e}")

print("Benchmark 2b complete.")

# --- 4. Plot the Results (Log-Linear Plot) ---
plt.figure(figsize=(10, 6))
plt.semilogy(k_range, errors_opt, 'b-', marker='o', label='Optimal (Eckart-Young: $\sigma_{k+1}$)') 
plt.semilogy(k_range, errors_rand, 'r--', marker='x', label=f'Randomized SVD (p={oversampling_p})')
plt.xlabel('Target Rank (k)')
plt.ylabel('Approximation Error (Spectral Norm)')
plt.title('Benchmark 2b: Accuracy vs. Optimum (Slow Decay)')
plt.legend()
plt.grid(True, which="both", ls="--")

# --- 5. Save Figure ---
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
save_path = os.path.join(FIGURE_DIR, 'benchmark_2b_accuracy_slow.png') # New file name
plt.savefig(save_path)
print(f"Saved accuracy plot to {save_path}")
plt.show()