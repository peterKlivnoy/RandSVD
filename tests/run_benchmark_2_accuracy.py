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

print("Running Benchmark 2: Accuracy vs. Optimal (Fast Decay)")

# --- 1. Create Test Matrix with FAST Decaying Spectrum ---
m, n = 2000, 1000  # A tall matrix
print(f"Creating {m}x{n} test matrix with (fast) exponential spectrum...")

# Create random orthogonal matrices U and V
U_full, _ = np.linalg.qr(np.random.randn(m, m))
U = U_full[:, :n] # Truncate to m x n

V_full, _ = np.linalg.qr(np.random.randn(n, n))
V_t = V_full.T # V is n x n, so V_t is n x n

# Create a known, EXPONENTIALLY decaying spectrum 
# This is our "ground truth"
S_full = np.exp(-0.1 * np.arange(n)) 
Sigma_diag = np.diag(S_full)

# Assemble the "ground truth" matrix A = U * Sigma * V^T [5, 2]
A = U @ Sigma_diag @ V_t
print("Test matrix A (fast decay) created.")

# --- 2. Set Benchmark Parameters ---
k_range = np.arange(5, 101, 5) # Test ranks from 5 to 100
oversampling_p = 10
errors_rand = []
errors_opt = []

print(f"Testing ranks k={list(k_range)}...")

# --- 3. Loop Over Ranks and Calculate Errors ---
for k in k_range:
    # A. Get the OPTIMAL error (Eckart-Young Theorem) 
    # The best rank-k error is the first truncated singular value, sigma_{k+1}
    # In Python (0-indexed), this is the singular value at index k [7, 4]
    opt_error = S_full[k]
    errors_opt.append(opt_error)
    
    # B. Get the RANDOMIZED error (Your Function)
    U_r, S_r, V_r_t = randSVD(A, k, oversampling_p)
    
    # Reconstruct the rank-k approximation (we must truncate the k+p factors) [5]
    A_approx = U_r[:, :k] @ np.diag(S_r[:k]) @ V_r_t[:k, :]
    
    # Calculate the spectral norm of the error (ord=2) 
    rand_error = np.linalg.norm(A - A_approx, ord=2) 
    errors_rand.append(rand_error)
    
    print(f"  k={k}: Optimal Error = {opt_error:.4e}, RandSVD Error = {rand_error:.4e}")

print("Benchmark 2 complete.")

# --- 4. Plot the Results (Log-Linear Plot) ---
plt.figure(figsize=(10, 6))
# Use semilogy for log-linear plot 
plt.semilogy(k_range, errors_opt, 'b-', marker='o', label='Optimal (Eckart-Young: $\sigma_{k+1}$)') 
plt.semilogy(k_range, errors_rand, 'r--', marker='x', label=f'Randomized SVD (p={oversampling_p})')
plt.xlabel('Target Rank (k)')
plt.ylabel('Approximation Error (Spectral Norm)')
plt.title('Benchmark 2: Accuracy vs. Optimum (Fast Decay)') 
plt.legend()
plt.grid(True, which="both", ls="--")

# --- 5. Save Figure ---
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
save_path = os.path.join(FIGURE_DIR, 'benchmark_2_accuracy.png')
plt.savefig(save_path)
print(f"Saved accuracy plot to {save_path}")
plt.show()