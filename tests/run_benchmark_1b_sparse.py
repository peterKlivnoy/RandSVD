import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import svds

# --- Path Hack ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Hack ---

from src.randsvd_algorithm import randSVD

print("Running Benchmark 1b: Computational Speed (Sparse Matrices)")

# --- Parameters ---
k_fixed = 50
p_fixed = 10
# For sparse, we can test much larger matrices
N_range = np.arange(1000, 10001, 1000) # From 1000x1000 to 10000x10000
density = 0.01 # 1% sparse
num_trials = 3

times_scipy = []
times_rand = []

# --- Run Benchmark ---
for N in N_range:
    print(f"  Testing sparse matrix size: {N}x{N} with {density*100}% density...")
    
    t_scipy_avg = 0
    t_rand_avg = 0
    
    for _ in range(num_trials):
        # 1. Create a sparse N x N matrix
        # 'csr' format is efficient for matrix-vector products
        A_sparse = sparse_random(N, N, density=density, format='csr')

        # 2. Time the standard scipy.sparse.linalg.svds
        # This is an iterative solver (ARPACK)
        start = time.perf_counter()
        _ = svds(A_sparse, k=k_fixed)
        t_scipy_avg += (time.perf_counter() - start)

        # 3. Time your randSVD
        # Your function works because sparse A @ dense omega is defined
        start = time.perf_counter()
        _ = randSVD(A_sparse, k_fixed, p_fixed)
        t_rand_avg += (time.perf_counter() - start)

    times_scipy.append(t_scipy_avg / num_trials)
    times_rand.append(t_rand_avg / num_trials)

print("Benchmark 1b complete.")

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.loglog(N_range, times_scipy, 'g-s', label='scipy.sparse.linalg.svds (Iterative)') 
plt.loglog(N_range, times_rand, 'r-x', label=f'randSVD (k={k_fixed}, Probabilistic)') 
plt.xlabel('Matrix Size (N) for N x N matrix')
plt.ylabel('Wall-Clock Time (seconds)')
plt.title('Benchmark 1b: Speed vs. Matrix Size (Sparse) - Log-Log Plot')
plt.grid(True, which="both", ls="--")
plt.legend()

# --- Save Figure ---
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

save_path = os.path.join(FIGURE_DIR, 'benchmark_1b_sparse.png')
plt.savefig(save_path)
print(f"Saved sparse speed plot to {save_path}")

plt.show()