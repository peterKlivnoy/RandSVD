import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# --- Path Hack ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Hack ---

from src.randsvd_algorithm import randSVD
from src.structured_sketch import srht_operator # Import the SRHT function directly

print("Running Benchmark 1c: Sketching Speed (Gaussian vs. SRHT)")

# --- 1. Set Benchmark Parameters ---
# Use matrix sizes that are powers of 2 for optimal FWHT performance.
N_RANGE = [512, 1024, 2048, 4096, 8192, 16384] 
k = 50 
p = 10
l = k + p
NUM_TRIALS = 3

times_gaussian = []
times_srht = []

print(f"Testing N={N_RANGE} with fixed sketch size l={l}...")

# --- 2. Loop Over Matrix Sizes N ---
for N in N_RANGE:
    # Create a dense N x N matrix
    A = np.random.randn(N, N)
    print(f"\n--- Matrix Size N={N} ---")

    # --- Gaussian Sketch Timing ---
    start_g = time.perf_counter()
    for _ in range(NUM_TRIALS):
        # We only need the sketching time, which is Y = A @ omega
        rng = np.random.default_rng(seed=0)
        omega = rng.normal(loc=0, scale=1, size=(N, l))
        Y_g = A @ omega
    end_g = time.perf_counter()
    time_g = (end_g - start_g) / NUM_TRIALS
    times_gaussian.append(time_g)
    print(f"  Gaussian Sketch Time (O(N^2 * l)): {time_g:.6f} seconds")

    # --- SRHT Sketch Timing ---
    # Since the SRHT operator is designed to work on the m x n matrix A, 
    # we call it directly to time the O(N^2 log N) operation.
    start_s = time.perf_counter()
    for _ in range(NUM_TRIALS):
        Y_s = srht_operator(A, l)
    end_s = time.perf_counter()
    time_s = (end_s - start_s) / NUM_TRIALS
    times_srht.append(time_s)
    print(f"  SRHT Sketch Time (O(N^2 log N)): {time_s:.6f} seconds")


# --- 3. Plot the Results (Log-Log Plot) ---
plt.figure(figsize=(10, 6))
# Log-log scale to determine the empirical complexity exponent
plt.loglog(N_RANGE, times_gaussian, 'b-x', label=f'Gaussian Sketch (O(N^2 * l))') 
plt.loglog(N_RANGE, times_srht, 'r-o', label=f'SRHT Sketch (O(N^2 log N))')
plt.xlabel('Matrix Size (N) for N x N matrix (Log Scale)')
plt.ylabel('Wall-Clock Time (seconds) (Log Scale)')
plt.title('Benchmark 1c: Sketching Speed - Gaussian vs. SRHT')
plt.legend()
plt.grid(True, which="both", ls="--")

# --- 4. Save Figure ---
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
save_path = os.path.join(FIGURE_DIR, 'benchmark_1c_srht_speed.png') 
plt.savefig(save_path)
print(f"\nSaved speed plot to {save_path}")
plt.show()