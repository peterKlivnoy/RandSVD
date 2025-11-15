import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Path Hack ---
# This adds the project root (RANDSVD) to Python's path
# so we can import from the 'src' folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Hack ---

# Now this import will work:
from src.randsvd_algorithm import randSVD, randSVD_HH

print("Running Benchmark 1: Computational Speed (Dense Matrices)")

# --- Parameters ---
k_fixed = 50  # A fixed target rank
p_fixed = 10  # A fixed oversampling parameter [2]
# Define the range of matrix sizes (N x N)
N_range = np.arange(200, 2001, 200) # From 200x200 to 2000x2000
num_trials = 3 # Average over 3 runs to smooth out system noise

times_np = []
times_rand = []
times_rand_hh = []

# --- Run Benchmark ---
for N in N_range:
    print(f"  Testing dense matrix size: {N}x{N}...")
    
    t_np_avg = 0
    t_rand_avg = 0
    t_rand_hh_avg = 0
    
    
    for _ in range(num_trials):
        # 1. Create a dense N x N matrix
        A = np.random.randn(N, N)

        # 2. Time the standard numpy.linalg.svd
        start = time.perf_counter()
        _ = np.linalg.svd(A, full_matrices=False)
        t_np_avg += (time.perf_counter() - start)

        # 3. Time your randSVD
        start = time.perf_counter()
        _ = randSVD(A, k_fixed, p_fixed)
        t_rand_avg += (time.perf_counter() - start)
        # 4. Time your randSVD_HH
        start = time.perf_counter()
        _ = randSVD_HH(A, k_fixed, p_fixed)
        t_rand_hh_avg += (time.perf_counter() - start)

    # Append the average time
    times_np.append(t_np_avg / num_trials)
    times_rand.append(t_rand_avg / num_trials)
    times_rand_hh.append(t_rand_hh_avg / num_trials)
print("Benchmark 1 complete.")

# --- Plot Results ---
plt.figure(figsize=(10, 6))
# A log-log plot shows the exponent as the slope [4, 5, 6, 1]
plt.loglog(N_range, times_np, 'b-o', label='numpy.linalg.svd (O(N^3))') 
plt.loglog(N_range, times_rand, 'r-x', label=f'randSVD (k={k_fixed}, O(N^2))')
plt.loglog(N_range, times_rand_hh, 'm-^', label=f'randSVD_HH (k={k_fixed}, O(N^2))')
plt.xlabel('Matrix Size (N) for N x N matrix')
plt.ylabel('Wall-Clock Time (seconds)')
plt.title('Benchmark 1: Speed vs. Matrix Size (Dense) - Log-Log Plot') 
plt.grid(True, which="both", ls="--")
plt.legend()

# --- Save Figure ---
# Use the PROJECT_ROOT to build a robust path to the 'figures' dir
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

save_path = os.path.join(FIGURE_DIR, 'benchmark_1_speed.png')
plt.savefig(save_path)
print(f"Saved speed plot to {save_path}")

plt.show()
