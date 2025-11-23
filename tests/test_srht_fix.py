import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.structured_sketch import srht_operator

# Compare Gaussian vs SRHT with the fix
m, n = 200, 100
k, p = 20, 10
l = k + p

A = np.random.randn(m, n)
U_true, s_true, Vt_true = np.linalg.svd(A, full_matrices=False)

# Test 1: Gaussian
rng1 = np.random.default_rng(seed=0)
Omega_gauss = rng1.normal(0, 1, size=(n, l))
Y_gauss = A @ Omega_gauss
Q_gauss, _ = np.linalg.qr(Y_gauss)
B_gauss = Q_gauss.T @ A
_, s_gauss, _ = np.linalg.svd(B_gauss, full_matrices=False)

print('Gaussian sketching:')
print(f'  Max error in top-{k} σ: {np.abs(s_true[:k] - s_gauss[:k]).max():.4e}')

# Test 2: SRHT with fix
Y_srht = srht_operator(A, l)
Q_srht, _ = np.linalg.qr(Y_srht)
B_srht = Q_srht.T @ A
_, s_srht, _ = np.linalg.svd(B_srht, full_matrices=False)

print('\nSRHT sketching (WITH FIX):')
print(f'  Max error in top-{k} σ: {np.abs(s_true[:k] - s_srht[:k]).max():.4e}')

error_ratio = np.abs(s_true[:k] - s_srht[:k]).max() / np.abs(s_true[:k] - s_gauss[:k]).max()
print(f'\nRatio (SRHT/Gaussian): {error_ratio:.4f}')
print('(Should be close to 1.0 for comparable accuracy!)')

if error_ratio < 1.5:
    print('\n✓ SUCCESS: SRHT and Gaussian have similar accuracy!')
else:
    print(f'\n✗ PROBLEM: SRHT is still {error_ratio:.2f}x worse than Gaussian')
