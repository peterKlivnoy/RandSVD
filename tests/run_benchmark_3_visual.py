import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Path Hack ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End Path Hack ---

from src.randsvd_algorithm import randSVD

print("Running Benchmark 3: Visual Proof (Image Compression with multiple k)")

# --- Parameters ---
k_list = [10, 40, 80]   # The list of ranks we want to compare
p_fixed = 10           # Oversampling
# Build robust paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
image_path = os.path.join(DATA_DIR, 'test_image.png') # Assumes test_image.png is in data/

# --- 1. Load and Prepare Image ---
try:
    img = mpimg.imread(image_path)
except FileNotFoundError:
    print(f"Error: Test image not found at {image_path}")
    print("Please add a 'test_image.png' to your 'data/' folder to run this.")
    sys.exit()

# Convert to grayscale by averaging color channels
# SVD works on 2D matrices [2, 3]
if img.ndim == 3:
    A = np.mean(img, axis=2)
else:
    A = img

print(f"Loaded image, shape: {A.shape}")

# --- 2. Create the Figure Grid ---
# We'll create a 2x4 grid:
# Row 1: Original | Optimal k=10 | Optimal k=40 | Optimal k=80
# Row 2: Original | RandSVD k=10 | RandSVD k=40 | RandSVD k=80
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
plt.suptitle(f'Image Compression vs. Rank (k)', fontsize=20)

# --- 3. Plot the Original Image in the first column ---
for i in range(2):
    axes[i, 0].imshow(A, cmap='gray')
    axes[i, 0].set_title(f'Original (Shape: {A.shape})')
    axes[i, 0].axis('off')

# --- 4. Loop over k, compute SVDs, and plot ---
print("Computing SVD and RandSVD for k=10, 40, 80...")
for i, k in enumerate(k_list):
    col = i + 1 # Start plotting in the second column (index 1)
    
    # --- A. Optimal Reconstruction (Standard SVD) ---
    U, S, Vt = np.linalg.svd(A, full_matrices=False) 
    A_std = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # --- B. Randomized Reconstruction (randSVD) ---
    U_r, S_r, Vt_r = randSVD(A, k, p_fixed)
    A_rand = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]

    # --- C. Plot the results ---
    axes[0, col].imshow(A_std, cmap='gray')
    axes[0, col].set_title(f'Optimal SVD (k={k})')
    axes[0, col].axis('off')
    
    axes[1, col].imshow(A_rand, cmap='gray')
    axes[1, col].set_title(f'Randomized SVD (k={k})')
    axes[1, col].axis('off')

print("Reconstructions complete.")

# --- 5. Save and Show Figure ---
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
save_path = os.path.join(FIGURE_DIR, 'benchmark_3_visual.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.savefig(save_path)
print(f"Saved visual plot to {save_path}")
plt.show()