"""
Experiment 4: Comprehensive Summary Figure

This creates the main summary figure for the paper showing:
  - Complexity comparison (theoretical vs empirical)
  - Method selection guidelines
  - Key trade-offs visualization

Figures produced:
  - Fig 4: Summary panel with complexity, speedup, and recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.randsvd_algorithm import randSVD
from src.structured_sketch import srft_operator, srht_operator
from src.sparse_sketching import countsketch_operator
from scipy.sparse import random as sparse_random

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'gaussian': '#0072B2',
    'srft': '#D55E00',
    'srht': '#009E73',
    'countsketch': '#E69F00',
}


def benchmark_scaling(sizes, l, num_trials=3):
    """Benchmark how methods scale with matrix size."""
    results = {
        'gaussian': [], 'srft': [], 'srht': [],
        'countsketch_1pct': [], 'countsketch_01pct': []
    }
    
    for n in sizes:
        print(f"  n={n}: ", end="", flush=True)
        
        # Dense matrix
        np.random.seed(42)
        A_dense = np.random.randn(n, n)
        
        # Sparse matrices
        A_sparse_1pct = sparse_random(n, n, density=0.01, format='csr', random_state=42)
        A_sparse_01pct = sparse_random(n, n, density=0.001, format='csr', random_state=42)
        
        # Gaussian
        times = []
        for _ in range(num_trials):
            Omega = np.random.randn(n, l)
            t0 = time.perf_counter()
            Y = A_dense @ Omega
            times.append(time.perf_counter() - t0)
        results['gaussian'].append(np.median(times))
        print(f"G={np.median(times)*1000:.0f}ms ", end="")
        
        # SRFT
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            Y = srft_operator(A_dense, l)
            times.append(time.perf_counter() - t0)
        results['srft'].append(np.median(times))
        print(f"SRFT={np.median(times)*1000:.0f}ms ", end="")
        
        # SRHT
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            Y = srht_operator(A_dense, l)
            times.append(time.perf_counter() - t0)
        results['srht'].append(np.median(times))
        print(f"SRHT={np.median(times)*1000:.0f}ms ", end="")
        
        # CountSketch 1%
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            Y = countsketch_operator(A_sparse_1pct, l)
            times.append(time.perf_counter() - t0)
        results['countsketch_1pct'].append(np.median(times))
        print(f"CS1%={np.median(times)*1000:.0f}ms ", end="")
        
        # CountSketch 0.1%
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            Y = countsketch_operator(A_sparse_01pct, l)
            times.append(time.perf_counter() - t0)
        results['countsketch_01pct'].append(np.median(times))
        print(f"CS0.1%={np.median(times)*1000:.0f}ms")
    
    return results


def create_summary_figure(results_scaling, sizes, output_dir):
    """Create comprehensive summary figure."""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # ===== Panel A: Scaling with matrix size =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    sizes_arr = np.array(sizes)
    
    # Dense methods
    ax1.plot(sizes_arr, np.array(results_scaling['gaussian']) * 1000, 
             'o-', color=COLORS['gaussian'], linewidth=2, markersize=8, label='Gaussian (dense)')
    ax1.plot(sizes_arr, np.array(results_scaling['srft']) * 1000, 
             's-', color=COLORS['srft'], linewidth=2, markersize=8, label='SRFT (dense)')
    ax1.plot(sizes_arr, np.array(results_scaling['srht']) * 1000, 
             '^-', color=COLORS['srht'], linewidth=2, markersize=8, label='SRHT (dense)')
    
    # Sparse methods
    ax1.plot(sizes_arr, np.array(results_scaling['countsketch_1pct']) * 1000, 
             'D--', color=COLORS['countsketch'], linewidth=2, markersize=8, label='CountSketch (1% sparse)')
    ax1.plot(sizes_arr, np.array(results_scaling['countsketch_01pct']) * 1000, 
             'v:', color=COLORS['countsketch'], linewidth=2, markersize=8, label='CountSketch (0.1% sparse)')
    
    ax1.set_xlabel('Matrix size ($n$)')
    ax1.set_ylabel('Sketch time (ms)')
    ax1.set_title('(a) Scaling with Matrix Size\n$\\ell = 100$', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # ===== Panel B: Complexity comparison table =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Create table
    table_data = [
        ['Method', 'Complexity', 'Best For'],
        ['Gaussian', '$O(mn\\ell)$', 'Small ℓ, BLAS available'],
        ['SRFT', '$O(mn \\log n)$', 'Large ℓ, dense data'],
        ['SRHT', '$O(mn \\log n)$', 'Large ℓ, dense data'],
        ['CountSketch', '$O(\\mathrm{nnz}(A) \\cdot \\ell)$', 'Sparse data'],
    ]
    
    # Draw table with colored header
    cell_height = 0.12
    cell_width = [0.25, 0.35, 0.4]
    start_y = 0.85
    
    for i, row in enumerate(table_data):
        y = start_y - i * cell_height
        x = 0.0
        
        for j, (cell, width) in enumerate(zip(row, cell_width)):
            # Background color
            if i == 0:
                facecolor = '#4472C4'
                textcolor = 'white'
                fontweight = 'bold'
            elif i % 2 == 1:
                facecolor = '#D6DCE5'
                textcolor = 'black'
                fontweight = 'normal'
            else:
                facecolor = '#FFFFFF'
                textcolor = 'black'
                fontweight = 'normal'
            
            rect = FancyBboxPatch((x, y - cell_height), width, cell_height,
                                   boxstyle="square,pad=0",
                                   facecolor=facecolor, edgecolor='gray', linewidth=0.5)
            ax2.add_patch(rect)
            
            # Text (handle math mode)
            if '$' in cell:
                ax2.text(x + width/2, y - cell_height/2, cell,
                        ha='center', va='center', fontsize=10,
                        color=textcolor, fontweight=fontweight)
            else:
                ax2.text(x + width/2, y - cell_height/2, cell,
                        ha='center', va='center', fontsize=10,
                        color=textcolor, fontweight=fontweight)
            
            x += width
    
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0.2, 1.0)
    ax2.set_title('(b) Complexity Summary', fontweight='bold', pad=20)
    
    # ===== Panel C: Speedup factors =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Compute speedups relative to Gaussian
    speedup_srft = np.array(results_scaling['gaussian']) / np.array(results_scaling['srft'])
    speedup_srht = np.array(results_scaling['gaussian']) / np.array(results_scaling['srht'])
    speedup_cs_1pct = np.array(results_scaling['gaussian']) / np.array(results_scaling['countsketch_1pct'])
    speedup_cs_01pct = np.array(results_scaling['gaussian']) / np.array(results_scaling['countsketch_01pct'])
    
    ax3.plot(sizes_arr, speedup_srft, 's-', color=COLORS['srft'], 
             linewidth=2, markersize=8, label='SRFT vs Gaussian')
    ax3.plot(sizes_arr, speedup_srht, '^-', color=COLORS['srht'], 
             linewidth=2, markersize=8, label='SRHT vs Gaussian')
    ax3.plot(sizes_arr, speedup_cs_1pct, 'D--', color=COLORS['countsketch'], 
             linewidth=2, markersize=8, label='CountSketch (1%) vs Gaussian')
    ax3.plot(sizes_arr, speedup_cs_01pct, 'v:', color=COLORS['countsketch'], 
             linewidth=2, markersize=8, label='CountSketch (0.1%) vs Gaussian')
    
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1.5)
    ax3.fill_between(sizes_arr, 0, 1, alpha=0.1, color='red', label='Gaussian faster')
    ax3.fill_between(sizes_arr, 1, 100, alpha=0.1, color='green', label='Structured faster')
    
    ax3.set_xlabel('Matrix size ($n$)')
    ax3.set_ylabel('Speedup vs Gaussian')
    ax3.set_title('(c) Speedup Factors\n$\\ell = 100$', fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim([0.1, 200])
    
    # ===== Panel D: Decision flowchart =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Simple flowchart
    box_props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='navy', linewidth=2)
    decision_props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', linewidth=2)
    result_props = dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    
    # Start
    ax4.text(0.5, 0.95, 'Start', ha='center', va='center', fontsize=12, 
             fontweight='bold', bbox=box_props)
    
    # Decision 1: Sparse?
    ax4.text(0.5, 0.75, 'Is matrix sparse?\n(density < 10%)', ha='center', va='center', 
             fontsize=11, bbox=decision_props)
    ax4.annotate('', xy=(0.5, 0.88), xytext=(0.5, 0.83),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Yes -> CountSketch
    ax4.text(0.15, 0.55, 'CountSketch', ha='center', va='center', 
             fontsize=11, fontweight='bold', bbox=result_props)
    ax4.annotate('Yes', xy=(0.15, 0.67), xytext=(0.35, 0.70),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # No -> Decision 2
    ax4.text(0.75, 0.55, 'Is ℓ large?\n(ℓ > 100)', ha='center', va='center', 
             fontsize=11, bbox=decision_props)
    ax4.annotate('No', xy=(0.65, 0.67), xytext=(0.65, 0.70),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # ℓ large -> SRFT/SRHT
    ax4.text(0.55, 0.30, 'SRFT or SRHT', ha='center', va='center', 
             fontsize=11, fontweight='bold', bbox=result_props)
    ax4.annotate('Yes', xy=(0.60, 0.42), xytext=(0.70, 0.45),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # ℓ small -> Gaussian
    ax4.text(0.90, 0.30, 'Gaussian', ha='center', va='center', 
             fontsize=11, fontweight='bold', bbox=result_props)
    ax4.annotate('No', xy=(0.85, 0.42), xytext=(0.80, 0.45),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Title
    ax4.set_title('(d) Method Selection Guide', fontweight='bold', pad=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0.15, 1.0)
    
    plt.savefig(output_dir / 'fig4_summary.pdf')
    plt.savefig(output_dir / 'fig4_summary.png')
    print(f"\n✓ Saved: {output_dir / 'fig4_summary.pdf'}")
    
    return fig


def main():
    print("="*70)
    print("EXPERIMENT 4: Comprehensive Summary")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Benchmark scaling
    print("\n[1/2] Benchmarking scaling behavior...")
    sizes = [256, 512, 1024, 2048, 4096]
    results_scaling = benchmark_scaling(sizes, l=100)
    
    # Create figure
    print("\n[2/2] Creating summary figure...")
    fig = create_summary_figure(results_scaling, sizes, output_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("""
Generated Figures:
  - fig1_speed_dense.pdf   : Dense matrix speed comparison
  - fig2_speed_sparse.pdf  : Sparse matrix speed comparison  
  - fig3_accuracy.pdf      : Accuracy analysis
  - fig4_summary.pdf       : Comprehensive summary

Key Takeaways:
  1. BLAS makes Gaussian competitive for practical dense matrices
  2. CountSketch provides dramatic wins for sparse data
  3. All methods achieve equivalent accuracy
  4. Power iterations (q) and oversampling (p) control accuracy
""")


if __name__ == "__main__":
    main()
