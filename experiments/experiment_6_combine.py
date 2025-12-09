#!/usr/bin/env python3
"""
Experiment 6 - Combine: Generate figure from saved results.
Based on Musco & Musco (2015)

Run this after experiment_6_part1_newsgroups.py and experiment_6_part2_synthetic.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_figure_paper_style(results_by_config, config_labels, q_values, output_dir):
    """
    Create figure in the style of Musco & Musco (2015).
    
    Each subplot shows ONE dataset with ALL 6 lines:
    - Block Krylov: Frobenius, Spectral, Per-Vector
    - Simult. Iter.: Frobenius, Spectral, Per-Vector
    """
    
    n_configs = len(results_by_config)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4))
    
    if n_configs == 1:
        axes = [axes]
    
    # Colors matching the paper style
    # Block Krylov: green/yellow shades
    # Simultaneous: blue shades
    colors = {
        ('krylov', 'frob'): '#2ca02c',      # Green
        ('krylov', 'spec'): '#98df8a',      # Light green
        ('krylov', 'pervec'): '#bcbd22',    # Yellow-green
        ('simul', 'frob'): '#1f77b4',       # Blue
        ('simul', 'spec'): '#aec7e8',       # Light blue
        ('simul', 'pervec'): '#17becf',     # Cyan
    }
    
    markers = {
        ('krylov', 'frob'): 'o',
        ('krylov', 'spec'): 'o',
        ('krylov', 'pervec'): '^',
        ('simul', 'frob'): 'o',
        ('simul', 'spec'): 'o',
        ('simul', 'pervec'): '^',
    }
    
    metric_display = {
        'frob': 'Frobenius Error',
        'spec': 'Spectral Error',
        'pervec': 'Per Vector Error'
    }
    
    method_display = {
        'krylov': 'Block Krylov',
        'simul': 'Simult. Iter.'
    }
    
    configs = list(results_by_config.keys())
    
    for col, config in enumerate(configs):
        ax = axes[col]
        results = results_by_config[config]
        
        # Find max value for y-axis scaling (excluding Frobenius for newsgroups)
        all_vals = []
        
        # Plot all 6 lines
        for method in ['krylov', 'simul']:
            for metric in ['frob', 'spec', 'pervec']:
                vals = results[method][metric]
                # Clip negative values to small positive for display
                vals = [max(v, 0.001) for v in vals]
                
                # Skip Frobenius for newsgroups (it's off scale due to heavy tail)
                if config == 'newsgroups' and metric == 'frob':
                    continue
                
                all_vals.extend(vals)
                
                label = f"{method_display[method]} – {metric_display[metric]}"
                ax.plot(q_values, vals,
                       marker=markers[(method, metric)], 
                       linewidth=1.5, markersize=4,
                       color=colors[(method, metric)], 
                       label=label)
        
        ax.set_xlabel('Iterations q')
        ax.set_ylabel('Error ε')
        ax.set_title(config_labels[config].replace('\n', ', '))
        
        # Set y-axis to start at 0, with reasonable max
        ax.set_ylim(bottom=0, top=min(max(all_vals) * 1.1, 2.5) if all_vals else 1)
        ax.set_xlim(left=0, right=max(q_values))
        
        # Legend
        ax.legend(loc='upper right', fontsize=6, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'fig6_krylov.pdf')
    plt.savefig(output_dir / 'fig6_krylov.png', dpi=150)
    plt.close()
    
    return output_dir / 'fig6_krylov.pdf'


def main():
    print("=" * 70)
    print("EXPERIMENT 6 - COMBINE: Generate Figure from Saved Results")
    print("=" * 70)
    
    data_dir = Path(__file__).parent.parent / 'data'
    fig_dir = Path(__file__).parent.parent / 'figures'
    
    results_by_config = {}
    config_labels = {}
    q_values = None
    
    # Load Part 1 (20 Newsgroups)
    news_file = data_dir / 'exp6_newsgroups_results.pkl'
    if news_file.exists():
        print(f"\n✓ Loading {news_file.name}...")
        with open(news_file, 'rb') as f:
            news_data = pickle.load(f)
        results_by_config['newsgroups'] = news_data['results']
        config_labels['newsgroups'] = news_data['config_label']
        q_values = news_data['q_values']
        print(f"  Matrix: {news_data['matrix_shape']}")
    else:
        print(f"\n✗ {news_file.name} not found - run experiment_6_part1_newsgroups.py first")
    
    # Load Part 2 (Synthetic)
    synth_file = data_dir / 'exp6_synthetic_results.pkl'
    if synth_file.exists():
        print(f"\n✓ Loading {synth_file.name}...")
        with open(synth_file, 'rb') as f:
            synth_data = pickle.load(f)
        results_by_config.update(synth_data['results'])
        config_labels.update(synth_data['labels'])
        if q_values is None:
            q_values = synth_data['q_values']
        print(f"  Matrix: {synth_data['matrix_size']}")
    else:
        print(f"\n✗ {synth_file.name} not found - run experiment_6_part2_synthetic.py first")
    
    if not results_by_config:
        print("\nERROR: No results files found. Run the individual experiment parts first.")
        return
    
    # Reorder: newsgroups first if present
    ordered_configs = {}
    ordered_labels = {}
    for key in ['newsgroups', 'synth_heavy', 'synth_zipf']:
        if key in results_by_config:
            ordered_configs[key] = results_by_config[key]
            ordered_labels[key] = config_labels[key]
    
    print(f"\nGenerating figure with {len(ordered_configs)} configuration(s)...")
    output_path = create_figure_paper_style(ordered_configs, ordered_labels, q_values, fig_dir)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Saved: {output_path.with_suffix('.png')}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Block Krylov vs Simultaneous Iteration")
    print("=" * 70)
    
    for config, label in ordered_labels.items():
        print(f"\n{label.replace(chr(10), ' ')}:")
        res = ordered_configs[config]
        print(f"  At q=1: Simul PV={res['simul']['pervec'][1]:.2f} vs Krylov PV={res['krylov']['pervec'][1]:.2f}")
        if res['simul']['pervec'][1] > 0:
            ratio = res['simul']['pervec'][1] / max(res['krylov']['pervec'][1], 1e-10)
            print(f"  Krylov advantage: {ratio:.1f}× better")


if __name__ == "__main__":
    main()
