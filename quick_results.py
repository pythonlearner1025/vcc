#!/usr/bin/env python3
"""
Quick results visualization for HepG2 experiment.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def find_latest_results():
    """Find the most recent results file."""
    result_files = glob.glob("experiments/hepg2_simple_results_*.json")
    if not result_files:
        print("No results files found!")
        return None
    
    latest_file = max(result_files, key=lambda x: x.split('_')[-1].replace('.json', ''))
    print(f"Loading: {latest_file}")
    return latest_file

def plot_quick_results():
    """Generate a quick results plot."""
    results_file = find_latest_results()
    if not results_file:
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    latent_dims = results['latent_dims']
    n_params = results['n_parameters']
    r2_scores = [results['final_metrics'][str(dim)]['cell_r2_mean'] for dim in latent_dims]
    pearson_scores = [results['final_metrics'][str(dim)]['cell_pearson_mean'] for dim in latent_dims]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: R² vs Latent Dimension
    axes[0].plot(latent_dims, r2_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Reconstruction Quality')
    axes[0].grid(True)
    axes[0].set_ylim(0, max(r2_scores) * 1.1)
    
    # Plot 2: Parameters vs Latent Dimension
    axes[1].plot(latent_dims, [p/1e6 for p in n_params], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].set_title('Model Size')
    axes[1].grid(True)
    
    # Plot 3: Efficiency (R² per Million Parameters)
    efficiency = [r2_scores[i] / (n_params[i]/1e6) for i in range(len(latent_dims))]
    axes[2].plot(latent_dims, efficiency, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('R² per Million Parameters')
    axes[2].set_title('Efficiency')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"experiments/hepg2_quick_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Quick results plot saved: {plot_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("QUICK RESULTS SUMMARY")
    print("="*50)
    print(f"{'Latent':>6} {'Params(M)':>9} {'R²':>6} {'Pearson':>7} {'Efficiency':>10}")
    print("-" * 50)
    
    for i, dim in enumerate(latent_dims):
        print(f"{dim:6d} {n_params[i]/1e6:9.1f} {r2_scores[i]:6.4f} "
              f"{pearson_scores[i]:7.4f} {efficiency[i]:10.2f}")
    
    # Find best
    best_r2_idx = np.argmax(r2_scores)
    best_eff_idx = np.argmax(efficiency)
    
    print("\nKey Findings:")
    print(f"- Best R²: {latent_dims[best_r2_idx]} ({r2_scores[best_r2_idx]:.4f})")
    print(f"- Most Efficient: {latent_dims[best_eff_idx]} ({efficiency[best_eff_idx]:.2f})")

if __name__ == '__main__':
    plot_quick_results()
