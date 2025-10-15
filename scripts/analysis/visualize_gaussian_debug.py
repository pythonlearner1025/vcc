#!/usr/bin/env python3
"""Visualization script for the saved gaussian debug data."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import glob

def visualize_debug_data(filename):
    """Visualize the saved debug data for top 16 genes."""
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nLoaded debug data from {filename}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Number of replicates (S): {data['S_replicates']}")
    print(f"Total number of genes (N): {data['N_genes']}")
    print(f"Sigma floor: {data['sigma_floor']}")
    ranking_method = data.get('ranking_method', 'max_abs_delta')  # Backward compatibility
    print(f"Ranking method: {ranking_method}")
    
    # Create figure with subplots for top 16 genes
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    bin_centers = data['bin_centers']
    
    for i in range(min(16, len(data['top_gene_indices']))):
        ax = axes[i]
        
        # Get data for this gene
        gene_idx = data['top_gene_indices'][i]
        delta_dist = data['delta_distributions'][:, i]  # S replicates for this gene
        mu = data['mu_values'][i]
        abs_mu = data.get('abs_mu_values', np.abs(data['mu_values']))[i] if 'abs_mu_values' in data else np.abs(mu)
        sigma_unclamped = data['sigma_unclamped'][i]
        sigma_clamped = data['sigma_clamped'][i]
        max_abs_delta = data['max_abs_delta'][i]
        
        # Plot histogram of actual delta values
        ax.hist(delta_dist, bins=30, density=True, alpha=0.5, label='Actual deltas', color='blue')
        
        # Plot fitted Gaussian using clamped sigma
        x_range = np.linspace(delta_dist.min() - 1, delta_dist.max() + 1, 200)
        gaussian_pdf = norm.pdf(x_range, mu, sigma_clamped)
        ax.plot(x_range, gaussian_pdf, 'r-', linewidth=2, label=f'Gaussian (σ={sigma_clamped:.3f})')
        
        # If sigma was clamped, also show unclamped Gaussian
        if abs(sigma_clamped - sigma_unclamped) > 1e-6:
            gaussian_pdf_unclamped = norm.pdf(x_range, mu, sigma_unclamped)
            ax.plot(x_range, gaussian_pdf_unclamped, 'g--', linewidth=1, 
                   label=f'Unclamped (σ={sigma_unclamped:.3f})', alpha=0.7)
        
        # Mark bin centers with vertical lines
        for bc in bin_centers[::10]:  # Show every 10th bin center to avoid clutter
            ax.axvline(bc, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Add vertical line at mean
        ax.axvline(mu, color='black', linestyle='--', alpha=0.7, label=f'μ={mu:.3f}')
        
        ax.set_title(f'Gene {gene_idx} (|μ|={abs_mu:.3f}, max|Δ|={max_abs_delta:.3f})', fontsize=10)
        ax.set_xlabel('Delta value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Top 16 Genes - Delta Distributions and Gaussian Fits\n{filename}', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    output_file = filename.replace('.pkl', '_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics for Top 16 Genes:")
    print("="*75)
    print(f"{'Gene':<8} {'|μ|':<10} {'μ':<10} {'Max|Δ|':<10} {'σ_orig':<10} {'σ_clamp':<10} {'Clamped?':<10}")
    print("-"*75)
    for i in range(min(16, len(data['top_gene_indices']))):
        gene_idx = data['top_gene_indices'][i]
        abs_mu = data.get('abs_mu_values', np.abs(data['mu_values']))[i] if 'abs_mu_values' in data else np.abs(data['mu_values'][i])
        was_clamped = abs(data['sigma_clamped'][i] - data['sigma_unclamped'][i]) > 1e-6
        print(f"{gene_idx:<8} {abs_mu:<10.4f} {data['mu_values'][i]:<10.4f} {data['max_abs_delta'][i]:<10.4f} "
              f"{data['sigma_unclamped'][i]:<10.4f} {data['sigma_clamped'][i]:<10.4f} "
              f"{'Yes' if was_clamped else 'No':<10}")

def main():
    # Find the most recent debug file
    debug_files = glob.glob('debug_gaussian/top16_genes_*.pkl')
    
    if not debug_files:
        print("No debug files found in debug_gaussian/ directory")
        print("Run training first to generate debug data.")
        return
    
    # Sort by modification time and get the most recent
    debug_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = debug_files[0]
    
    print(f"Found {len(debug_files)} debug file(s)")
    print(f"Visualizing most recent: {latest_file}")
    for file in debug_files: 
        visualize_debug_data(file)

if __name__ == "__main__":
    main()