#!/usr/bin/env python3
"""
Simple script to view results from a single experiment JSON file.
"""

import json
import sys
import matplotlib.pyplot as plt
import pandas as pd

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def print_summary(results):
    """Print a summary of the experiment results."""
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    for i, latent_dim in enumerate(results['latent_dims']):
        metrics = results['best_metrics'][str(latent_dim)]
        print(f"\nLatent Dimension: {latent_dim}")
        print(f"  Parameters: {results['n_parameters'][i]:,}")
        print(f"  Final Loss: {results['final_losses'][i]:.4f}")
        print(f"  Cell R²: {metrics['cell_r2_mean']:.4f} ± {metrics['cell_r2_std']:.4f}")
        print(f"  Cell Pearson: {metrics['cell_pearson_mean']:.4f} ± {metrics['cell_pearson_std']:.4f}")
        print(f"  Cell Spearman: {metrics['cell_spearman_mean']:.4f} ± {metrics['cell_spearman_std']:.4f}")
        print(f"  Cell MSE: {metrics['cell_mse_mean']:.4f} ± {metrics['cell_mse_std']:.4f}")
        print(f"  Cell Cosine Sim: {metrics['cell_cosine_sim_mean']:.4f} ± {metrics['cell_cosine_sim_std']:.4f}")
        print(f"  Gene R²: {metrics['gene_r2_mean']:.4f} ± {metrics['gene_r2_std']:.4f}")
        print(f"  Gene Pearson: {metrics['gene_pearson_mean']:.4f} ± {metrics['gene_pearson_std']:.4f}")

def create_visualization(results, save_path=None):
    """Create visualization plots for the results."""
    
    if len(results['latent_dims']) == 1:
        # Single experiment - show metric distributions
        latent_dim = results['latent_dims'][0]
        metrics = results['best_metrics'][str(latent_dim)]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Cell-wise metrics
        cell_metrics = ['cell_r2', 'cell_pearson', 'cell_spearman', 'cell_mse', 'cell_mae', 'cell_cosine_sim']
        titles = ['Cell R²', 'Cell Pearson r', 'Cell Spearman r', 'Cell MSE', 'Cell MAE', 'Cell Cosine Similarity']
        
        for i, (metric, title) in enumerate(zip(cell_metrics, titles)):
            row, col = i // 3, i % 3
            mean_val = metrics[f'{metric}_mean']
            std_val = metrics[f'{metric}_std']
            
            # Create a simple bar plot with error bars
            axes[row, col].bar([latent_dim], [mean_val], yerr=[std_val], capsize=5, alpha=0.7)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Latent Dimension')
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add text annotation with the value
            axes[row, col].text(latent_dim, mean_val + std_val/2, f'{mean_val:.3f}±{std_val:.3f}', 
                              ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'HepG2 VAE Reconstruction Metrics (Latent Dim = {latent_dim})', fontsize=16)
        plt.tight_layout()
        
    else:
        # Multiple experiments - comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        latent_dims = results['latent_dims']
        
        # Loss vs Latent Dimension
        axes[0, 0].plot(latent_dims, results['final_losses'], 'bo-')
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Final Validation Loss')
        axes[0, 0].set_title('Reconstruction Loss vs Latent Dimension')
        axes[0, 0].grid(True)
        
        # Parameters vs Latent Dimension
        axes[0, 1].plot(latent_dims, results['n_parameters'], 'ro-')
        axes[0, 1].set_xlabel('Latent Dimension')
        axes[0, 1].set_ylabel('Number of Parameters')
        axes[0, 1].set_title('Model Size vs Latent Dimension')
        axes[0, 1].grid(True)
        
        # R² vs Latent Dimension
        r2_means = [results['best_metrics'][str(dim)]['cell_r2_mean'] for dim in latent_dims]
        r2_stds = [results['best_metrics'][str(dim)]['cell_r2_std'] for dim in latent_dims]
        axes[0, 2].errorbar(latent_dims, r2_means, yerr=r2_stds, fmt='go-', capsize=5)
        axes[0, 2].set_xlabel('Latent Dimension')
        axes[0, 2].set_ylabel('Cell-wise R²')
        axes[0, 2].set_title('Reconstruction R² vs Latent Dimension')
        axes[0, 2].grid(True)
        
        # Pearson Correlation vs Latent Dimension
        pearson_means = [results['best_metrics'][str(dim)]['cell_pearson_mean'] for dim in latent_dims]
        pearson_stds = [results['best_metrics'][str(dim)]['cell_pearson_std'] for dim in latent_dims]
        axes[1, 0].errorbar(latent_dims, pearson_means, yerr=pearson_stds, fmt='mo-', capsize=5)
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Cell-wise Pearson r')
        axes[1, 0].set_title('Reconstruction Correlation vs Latent Dimension')
        axes[1, 0].grid(True)
        
        # MSE vs Latent Dimension
        mse_means = [results['best_metrics'][str(dim)]['cell_mse_mean'] for dim in latent_dims]
        mse_stds = [results['best_metrics'][str(dim)]['cell_mse_std'] for dim in latent_dims]
        axes[1, 1].errorbar(latent_dims, mse_means, yerr=mse_stds, fmt='co-', capsize=5)
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Cell-wise MSE')
        axes[1, 1].set_title('Reconstruction MSE vs Latent Dimension')
        axes[1, 1].grid(True)
        
        # Efficiency (R² / Parameters)
        efficiency = [r2_means[i] / results['n_parameters'][i] * 1e6 for i in range(len(latent_dims))]
        axes[1, 2].plot(latent_dims, efficiency, 'yo-')
        axes[1, 2].set_xlabel('Latent Dimension')
        axes[1, 2].set_ylabel('Efficiency (R² / Params × 10⁶)')
        axes[1, 2].set_title('Reconstruction Efficiency vs Latent Dimension')
        axes[1, 2].grid(True)
        
        plt.suptitle('HepG2 VAE Latent Dimension Comparison', fontsize=16)
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python view_single_result.py <results.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        results = load_results(json_path)
        print_summary(results)
        
        # Create plot
        save_path = json_path.replace('.json', '_visualization.png')
        create_visualization(results, save_path)
        
    except Exception as e:
        print(f"Error processing results: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
