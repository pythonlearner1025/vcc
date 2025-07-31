#!/usr/bin/env python3
"""
HepG2 VAE Experiment Analysis

This script analyzes the results from the HepG2 VAE experiments and generates
comprehensive analysis plots and reports.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_experiment_results(experiments_dir: str = "experiments") -> Dict:
    """Load results from completed experiments."""
    
    # Find all result JSON files
    result_files = glob.glob(os.path.join(experiments_dir, "hepg2_latent_comparison_*.json"))
    
    if not result_files:
        print("No experiment result files found!")
        return None
    
    # Load the most recent results
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def load_tensorboard_data(log_dir: str) -> Dict:
    """Load data from TensorBoard logs."""
    
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def analyze_training_curves(experiments_dir: str = "experiments"):
    """Analyze training curves from all experiments."""
    
    # Find all experiment directories
    exp_dirs = [d for d in glob.glob(os.path.join(experiments_dir, "hepg2_latent_*")) 
                if os.path.isdir(d) and not d.endswith('.json')]
    
    if not exp_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Load training curves for each experiment
    training_data = {}
    
    for exp_dir in exp_dirs:
        # Extract latent dimension from directory name
        dir_name = os.path.basename(exp_dir)
        if 'latent_' in dir_name:
            try:
                latent_dim = int(dir_name.split('latent_')[1].split('_')[0])
            except:
                print(f"Could not extract latent dim from {dir_name}")
                continue
        else:
            continue
        
        try:
            tb_data = load_tensorboard_data(exp_dir)
            training_data[latent_dim] = tb_data
            print(f"Loaded training data for latent_dim={latent_dim}")
        except Exception as e:
            print(f"Failed to load data from {exp_dir}: {e}")
    
    if not training_data:
        print("No training data loaded!")
        return
    
    # Create training curve plots
    plot_training_curves(training_data, experiments_dir)


def plot_training_curves(training_data: Dict, save_dir: str):
    """Plot training curves for all experiments."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    latent_dims = sorted(training_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(latent_dims)))
    
    # Plot 1: Total Loss
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Loss/Val_total_loss' in data:
            steps = data['Loss/Val_total_loss']['steps']
            values = data['Loss/Val_total_loss']['values']
            axes[0, 0].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Total Loss')
    axes[0, 0].set_title('Training Progress: Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Reconstruction Loss
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Loss/Val_reconstruction_loss' in data:
            steps = data['Loss/Val_reconstruction_loss']['steps']
            values = data['Loss/Val_reconstruction_loss']['values']
            axes[0, 1].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Reconstruction Loss')
    axes[0, 1].set_title('Training Progress: Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: KLD Loss
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Loss/Val_kld_loss' in data:
            steps = data['Loss/Val_kld_loss']['steps']
            values = data['Loss/Val_kld_loss']['values']
            axes[0, 2].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Validation KLD Loss')
    axes[0, 2].set_title('Training Progress: KLD Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: R² Score
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Metrics/Val_cell_r2_mean' in data:
            steps = data['Metrics/Val_cell_r2_mean']['steps']
            values = data['Metrics/Val_cell_r2_mean']['values']
            axes[1, 0].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation R² Score')
    axes[1, 0].set_title('Training Progress: R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Pearson Correlation
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Metrics/Val_cell_pearson_mean' in data:
            steps = data['Metrics/Val_cell_pearson_mean']['steps']
            values = data['Metrics/Val_cell_pearson_mean']['values']
            axes[1, 1].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Pearson Correlation')
    axes[1, 1].set_title('Training Progress: Pearson Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Learning Rate
    for i, latent_dim in enumerate(latent_dims):
        data = training_data[latent_dim]
        if 'Learning_Rate' in data:
            steps = data['Learning_Rate']['steps']
            values = data['Learning_Rate']['values']
            axes[1, 2].plot(steps, values, color=colors[i], label=f'Latent {latent_dim}')
    
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Training Progress: Learning Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, "hepg2_training_curves_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves analysis saved to: {plot_path}")


def generate_comprehensive_report(results: Dict, save_dir: str):
    """Generate a comprehensive analysis report."""
    
    if not results:
        print("No results to analyze!")
        return
    
    report_path = os.path.join(save_dir, "hepg2_experiment_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# HepG2 Control Cell VAE Experiment Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Overview\n\n")
        f.write("This experiment evaluates the reconstruction efficiency of a Variational Autoencoder (VAE) ")
        f.write("on HepG2 non-targeting control cells across different latent space dimensions.\n\n")
        
        f.write("## Results Summary\n\n")
        
        # Create results table
        f.write("| Latent Dim | Parameters | Final Loss | R² Mean | R² Std | Pearson Mean | Pearson Std |\n")
        f.write("|------------|------------|------------|---------|--------|--------------|-------------|\n")
        
        for i, latent_dim in enumerate(results['latent_dims']):
            metrics = results['best_metrics'][str(latent_dim)]
            f.write(f"| {latent_dim} | {results['n_parameters'][i]:,} | ")
            f.write(f"{results['final_losses'][i]:.2f} | ")
            f.write(f"{metrics['cell_r2_mean']:.4f} | ")
            f.write(f"{metrics['cell_r2_std']:.4f} | ")
            f.write(f"{metrics['cell_pearson_mean']:.4f} | ")
            f.write(f"{metrics['cell_pearson_std']:.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best performing models
        r2_scores = [results['best_metrics'][str(dim)]['cell_r2_mean'] for dim in results['latent_dims']]
        best_r2_idx = np.argmax(r2_scores)
        best_r2_dim = results['latent_dims'][best_r2_idx]
        
        pearson_scores = [results['best_metrics'][str(dim)]['cell_pearson_mean'] for dim in results['latent_dims']]
        best_pearson_idx = np.argmax(pearson_scores)
        best_pearson_dim = results['latent_dims'][best_pearson_idx]
        
        # Calculate efficiency (R² per parameter)
        efficiencies = []
        for i, dim in enumerate(results['latent_dims']):
            r2 = results['best_metrics'][str(dim)]['cell_r2_mean']
            params = results['n_parameters'][i]
            efficiency = r2 / params * 1e6  # Scale for readability
            efficiencies.append(efficiency)
        
        best_efficiency_idx = np.argmax(efficiencies)
        best_efficiency_dim = results['latent_dims'][best_efficiency_idx]
        
        f.write(f"### Best Reconstruction Quality (R²)\n")
        f.write(f"- **Latent Dimension**: {best_r2_dim}\n")
        f.write(f"- **R² Score**: {r2_scores[best_r2_idx]:.4f}\n")
        f.write(f"- **Parameters**: {results['n_parameters'][best_r2_idx]:,}\n\n")
        
        f.write(f"### Best Correlation (Pearson)\n")
        f.write(f"- **Latent Dimension**: {best_pearson_dim}\n")
        f.write(f"- **Pearson Correlation**: {pearson_scores[best_pearson_idx]:.4f}\n")
        f.write(f"- **Parameters**: {results['n_parameters'][best_pearson_idx]:,}\n\n")
        
        f.write(f"### Best Efficiency (R² per Parameter)\n")
        f.write(f"- **Latent Dimension**: {best_efficiency_dim}\n")
        f.write(f"- **Efficiency**: {efficiencies[best_efficiency_idx]:.2f}\n")
        f.write(f"- **R² Score**: {results['best_metrics'][str(best_efficiency_dim)]['cell_r2_mean']:.4f}\n")
        f.write(f"- **Parameters**: {results['n_parameters'][best_efficiency_idx]:,}\n\n")
        
        f.write("## Analysis\n\n")
        f.write("### Reconstruction Quality vs Model Size\n")
        f.write("The experiment shows the trade-off between model size and reconstruction quality. ")
        f.write("Key observations:\n\n")
        
        # Analyze trends
        if len(results['latent_dims']) > 1:
            r2_trend = "increasing" if r2_scores[-1] > r2_scores[0] else "decreasing"
            param_trend = "increasing" if results['n_parameters'][-1] > results['n_parameters'][0] else "decreasing"
            
            f.write(f"- R² scores are generally **{r2_trend}** with latent dimension\n")
            f.write(f"- Model parameters are **{param_trend}** with latent dimension\n")
            
            # Find diminishing returns point
            r2_improvements = [r2_scores[i+1] - r2_scores[i] for i in range(len(r2_scores)-1)]
            if len(r2_improvements) > 0:
                min_improvement_idx = np.argmin(r2_improvements)
                f.write(f"- Diminishing returns start around latent dimension {results['latent_dims'][min_improvement_idx+1]}\n")
        
        f.write("\n### Recommendations\n\n")
        
        if efficiencies:
            f.write(f"- For **efficiency**, use latent dimension **{best_efficiency_dim}**\n")
        if r2_scores:
            f.write(f"- For **best quality**, use latent dimension **{best_r2_dim}**\n")
        
        # Practical recommendations
        min_acceptable_r2 = 0.5
        acceptable_dims = [dim for i, dim in enumerate(results['latent_dims']) 
                          if results['best_metrics'][str(dim)]['cell_r2_mean'] >= min_acceptable_r2]
        
        if acceptable_dims:
            smallest_acceptable = min(acceptable_dims)
            f.write(f"- For **practical use** (R² ≥ 0.5), minimum latent dimension: **{smallest_acceptable}**\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- Training logs: Individual experiment directories\n")
        f.write("- Comparison plots: `hepg2_latent_comparison_*.png`\n")
        f.write("- Training curves: `hepg2_training_curves_analysis.png`\n")
        f.write("- This report: `hepg2_experiment_report.md`\n")
    
    print(f"Comprehensive report saved to: {report_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze HepG2 VAE experiment results')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    print("HepG2 VAE Experiment Analysis")
    print("="*40)
    
    # Load experiment results
    results = load_experiment_results(args.experiments_dir)
    
    if results:
        print(f"Loaded results for {len(results['latent_dims'])} experiments")
        
        # Generate comprehensive report
        generate_comprehensive_report(results, args.experiments_dir)
        
        # Print summary to console
        print("\nExperiment Summary:")
        print("-" * 50)
        for i, latent_dim in enumerate(results['latent_dims']):
            metrics = results['best_metrics'][str(latent_dim)]
            print(f"Latent {latent_dim:3d}: R²={metrics['cell_r2_mean']:.4f}, "
                  f"Pearson={metrics['cell_pearson_mean']:.4f}, "
                  f"Params={results['n_parameters'][i]:,}")
    
    # Analyze training curves
    print("\nAnalyzing training curves...")
    analyze_training_curves(args.experiments_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
