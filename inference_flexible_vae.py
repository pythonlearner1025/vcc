#!/usr/bin/env python3
"""
Inference script for trained Flexible VAE models.

This script provides utilities for:
1. Loading trained models
2. Perturbation injection on new data
3. Latent space analysis and visualization
4. Batch prediction and processing
5. Model evaluation and benchmarking

Features:
- Easy model loading with configuration validation
- Batch processing for large datasets
- Comprehensive perturbation analysis
- Latent space clustering and visualization
- Export capabilities for downstream analysis

Usage:
    # Basic perturbation injection
    python inference_flexible_vae.py inject \
        --model checkpoints/phase2_best.pt \
        --data data/test_cells.npz \
        --perturbation_id 5 \
        --output results/predictions.npz

    # Latent space analysis
    python inference_flexible_vae.py analyze \
        --model checkpoints/phase1_best.pt \
        --data data/cells.npz \
        --output results/latent_analysis.png

    # Batch prediction
    python inference_flexible_vae.py predict \
        --model checkpoints/phase2_best.pt \
        --data data/large_dataset.npz \
        --batch_size 1000 \
        --output results/predictions/
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import scanpy as sc
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flexible_vae import (
    FlexibleVAE, VAEConfig, LearnedGeneEmbedding, 
    PretrainedGeneEmbedding, flexible_vae_loss
)
from dataset.flexible_dataloader import (
    FlexibleDataset, load_data_from_file
)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FlexibleVAEInference:
    """
    Inference engine for trained Flexible VAE models.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_path = model_path
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.model.eval()
        
        print(f"Model loaded successfully from: {model_path}")
        print(f"Model configuration: latent_dim={self.config.latent_dim}, input_dim={self.config.input_dim}")
    
    def _load_model(self) -> Tuple[FlexibleVAE, VAEConfig]:
        """Load model and configuration from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to load config from same directory
            config_path = Path(self.model_path).parent.parent / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config = VAEConfig(**config_dict)
            else:
                raise ValueError("No configuration found. Please provide config.yaml or use checkpoint with embedded config.")
        
        # Create gene embedding (assume learned for now)
        gene_embedding = LearnedGeneEmbedding(config.n_genes, config.target_gene_embed_dim)
        
        # Create model
        model = FlexibleVAE(config, gene_embedding).to(self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model, config
    
    def inject_perturbation(self, 
                           expression_data: np.ndarray,
                           experiment_ids: np.ndarray,
                           target_gene_ids: Union[int, List[int], np.ndarray],
                           batch_size: int = 256) -> Dict[str, np.ndarray]:
        """
        Inject perturbations into control cells.
        
        Args:
            expression_data: Control cell expression data [n_cells, n_genes]
            experiment_ids: Experiment IDs for each cell
            target_gene_ids: Target gene(s) to perturb (single ID or array)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with predictions and metadata
        """
        print(f"Injecting perturbations into {len(expression_data)} cells...")
        
        # Prepare data
        expression_tensor = torch.FloatTensor(expression_data)
        experiment_tensor = torch.LongTensor(experiment_ids)
        
        # Handle target gene IDs
        if isinstance(target_gene_ids, int):
            target_gene_ids = [target_gene_ids] * len(expression_data)
        target_tensor = torch.LongTensor(target_gene_ids)
        
        # Process in batches
        all_predictions = []
        all_latents = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(expression_data), batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, len(expression_data))
                
                batch_expression = expression_tensor[i:batch_end].to(self.device)
                batch_experiments = experiment_tensor[i:batch_end].to(self.device)
                batch_targets = target_tensor[i:batch_end].to(self.device)
                
                # Forward pass with perturbation
                outputs = self.model(
                    batch_expression, 
                    batch_experiments, 
                    target_gene_ids=batch_targets
                )
                
                all_predictions.append(outputs['reconstruction'].cpu().numpy())
                
                # Also get latent representations
                encoded = self.model.encode(batch_expression, batch_experiments)
                all_latents.append(encoded['mu'].cpu().numpy())
        
        # Combine results
        predictions = np.vstack(all_predictions)
        latents = np.vstack(all_latents)
        
        # Calculate perturbation effects
        control_outputs = self.predict_control(expression_data, experiment_ids, batch_size)
        perturbation_effects = predictions - control_outputs['predictions']
        
        results = {
            'predictions': predictions,
            'control_predictions': control_outputs['predictions'],
            'perturbation_effects': perturbation_effects,
            'latent_representations': latents,
            'target_gene_ids': np.array(target_gene_ids),
            'experiment_ids': experiment_ids
        }
        
        print(f"Perturbation injection completed!")
        print(f"Mean absolute effect: {np.mean(np.abs(perturbation_effects)):.4f}")
        print(f"Max effect: {np.max(np.abs(perturbation_effects)):.4f}")
        
        return results
    
    def predict_control(self, 
                       expression_data: np.ndarray,
                       experiment_ids: np.ndarray,
                       batch_size: int = 256) -> Dict[str, np.ndarray]:
        """
        Predict control expression (reconstruction without perturbation).
        """
        expression_tensor = torch.FloatTensor(expression_data)
        experiment_tensor = torch.LongTensor(experiment_ids)
        
        all_predictions = []
        all_latents = []
        
        with torch.no_grad():
            for i in range(0, len(expression_data), batch_size):
                batch_end = min(i + batch_size, len(expression_data))
                
                batch_expression = expression_tensor[i:batch_end].to(self.device)
                batch_experiments = experiment_tensor[i:batch_end].to(self.device)
                
                # Forward pass without perturbation
                outputs = self.model(batch_expression, batch_experiments)
                all_predictions.append(outputs['reconstruction'].cpu().numpy())
                
                # Get latent representations
                encoded = self.model.encode(batch_expression, batch_experiments)
                all_latents.append(encoded['mu'].cpu().numpy())
        
        return {
            'predictions': np.vstack(all_predictions),
            'latent_representations': np.vstack(all_latents)
        }
    
    def analyze_latent_space(self, 
                           expression_data: np.ndarray,
                           experiment_ids: np.ndarray,
                           metadata: Optional[Dict[str, np.ndarray]] = None,
                           n_clusters: int = 10,
                           output_file: str = None) -> Dict[str, np.ndarray]:
        """
        Comprehensive latent space analysis.
        """
        print("Analyzing latent space...")
        
        # Get latent representations
        results = self.predict_control(expression_data, experiment_ids)
        latents = results['latent_representations']
        
        # Apply PCA for visualization
        pca = PCA(n_components=min(50, latents.shape[1]))
        latents_pca = pca.fit_transform(latents)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latents_pca[:, :10])  # Use first 10 PCs for clustering
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: PCA colored by experiment
        scatter = axes[0, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                   c=experiment_ids, cmap='viridis', alpha=0.6, s=10)
        axes[0, 0].set_title('Latent Space - Experiments')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Plot 2: PCA colored by clusters
        scatter = axes[0, 1].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                   c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
        axes[0, 1].set_title(f'Latent Space - Clusters (k={n_clusters})')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Plot 3: Metadata coloring (if available)
        if metadata and 'cell_types' in metadata:
            scatter = axes[0, 2].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                       c=metadata['cell_types'], cmap='Set1', alpha=0.6, s=10)
            axes[0, 2].set_title('Latent Space - Cell Types')
            axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, ax=axes[0, 2])
        else:
            axes[0, 2].text(0.5, 0.5, 'No cell type\nmetadata available', 
                          ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Cell Types - N/A')
        
        # Plot 4: Explained variance
        axes[1, 0].plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
                       pca.explained_variance_ratio_[:20], 'o-')
        axes[1, 0].set_title('PCA Explained Variance')
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Explained Variance Ratio')
        axes[1, 0].grid(True)
        
        # Plot 5: Cumulative explained variance
        axes[1, 1].plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
                       np.cumsum(pca.explained_variance_ratio_[:20]), 'o-')
        axes[1, 1].set_title('Cumulative Explained Variance')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].grid(True)
        
        # Plot 6: Latent dimension histogram
        axes[1, 2].hist(latents.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Latent Space Distribution')
        axes[1, 2].set_xlabel('Latent Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Latent space analysis saved to: {output_file}")
        
        plt.show()
        
        # Return analysis results
        analysis_results = {
            'latent_representations': latents,
            'pca_components': latents_pca,
            'cluster_labels': cluster_labels,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pca_model': pca,
            'kmeans_model': kmeans
        }
        
        print(f"Latent space analysis completed!")
        print(f"First 10 PCs explain {pca.explained_variance_ratio_[:10].sum():.2%} of variance")
        print(f"Identified {n_clusters} clusters in latent space")
        
        return analysis_results
    
    def evaluate_reconstruction(self, 
                              expression_data: np.ndarray,
                              experiment_ids: np.ndarray,
                              batch_size: int = 256) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        """
        print("Evaluating reconstruction quality...")
        
        results = self.predict_control(expression_data, experiment_ids, batch_size)
        predictions = results['predictions']
        
        # Calculate metrics
        r2_scores = []
        pearson_correlations = []
        mse_scores = []
        
        for i in range(len(expression_data)):
            r2 = r2_score(expression_data[i], predictions[i])
            pearson_r, _ = pearsonr(expression_data[i], predictions[i])
            mse = mean_squared_error(expression_data[i], predictions[i])
            
            r2_scores.append(r2)
            pearson_correlations.append(pearson_r)
            mse_scores.append(mse)
        
        # Gene-wise metrics
        gene_r2_scores = []
        for gene_idx in range(expression_data.shape[1]):
            gene_r2 = r2_score(expression_data[:, gene_idx], predictions[:, gene_idx])
            gene_r2_scores.append(gene_r2)
        
        metrics = {
            'mean_cell_r2': np.mean(r2_scores),
            'std_cell_r2': np.std(r2_scores),
            'mean_cell_pearson': np.mean(pearson_correlations),
            'std_cell_pearson': np.std(pearson_correlations),
            'mean_cell_mse': np.mean(mse_scores),
            'std_cell_mse': np.std(mse_scores),
            'mean_gene_r2': np.mean(gene_r2_scores),
            'std_gene_r2': np.std(gene_r2_scores),
            'median_cell_r2': np.median(r2_scores),
            'median_gene_r2': np.median(gene_r2_scores)
        }
        
        # Print summary
        print(f"Reconstruction Evaluation Results:")
        print(f"  Mean Cell R²: {metrics['mean_cell_r2']:.4f} ± {metrics['std_cell_r2']:.4f}")
        print(f"  Mean Cell Pearson r: {metrics['mean_cell_pearson']:.4f} ± {metrics['std_cell_pearson']:.4f}")
        print(f"  Mean Gene R²: {metrics['mean_gene_r2']:.4f} ± {metrics['std_gene_r2']:.4f}")
        print(f"  Median Cell R²: {metrics['median_cell_r2']:.4f}")
        print(f"  Median Gene R²: {metrics['median_gene_r2']:.4f}")
        
        return metrics


def inject_perturbations_cli(args):
    """CLI function for perturbation injection."""
    # Load inference engine
    inference = FlexibleVAEInference(args.model, args.device)
    
    # Load data
    expression_data, experiment_ids, metadata = load_data_from_file(args.data)
    
    # Run perturbation injection
    results = inference.inject_perturbation(
        expression_data, 
        experiment_ids, 
        args.perturbation_id,
        args.batch_size
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **results)
    print(f"Results saved to: {output_path}")


def analyze_latent_cli(args):
    """CLI function for latent space analysis."""
    # Load inference engine
    inference = FlexibleVAEInference(args.model, args.device)
    
    # Load data
    expression_data, experiment_ids, metadata = load_data_from_file(args.data)
    
    # Run analysis
    results = inference.analyze_latent_space(
        expression_data, 
        experiment_ids, 
        metadata,
        args.n_clusters,
        args.output
    )
    
    # Save detailed results if requested
    if args.save_results:
        results_path = Path(args.output).with_suffix('.npz')
        np.savez(results_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
        print(f"Detailed results saved to: {results_path}")


def predict_batch_cli(args):
    """CLI function for batch prediction."""
    # Load inference engine
    inference = FlexibleVAEInference(args.model, args.device)
    
    # Load data
    expression_data, experiment_ids, metadata = load_data_from_file(args.data)
    
    # Run prediction
    results = inference.predict_control(expression_data, experiment_ids, args.batch_size)
    
    # Evaluate if requested
    if args.evaluate:
        metrics = inference.evaluate_reconstruction(expression_data, experiment_ids, args.batch_size)
        results['evaluation_metrics'] = metrics
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    np.savez(output_path / 'predictions.npz', **results)
    
    # Save metrics as JSON if evaluated
    if args.evaluate:
        import json
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"Batch prediction completed. Results saved to: {output_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Flexible VAE Inference')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model checkpoint')
    base_parser.add_argument('--data', type=str, required=True,
                           help='Path to input data file')
    base_parser.add_argument('--device', type=str, default='auto',
                           choices=['cuda', 'cpu', 'auto'], help='Device to use')
    base_parser.add_argument('--batch_size', type=int, default=256,
                           help='Batch size for processing')
    
    # Inject perturbations command
    inject_parser = subparsers.add_parser('inject', parents=[base_parser],
                                        help='Inject perturbations into control cells')
    inject_parser.add_argument('--perturbation_id', type=int, required=True,
                             help='Target gene ID for perturbation')
    inject_parser.add_argument('--output', type=str, required=True,
                             help='Output file path (.npz)')
    inject_parser.set_defaults(func=inject_perturbations_cli)
    
    # Analyze latent space command
    analyze_parser = subparsers.add_parser('analyze', parents=[base_parser],
                                         help='Analyze latent space representations')
    analyze_parser.add_argument('--output', type=str, required=True,
                              help='Output file path (.png)')
    analyze_parser.add_argument('--n_clusters', type=int, default=10,
                              help='Number of clusters for k-means')
    analyze_parser.add_argument('--save_results', action='store_true',
                              help='Save detailed results as .npz file')
    analyze_parser.set_defaults(func=analyze_latent_cli)
    
    # Batch prediction command
    predict_parser = subparsers.add_parser('predict', parents=[base_parser],
                                         help='Batch prediction and evaluation')
    predict_parser.add_argument('--output', type=str, required=True,
                              help='Output directory')
    predict_parser.add_argument('--evaluate', action='store_true',
                              help='Evaluate reconstruction quality')
    predict_parser.set_defaults(func=predict_batch_cli)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run the appropriate function
    args.func(args)


if __name__ == '__main__':
    main()
