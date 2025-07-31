#!/usr/bin/env python3
"""
Simplified HepG2 Control Cell VAE Experiment (without TensorBoard)

This script performs a comprehensive VAE training experiment on HepG2 non-targeting control cells
to evaluate reconstruction efficiency across different latent space dimensions.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import json

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flexible_vae import (
    FlexibleVAE, VAEConfig, LearnedGeneEmbedding, flexible_vae_loss
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress scanpy warnings
sc.settings.verbosity = 1
warnings.filterwarnings('ignore')


class HepG2Dataset(Dataset):
    """Dataset class for HepG2 control cells."""
    
    def __init__(self, expression_data: np.ndarray):
        self.expression_data = torch.FloatTensor(expression_data)
        self.n_cells, self.n_genes = self.expression_data.shape
        self.experiment_ids = torch.zeros(self.n_cells, dtype=torch.long)
        logger.info(f"Dataset initialized: {self.n_cells} cells, {self.n_genes} genes")
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, idx):
        return {
            'expression': self.expression_data[idx],
            'experiment_id': self.experiment_ids[idx]
        }


class ReconstructionMetrics:
    """Comprehensive metrics for evaluating VAE reconstruction quality."""
    
    @staticmethod
    def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive reconstruction metrics with NaN handling."""
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()
        
        # Check for NaN values and handle them
        if np.isnan(original_np).any() or np.isnan(reconstructed_np).any():
            logger.warning("NaN values detected in predictions, returning default metrics")
            return {
                'overall_r2': 0.0,
                'overall_pearson': 0.0,
                'overall_mse': float('inf'),
                'overall_mae': float('inf'),
                'cell_r2_mean': 0.0,
                'cell_r2_std': 0.0,
                'cell_pearson_mean': 0.0,
                'cell_pearson_std': 0.0,
                'gene_r2_mean': 0.0,
                'gene_r2_std': 0.0,
                'gene_pearson_mean': 0.0,
                'gene_pearson_std': 0.0,
            }
        
        try:
            # Overall metrics
            overall_r2 = r2_score(original_np.flatten(), reconstructed_np.flatten())
            overall_pearson, _ = pearsonr(original_np.flatten(), reconstructed_np.flatten())
            overall_mse = np.mean((original_np - reconstructed_np) ** 2)
            overall_mae = np.mean(np.abs(original_np - reconstructed_np))
            
            # Handle potential NaN results
            if np.isnan(overall_r2):
                overall_r2 = 0.0
            if np.isnan(overall_pearson):
                overall_pearson = 0.0
            
        except Exception as e:
            logger.warning(f"Error computing overall metrics: {e}")
            overall_r2 = 0.0
            overall_pearson = 0.0
            overall_mse = float('inf')
            overall_mae = float('inf')
        
        # Cell-wise metrics
        cell_r2_scores = []
        cell_pearson_scores = []
        
        for i in range(original_np.shape[0]):
            orig_cell = original_np[i]
            recon_cell = reconstructed_np[i]
            
            try:
                if np.var(orig_cell) > 1e-10:  # Add small threshold
                    cell_r2 = r2_score(orig_cell, recon_cell)
                    if not np.isnan(cell_r2):
                        cell_r2_scores.append(cell_r2)
                
                if np.var(orig_cell) > 1e-10 and np.var(recon_cell) > 1e-10:
                    cell_pearson, _ = pearsonr(orig_cell, recon_cell)
                    if not np.isnan(cell_pearson):
                        cell_pearson_scores.append(cell_pearson)
            except:
                continue
        
        # Gene-wise metrics (sample subset to avoid memory issues)
        gene_r2_scores = []
        gene_pearson_scores = []
        n_genes_sample = min(1000, original_np.shape[1])  # Sample genes for efficiency
        gene_indices = np.random.choice(original_np.shape[1], n_genes_sample, replace=False)
        
        for i in gene_indices:
            orig_gene = original_np[:, i]
            recon_gene = reconstructed_np[:, i]
            
            try:
                if np.var(orig_gene) > 1e-10:
                    gene_r2 = r2_score(orig_gene, recon_gene)
                    if not np.isnan(gene_r2):
                        gene_r2_scores.append(gene_r2)
                
                if np.var(orig_gene) > 1e-10 and np.var(recon_gene) > 1e-10:
                    gene_pearson, _ = pearsonr(orig_gene, recon_gene)
                    if not np.isnan(gene_pearson):
                        gene_pearson_scores.append(gene_pearson)
            except:
                continue
        
        return {
            'overall_r2': float(overall_r2),
            'overall_pearson': float(overall_pearson),
            'overall_mse': float(overall_mse),
            'overall_mae': float(overall_mae),
            'cell_r2_mean': float(np.mean(cell_r2_scores)) if cell_r2_scores else 0.0,
            'cell_r2_std': float(np.std(cell_r2_scores)) if cell_r2_scores else 0.0,
            'cell_pearson_mean': float(np.mean(cell_pearson_scores)) if cell_pearson_scores else 0.0,
            'cell_pearson_std': float(np.std(cell_pearson_scores)) if cell_pearson_scores else 0.0,
            'gene_r2_mean': float(np.mean(gene_r2_scores)) if gene_r2_scores else 0.0,
            'gene_r2_std': float(np.std(gene_r2_scores)) if gene_r2_scores else 0.0,
            'gene_pearson_mean': float(np.mean(gene_pearson_scores)) if gene_pearson_scores else 0.0,
            'gene_pearson_std': float(np.std(gene_pearson_scores)) if gene_pearson_scores else 0.0,
        }


class SimpleHepG2VAETrainer:
    """Simplified trainer for HepG2 control cell experiments."""
    
    def __init__(self, latent_dim: int, device: str = 'cuda'):
        self.latent_dim = latent_dim
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        logger.info(f"Simple Trainer initialized for latent_dim={latent_dim}")
    
    def setup_model(self, n_genes: int):
        """Setup model with data-specific dimensions."""
        config = VAEConfig(
            input_dim=n_genes,
            latent_dim=self.latent_dim,
            n_experiments=1,
            n_genes=n_genes,
            target_gene_embed_dim=min(self.latent_dim, 32),
            learning_rate=5e-4,  # Reduced learning rate for stability
            batch_size=128,
            kld_weight=0.1,  # Reduced KLD weight to avoid collapse
            hidden_dims=self._get_hidden_dims(n_genes)
        )
        
        gene_emb = LearnedGeneEmbedding(n_genes=n_genes, embed_dim=min(self.latent_dim, 32))
        self.model = FlexibleVAE(config, gene_emb).to(self.device)
        
        # Initialize weights more carefully
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=5e-4,  # Reduced learning rate
            weight_decay=1e-4,  # Increased weight decay
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        self.config = config
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created: {n_params:,} parameters")
        
        return n_params
    
    def _get_hidden_dims(self, input_dim: int) -> List[int]:
        """Determine appropriate hidden dimensions."""
        if self.latent_dim <= 32:
            return [min(1024, input_dim // 4), min(512, input_dim // 8)]
        elif self.latent_dim <= 128:
            return [min(2048, input_dim // 2), min(1024, input_dim // 4), min(512, input_dim // 8)]
        else:
            return [min(4096, input_dim // 2), min(2048, input_dim // 4), min(1024, input_dim // 8)]
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 50):
        """Train the VAE model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Logging
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.2f}, "
                           f"Val Loss={val_loss:.2f}, "
                           f"Val R²={val_metrics['cell_r2_mean']:.4f}, "
                           f"Val Pearson={val_metrics['cell_pearson_mean']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        logger.info("Training completed!")
        return best_val_loss, val_metrics
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch in train_loader:
            expression = batch['expression'].to(self.device)
            experiment_ids = batch['experiment_id'].to(self.device)
            
            # Check for NaN in input
            if torch.isnan(expression).any():
                logger.warning("NaN detected in input data, skipping batch")
                continue
            
            outputs = self.model(expression, experiment_ids)
            
            # Check for NaN in outputs
            if torch.isnan(outputs['reconstructed']).any():
                logger.warning("NaN detected in model outputs, skipping batch")
                continue
            
            loss_dict = flexible_vae_loss(outputs, expression, self.config)
            loss = loss_dict['total_loss']
            
            # Check for NaN or inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.warning("NaN gradients detected, skipping update")
                continue
            
            # Clip gradients more aggressively
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
        
        if valid_batches == 0:
            logger.error("No valid batches in training epoch!")
            return float('inf')
        
        return total_loss / valid_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            for batch in val_loader:
                expression = batch['expression'].to(self.device)
                experiment_ids = batch['experiment_id'].to(self.device)
                
                outputs = self.model(expression, experiment_ids)
                loss_dict = flexible_vae_loss(outputs, expression, self.config)
                
                total_loss += loss_dict['total_loss'].item()
                all_original.append(expression)
                all_reconstructed.append(outputs['reconstructed'])
        
        # Compute metrics
        original_all = torch.cat(all_original, dim=0)
        reconstructed_all = torch.cat(all_reconstructed, dim=0)
        metrics = ReconstructionMetrics.compute_metrics(original_all, reconstructed_all)
        
        return total_loss / len(val_loader), metrics


def load_hepg2_data(data_path: str) -> Tuple[np.ndarray, int]:
    """Load and preprocess HepG2 control cell data."""
    logger.info(f"Loading HepG2 data from {data_path}")
    
    adata = sc.read_h5ad(data_path)
    logger.info(f"Loaded data: {adata.shape} (cells x genes)")
    
    # Filter for non-targeting controls
    non_targeting_mask = adata.obs['target_gene'] == 'non-targeting'
    control_data = adata[non_targeting_mask]
    logger.info(f"Non-targeting control cells: {control_data.shape[0]}")
    
    # Get expression data
    expression_data = control_data.X
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    
    # Basic filtering
    gene_counts = (expression_data > 0).sum(axis=0)
    keep_genes = gene_counts >= 10
    expression_data = expression_data[:, keep_genes]
    
    cell_counts = (expression_data > 0).sum(axis=1)
    keep_cells = cell_counts >= 200
    expression_data = expression_data[keep_cells]
    
    # Normalize the data for numerical stability
    # Apply log(x + 1) transformation
    expression_data = np.log1p(expression_data)
    
    # Z-score normalization (per gene)
    gene_means = np.mean(expression_data, axis=0)
    gene_stds = np.std(expression_data, axis=0)
    # Avoid division by zero
    gene_stds[gene_stds < 1e-6] = 1.0
    expression_data = (expression_data - gene_means) / gene_stds
    
    # Clip extreme values to avoid numerical issues
    expression_data = np.clip(expression_data, -5, 5)
    
    logger.info(f"After filtering and normalization: {expression_data.shape} (cells x genes)")
    logger.info(f"Expression range: {expression_data.min():.3f} to {expression_data.max():.3f}")
    logger.info(f"Expression mean: {expression_data.mean():.3f}, std: {expression_data.std():.3f}")
    
    return expression_data, expression_data.shape[1]


def run_experiment(data_path: str, latent_dims: List[int], device: str = 'cuda', num_epochs: int = 50):
    """Run experiments with different latent dimensions."""
    
    # Load data once
    expression_data, n_genes = load_hepg2_data(data_path)
    dataset = HepG2Dataset(expression_data)
    
    # Split into train/val
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Results storage
    results = {
        'latent_dims': [],
        'n_parameters': [],
        'final_losses': [],
        'final_metrics': {}
    }
    
    for latent_dim in latent_dims:
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENT: Latent Dimension = {latent_dim}")
        logger.info(f"{'='*50}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Create trainer
        trainer = SimpleHepG2VAETrainer(latent_dim=latent_dim, device=device)
        n_params = trainer.setup_model(n_genes=n_genes)
        
        # Train model
        final_loss, final_metrics = trainer.train(train_loader, val_loader, num_epochs)
        
        # Store results
        results['latent_dims'].append(latent_dim)
        results['n_parameters'].append(n_params)
        results['final_losses'].append(final_loss)
        results['final_metrics'][latent_dim] = final_metrics
        
        logger.info(f"Completed latent_dim={latent_dim}: Loss={final_loss:.2f}, "
                   f"R²={final_metrics['cell_r2_mean']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"experiments/hepg2_simple_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Generate plots
    generate_plots(results, timestamp)
    
    return results


def generate_plots(results: Dict, timestamp: str):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    latent_dims = results['latent_dims']
    
    # Plot 1: Final Loss vs Latent Dimension
    axes[0, 0].plot(latent_dims, results['final_losses'], 'bo-')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Final Validation Loss')
    axes[0, 0].set_title('Reconstruction Loss vs Latent Dimension')
    axes[0, 0].grid(True)
    
    # Plot 2: Number of Parameters vs Latent Dimension
    axes[0, 1].plot(latent_dims, results['n_parameters'], 'ro-')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Number of Parameters')
    axes[0, 1].set_title('Model Size vs Latent Dimension')
    axes[0, 1].grid(True)
    
    # Plot 3: R² vs Latent Dimension
    r2_means = [results['final_metrics'][dim]['cell_r2_mean'] for dim in latent_dims]
    r2_stds = [results['final_metrics'][dim]['cell_r2_std'] for dim in latent_dims]
    axes[0, 2].errorbar(latent_dims, r2_means, yerr=r2_stds, fmt='go-', capsize=5)
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Cell-wise R²')
    axes[0, 2].set_title('Reconstruction R² vs Latent Dimension')
    axes[0, 2].grid(True)
    
    # Plot 4: Pearson Correlation vs Latent Dimension
    pearson_means = [results['final_metrics'][dim]['cell_pearson_mean'] for dim in latent_dims]
    pearson_stds = [results['final_metrics'][dim]['cell_pearson_std'] for dim in latent_dims]
    axes[1, 0].errorbar(latent_dims, pearson_means, yerr=pearson_stds, fmt='mo-', capsize=5)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Cell-wise Pearson r')
    axes[1, 0].set_title('Reconstruction Correlation vs Latent Dimension')
    axes[1, 0].grid(True)
    
    # Plot 5: Efficiency (R² / Parameters)
    efficiency = [r2_means[i] / results['n_parameters'][i] * 1e6 for i in range(len(latent_dims))]
    axes[1, 1].plot(latent_dims, efficiency, 'co-')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Efficiency (R² / Params × 10⁶)')
    axes[1, 1].set_title('Reconstruction Efficiency vs Latent Dimension')
    axes[1, 1].grid(True)
    
    # Plot 6: Trade-off Plot (Parameters vs R²)
    scatter = axes[1, 2].scatter(results['n_parameters'], r2_means, c=latent_dims, 
                                cmap='viridis', s=100)
    for i, dim in enumerate(latent_dims):
        axes[1, 2].annotate(str(dim), (results['n_parameters'][i], r2_means[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 2].set_xlabel('Number of Parameters')
    axes[1, 2].set_ylabel('Cell-wise R²')
    axes[1, 2].set_title('Reconstruction Quality vs Model Size')
    axes[1, 2].grid(True)
    plt.colorbar(scatter, ax=axes[1, 2], label='Latent Dimension')
    
    plt.tight_layout()
    plot_path = f"experiments/hepg2_simple_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to: {plot_path}")


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Simple HepG2 Control Cell VAE Experiment')
    
    parser.add_argument('--data_path', type=str, 
                       default='/Users/agreic/Documents/GitHub/virtual-cell/data/raw/competition_support_set/hepg2.h5',
                       help='Path to HepG2 H5 data file')
    parser.add_argument('--latent_dims', nargs='+', type=int,
                       default=[16, 32, 64, 128],
                       help='Latent dimensions to test')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    logger.info("Starting Simple HepG2 Control Cell VAE Experiment")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Latent dimensions: {args.latent_dims}")
    logger.info(f"Epochs: {args.num_epochs}")
    
    # Run experiments
    results = run_experiment(
        data_path=args.data_path,
        latent_dims=args.latent_dims,
        device=args.device,
        num_epochs=args.num_epochs
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    print("\n| Latent | Params | Loss | R² Mean | R² Std | Pearson | Efficiency |")
    print("|--------|--------|------|---------|--------|---------|------------|")
    
    for i, latent_dim in enumerate(results['latent_dims']):
        metrics = results['final_metrics'][latent_dim]
        efficiency = metrics['cell_r2_mean'] / results['n_parameters'][i] * 1e6
        print(f"| {latent_dim:6d} | {results['n_parameters'][i]:6,} | "
              f"{results['final_losses'][i]:4.0f} | "
              f"{metrics['cell_r2_mean']:7.4f} | "
              f"{metrics['cell_r2_std']:6.4f} | "
              f"{metrics['cell_pearson_mean']:7.4f} | "
              f"{efficiency:10.2f} |")
    
    # Find best models
    r2_scores = [results['final_metrics'][dim]['cell_r2_mean'] for dim in results['latent_dims']]
    best_r2_idx = np.argmax(r2_scores)
    best_r2_dim = results['latent_dims'][best_r2_idx]
    
    efficiencies = [results['final_metrics'][dim]['cell_r2_mean'] / results['n_parameters'][i] * 1e6 
                   for i, dim in enumerate(results['latent_dims'])]
    best_eff_idx = np.argmax(efficiencies)
    best_eff_dim = results['latent_dims'][best_eff_idx]
    
    logger.info(f"\nBest R² Score: Latent {best_r2_dim} (R² = {r2_scores[best_r2_idx]:.4f})")
    logger.info(f"Best Efficiency: Latent {best_eff_dim} (Eff = {efficiencies[best_eff_idx]:.2f})")
    
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()
