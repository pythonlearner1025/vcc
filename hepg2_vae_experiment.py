#!/usr/bin/env python3
"""
HepG2 Control Cell VAE Experiment

This script performs a comprehensive VAE training experiment on HepG2 non-targeting control cells
to evaluate reconstruction efficiency across different latent space dimensions.

Experiment Design:
1. Load HepG2 data and filter for non-targeting controls only
2. Train VAE models with different latent dimensions (16, 32, 64, 128, 256, 512)
3. Evaluate reconstruction efficiency with specialized metrics for scRNA-seq data
4. Log all metrics to TensorBoard for comparison
5. Generate comprehensive analysis plots

Features:
- Gene expression-specific reconstruction metrics
- Sparse data handling for scRNA-seq
- Comprehensive logging and visualization
- Automated latent dimension experiments
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
from torch.utils.tensorboard import SummaryWriter
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
        """
        Initialize dataset with expression data.
        
        Args:
            expression_data: Gene expression matrix [n_cells, n_genes]
        """
        self.expression_data = torch.FloatTensor(expression_data)
        self.n_cells, self.n_genes = self.expression_data.shape
        
        # Create dummy experiment IDs (all cells from same experiment)
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
    """
    Comprehensive metrics for evaluating VAE reconstruction quality on scRNA-seq data.
    """
    
    @staticmethod
    def compute_cell_wise_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction metrics for each cell.
        
        Args:
            original: Original expression [n_cells, n_genes]  
            reconstructed: Reconstructed expression [n_cells, n_genes]
            
        Returns:
            Dictionary of metric arrays
        """
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()
        
        n_cells = original_np.shape[0]
        
        # Per-cell metrics
        cell_r2 = []
        cell_pearson = []
        cell_spearman = []
        cell_mse = []
        cell_mae = []
        cell_cosine_sim = []
        
        for i in range(n_cells):
            orig_cell = original_np[i]
            recon_cell = reconstructed_np[i]
            
            # R-squared
            if np.var(orig_cell) > 0:
                r2 = r2_score(orig_cell, recon_cell)
            else:
                r2 = 0.0
            cell_r2.append(r2)
            
            # Pearson correlation
            if np.var(orig_cell) > 0 and np.var(recon_cell) > 0:
                pearson_r, _ = pearsonr(orig_cell, recon_cell)
                if np.isnan(pearson_r):
                    pearson_r = 0.0
            else:
                pearson_r = 0.0
            cell_pearson.append(pearson_r)
            
            # Spearman correlation
            try:
                spearman_r, _ = spearmanr(orig_cell, recon_cell)
                if np.isnan(spearman_r):
                    spearman_r = 0.0
            except:
                spearman_r = 0.0
            cell_spearman.append(spearman_r)
            
            # MSE and MAE
            cell_mse.append(np.mean((orig_cell - recon_cell) ** 2))
            cell_mae.append(np.mean(np.abs(orig_cell - recon_cell)))
            
            # Cosine similarity
            norm_orig = np.linalg.norm(orig_cell)
            norm_recon = np.linalg.norm(recon_cell)
            if norm_orig > 0 and norm_recon > 0:
                cosine_sim = np.dot(orig_cell, recon_cell) / (norm_orig * norm_recon)
            else:
                cosine_sim = 0.0
            cell_cosine_sim.append(cosine_sim)
        
        return {
            'cell_r2': np.array(cell_r2),
            'cell_pearson': np.array(cell_pearson),
            'cell_spearman': np.array(cell_spearman),
            'cell_mse': np.array(cell_mse),
            'cell_mae': np.array(cell_mae),
            'cell_cosine_sim': np.array(cell_cosine_sim)
        }
    
    @staticmethod
    def compute_gene_wise_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction metrics for each gene.
        
        Args:
            original: Original expression [n_cells, n_genes]
            reconstructed: Reconstructed expression [n_cells, n_genes]
            
        Returns:
            Dictionary of metric arrays
        """
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()
        
        n_genes = original_np.shape[1]
        
        # Per-gene metrics
        gene_r2 = []
        gene_pearson = []
        gene_spearman = []
        gene_mse = []
        gene_mae = []
        
        for i in range(n_genes):
            orig_gene = original_np[:, i]
            recon_gene = reconstructed_np[:, i]
            
            # R-squared
            if np.var(orig_gene) > 0:
                r2 = r2_score(orig_gene, recon_gene)
            else:
                r2 = 0.0
            gene_r2.append(r2)
            
            # Pearson correlation
            if np.var(orig_gene) > 0 and np.var(recon_gene) > 0:
                pearson_r, _ = pearsonr(orig_gene, recon_gene)
                if np.isnan(pearson_r):
                    pearson_r = 0.0
            else:
                pearson_r = 0.0
            gene_pearson.append(pearson_r)
            
            # Spearman correlation
            try:
                spearman_r, _ = spearmanr(orig_gene, recon_gene)
                if np.isnan(spearman_r):
                    spearman_r = 0.0
            except:
                spearman_r = 0.0
            gene_spearman.append(spearman_r)
            
            # MSE and MAE
            gene_mse.append(np.mean((orig_gene - recon_gene) ** 2))
            gene_mae.append(np.mean(np.abs(orig_gene - recon_gene)))
        
        return {
            'gene_r2': np.array(gene_r2),
            'gene_pearson': np.array(gene_pearson),
            'gene_spearman': np.array(gene_spearman),
            'gene_mse': np.array(gene_mse),
            'gene_mae': np.array(gene_mae)
        }
    
    @staticmethod
    def compute_summary_metrics(cell_metrics: Dict, gene_metrics: Dict) -> Dict[str, float]:
        """Compute summary statistics across cells and genes."""
        summary = {}
        
        # Cell-wise summaries
        for metric_name, values in cell_metrics.items():
            summary[f"{metric_name}_mean"] = np.mean(values)
            summary[f"{metric_name}_std"] = np.std(values)
            summary[f"{metric_name}_median"] = np.median(values)
            summary[f"{metric_name}_q25"] = np.percentile(values, 25)
            summary[f"{metric_name}_q75"] = np.percentile(values, 75)
        
        # Gene-wise summaries
        for metric_name, values in gene_metrics.items():
            summary[f"{metric_name}_mean"] = np.mean(values)
            summary[f"{metric_name}_std"] = np.std(values)
            summary[f"{metric_name}_median"] = np.median(values)
            summary[f"{metric_name}_q25"] = np.percentile(values, 25)
            summary[f"{metric_name}_q75"] = np.percentile(values, 75)
        
        return summary


class HepG2VAETrainer:
    """
    Specialized trainer for HepG2 control cell experiments.
    """
    
    def __init__(self, latent_dim: int, device: str = 'cuda', log_dir: str = None):
        self.latent_dim = latent_dim
        self.device = device
        self.log_dir = log_dir or f"experiments/hepg2_latent_{latent_dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.log_dir}/plots", exist_ok=True)
        
        # Initialize model components (will be set later)
        self.config = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Tensorboard logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"HepG2 Trainer initialized for latent_dim={latent_dim}")
        logger.info(f"Logging to: {self.log_dir}")
    
    def setup_model(self, n_genes: int, n_experiments: int = 1):
        """Setup model with data-specific dimensions."""
        
        # Create config
        self.config = VAEConfig(
            input_dim=n_genes,
            latent_dim=self.latent_dim,
            n_experiments=n_experiments,
            n_genes=n_genes,
            target_gene_embed_dim=min(self.latent_dim, 32),  # Cap embedding dimension
            learning_rate=1e-3,
            batch_size=128,
            kld_weight=1.0,  # Standard beta-VAE
            hidden_dims=self._get_hidden_dims(n_genes, self.latent_dim)
        )
        
        # Create gene embedding (smaller embedding for large gene sets)
        embed_dim = min(self.latent_dim, 32)  # Cap embedding dimension
        gene_emb = LearnedGeneEmbedding(
            n_genes=n_genes,
            embed_dim=embed_dim
        )
        
        # Create model
        self.model = FlexibleVAE(self.config, gene_emb).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5
        )
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created: {n_params:,} parameters")
        
        # Save config
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def _get_hidden_dims(self, input_dim: int, latent_dim: int) -> List[int]:
        """Determine appropriate hidden dimensions."""
        if latent_dim <= 32:
            return [min(1024, input_dim // 4), min(512, input_dim // 8)]
        elif latent_dim <= 128:
            return [min(2048, input_dim // 2), min(1024, input_dim // 4), min(512, input_dim // 8)]
        else:
            return [min(4096, input_dim // 2), min(2048, input_dim // 4), min(1024, input_dim // 8), min(512, input_dim // 16)]
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100):
        """Train the VAE model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss['total_loss'])
            
            # Logging
            self._log_epoch_metrics(train_loss, train_metrics, val_loss, val_metrics)
            
            # Save checkpoints
            if val_loss['total_loss'] < self.best_loss:
                self.best_loss = val_loss['total_loss']
                self._save_checkpoint(is_best=True)
            
            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                
            if (epoch + 1) % 10 == 0:
                self._generate_reconstruction_plots(val_loader)
        
        logger.info("Training completed!")
        return self.best_loss
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[Dict, Dict]:
        """Train one epoch."""
        self.model.train()
        total_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'kld_loss': 0}
        
        all_original = []
        all_reconstructed = []
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            expression = batch['expression'].to(self.device)
            experiment_ids = batch['experiment_id'].to(self.device)
            
            # Forward pass
            outputs = self.model(expression, experiment_ids)
            
            # Compute loss
            loss_dict = flexible_vae_loss(outputs, expression, self.config)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            
            # Collect data for metrics (subset to save memory)
            if batch_idx < 10:  # Only use first 10 batches for detailed metrics
                all_original.append(expression.detach())
                all_reconstructed.append(outputs['reconstructed'].detach())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(train_loader)
        
        # Compute reconstruction metrics on subset
        if all_original:
            original_subset = torch.cat(all_original, dim=0)
            reconstructed_subset = torch.cat(all_reconstructed, dim=0)
            cell_metrics = ReconstructionMetrics.compute_cell_wise_metrics(original_subset, reconstructed_subset)
            gene_metrics = ReconstructionMetrics.compute_gene_wise_metrics(original_subset, reconstructed_subset)
            metrics = ReconstructionMetrics.compute_summary_metrics(cell_metrics, gene_metrics)
        else:
            metrics = {}
        
        return total_losses, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[Dict, Dict]:
        """Validate one epoch."""
        self.model.eval()
        total_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'kld_loss': 0}
        
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            for batch in val_loader:
                expression = batch['expression'].to(self.device)
                experiment_ids = batch['experiment_id'].to(self.device)
                
                outputs = self.model(expression, experiment_ids)
                loss_dict = flexible_vae_loss(outputs, expression, self.config)
                
                for key in total_losses:
                    total_losses[key] += loss_dict[key].item()
                
                all_original.append(expression)
                all_reconstructed.append(outputs['reconstructed'])
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(val_loader)
        
        # Compute comprehensive reconstruction metrics
        original_all = torch.cat(all_original, dim=0)
        reconstructed_all = torch.cat(all_reconstructed, dim=0)
        
        cell_metrics = ReconstructionMetrics.compute_cell_wise_metrics(original_all, reconstructed_all)
        gene_metrics = ReconstructionMetrics.compute_gene_wise_metrics(original_all, reconstructed_all)
        metrics = ReconstructionMetrics.compute_summary_metrics(cell_metrics, gene_metrics)
        
        return total_losses, metrics
    
    def _log_epoch_metrics(self, train_loss: Dict, train_metrics: Dict, 
                          val_loss: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard and console."""
        
        # Log losses
        for key, value in train_loss.items():
            self.writer.add_scalar(f'Loss/Train_{key}', value, self.epoch)
        
        for key, value in val_loss.items():
            self.writer.add_scalar(f'Loss/Val_{key}', value, self.epoch)
        
        # Log detailed metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/Val_{key}', value, self.epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        
        # Console logging
        logger.info(f"Epoch {self.epoch}:")
        logger.info(f"  Train Loss: {train_loss['total_loss']:.4f} "
                   f"(Recon: {train_loss['reconstruction_loss']:.4f}, "
                   f"KLD: {train_loss['kld_loss']:.4f})")
        logger.info(f"  Val Loss: {val_loss['total_loss']:.4f} "
                   f"(Recon: {val_loss['reconstruction_loss']:.4f}, "
                   f"KLD: {val_loss['kld_loss']:.4f})")
        
        if val_metrics:
            logger.info(f"  Val R²: {val_metrics.get('cell_r2_mean', 0):.4f} ± {val_metrics.get('cell_r2_std', 0):.4f}")
            logger.info(f"  Val Pearson: {val_metrics.get('cell_pearson_mean', 0):.4f} ± {val_metrics.get('cell_pearson_std', 0):.4f}")
    
    def _save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"
        
        checkpoint_path = os.path.join(self.log_dir, "checkpoints", filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'latent_dim': self.latent_dim,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.log_dir, "checkpoints", "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def _generate_reconstruction_plots(self, val_loader: DataLoader):
        """Generate reconstruction visualization plots."""
        self.model.eval()
        
        # Get a batch for visualization
        batch = next(iter(val_loader))
        expression = batch['expression'].to(self.device)
        experiment_ids = batch['experiment_id'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(expression, experiment_ids)
            reconstructed = outputs['reconstructed']
        
        # Convert to numpy
        original_np = expression.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Scatter plot of original vs reconstructed (sample of genes)
        sample_genes = np.random.choice(original_np.shape[1], 1000, replace=False)
        orig_sample = original_np[:, sample_genes].flatten()
        recon_sample = reconstructed_np[:, sample_genes].flatten()
        
        axes[0, 0].scatter(orig_sample, recon_sample, alpha=0.1, s=1)
        axes[0, 0].plot([orig_sample.min(), orig_sample.max()], 
                       [orig_sample.min(), orig_sample.max()], 'r--')
        axes[0, 0].set_xlabel('Original Expression')
        axes[0, 0].set_ylabel('Reconstructed Expression')
        axes[0, 0].set_title('Original vs Reconstructed (Sample)')
        
        # Plot 2: Distribution comparison
        axes[0, 1].hist(orig_sample, bins=50, alpha=0.7, label='Original', density=True)
        axes[0, 1].hist(recon_sample, bins=50, alpha=0.7, label='Reconstructed', density=True)
        axes[0, 1].set_xlabel('Expression Level')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Expression Distribution')
        axes[0, 1].legend()
        
        # Plot 3: Cell-wise correlation distribution
        cell_correlations = []
        for i in range(min(100, original_np.shape[0])):
            if np.var(original_np[i]) > 0 and np.var(reconstructed_np[i]) > 0:
                corr, _ = pearsonr(original_np[i], reconstructed_np[i])
                if not np.isnan(corr):
                    cell_correlations.append(corr)
        
        axes[1, 0].hist(cell_correlations, bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Cell-wise Pearson Correlation')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Cell-wise Reconstruction Correlation')
        axes[1, 0].axvline(np.mean(cell_correlations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(cell_correlations):.3f}')
        axes[1, 0].legend()
        
        # Plot 4: Latent space visualization (first 2 dimensions)
        latent = outputs['z'].cpu().numpy()
        axes[1, 1].scatter(latent[:, 0], latent[:, 1], alpha=0.6, s=10)
        axes[1, 1].set_xlabel('Latent Dimension 1')
        axes[1, 1].set_ylabel('Latent Dimension 2')
        axes[1, 1].set_title(f'Latent Space (Dims 1-2, Total: {self.latent_dim})')
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "plots", f"reconstruction_epoch_{self.epoch}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log plot to tensorboard
        self.writer.add_figure('Reconstruction_Analysis', fig, self.epoch)


def load_hepg2_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess HepG2 control cell data."""
    logger.info(f"Loading HepG2 data from {data_path}")
    
    # Load data
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
    # Remove genes with very low expression
    gene_counts = (expression_data > 0).sum(axis=0)
    keep_genes = gene_counts >= 10  # Gene must be expressed in at least 10 cells
    expression_data = expression_data[:, keep_genes]
    
    # Remove cells with very few expressed genes
    cell_counts = (expression_data > 0).sum(axis=1)
    keep_cells = cell_counts >= 200  # Cell must express at least 200 genes
    expression_data = expression_data[keep_cells]
    
    logger.info(f"After filtering: {expression_data.shape} (cells x genes)")
    logger.info(f"Expression range: {expression_data.min():.3f} to {expression_data.max():.3f}")
    logger.info(f"Expression mean: {expression_data.mean():.3f}")
    
    # Create experiment labels (all same experiment)
    experiment_labels = np.zeros(expression_data.shape[0], dtype=int)
    
    return expression_data, experiment_labels


def run_latent_dim_experiment(data_path: str, latent_dims: List[int], 
                            device: str = 'cuda', num_epochs: int = 100):
    """Run experiments with different latent dimensions."""
    
    # Load data once
    expression_data, experiment_labels = load_hepg2_data(data_path)
    
    # Create dataset
    dataset = HepG2Dataset(expression_data)
    
    # Split into train/val
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Results storage
    results = {
        'latent_dims': [],
        'final_losses': [],
        'n_parameters': [],
        'best_metrics': {}
    }
    
    for latent_dim in latent_dims:
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENT: Latent Dimension = {latent_dim}")
        logger.info(f"{'='*50}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Create trainer
        trainer = HepG2VAETrainer(latent_dim=latent_dim, device=device)
        trainer.setup_model(n_genes=expression_data.shape[1])
        
        # Train model
        best_loss = trainer.train(train_loader, val_loader, num_epochs)
        
        # Record results
        results['latent_dims'].append(latent_dim)
        results['final_losses'].append(best_loss)
        results['n_parameters'].append(sum(p.numel() for p in trainer.model.parameters()))
        
        # Final evaluation
        trainer.model.eval()
        with torch.no_grad():
            all_original = []
            all_reconstructed = []
            
            for batch in val_loader:
                expression = batch['expression'].to(device)
                experiment_ids = batch['experiment_id'].to(device)
                outputs = trainer.model(expression, experiment_ids)
                
                all_original.append(expression)
                all_reconstructed.append(outputs['reconstructed'])
            
            # Compute final metrics
            original_all = torch.cat(all_original, dim=0)
            reconstructed_all = torch.cat(all_reconstructed, dim=0)
            
            cell_metrics = ReconstructionMetrics.compute_cell_wise_metrics(original_all, reconstructed_all)
            gene_metrics = ReconstructionMetrics.compute_gene_wise_metrics(original_all, reconstructed_all)
            final_metrics = ReconstructionMetrics.compute_summary_metrics(cell_metrics, gene_metrics)
            
            # Convert numpy types to Python native types for JSON serialization
            final_metrics_json = {}
            for key, value in final_metrics.items():
                if isinstance(value, np.ndarray):
                    final_metrics_json[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    final_metrics_json[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    final_metrics_json[key] = int(value)
                elif isinstance(value, np.bool_):
                    final_metrics_json[key] = bool(value)
                else:
                    final_metrics_json[key] = value
            
            results['best_metrics'][latent_dim] = final_metrics_json
        
        # Clean up
        trainer.writer.close()
        
        logger.info(f"Completed latent_dim={latent_dim}: Loss={best_loss:.4f}")
    
    # Convert all numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Save overall results
    results_path = f"experiments/hepg2_latent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Convert results before saving
    json_safe_results = convert_numpy_types(results)
    
    with open(results_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    logger.info(f"Experiment results saved to: {results_path}")
    
    # Generate comparison plots
    generate_comparison_plots(results)
    
    return results


def generate_comparison_plots(results: Dict):
    """Generate comparison plots across latent dimensions."""
    
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
    r2_means = [results['best_metrics'][dim]['cell_r2_mean'] for dim in latent_dims]
    r2_stds = [results['best_metrics'][dim]['cell_r2_std'] for dim in latent_dims]
    axes[0, 2].errorbar(latent_dims, r2_means, yerr=r2_stds, fmt='go-', capsize=5)
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Cell-wise R²')
    axes[0, 2].set_title('Reconstruction R² vs Latent Dimension')
    axes[0, 2].grid(True)
    
    # Plot 4: Pearson Correlation vs Latent Dimension
    pearson_means = [results['best_metrics'][dim]['cell_pearson_mean'] for dim in latent_dims]
    pearson_stds = [results['best_metrics'][dim]['cell_pearson_std'] for dim in latent_dims]
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
    axes[1, 2].scatter(results['n_parameters'], r2_means, c=latent_dims, cmap='viridis', s=100)
    for i, dim in enumerate(latent_dims):
        axes[1, 2].annotate(str(dim), (results['n_parameters'][i], r2_means[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 2].set_xlabel('Number of Parameters')
    axes[1, 2].set_ylabel('Cell-wise R²')
    axes[1, 2].set_title('Reconstruction Quality vs Model Size')
    axes[1, 2].grid(True)
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('Latent Dimension')
    
    plt.tight_layout()
    plot_path = f"experiments/hepg2_latent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to: {plot_path}")


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='HepG2 Control Cell VAE Experiment')
    
    parser.add_argument('--data_path', type=str, 
                       default='/Users/agreic/Documents/GitHub/virtual-cell/data/raw/competition_support_set/hepg2.h5',
                       help='Path to HepG2 H5 data file')
    parser.add_argument('--latent_dims', nargs='+', type=int,
                       default=[16, 32, 64, 128, 256, 512],
                       help='Latent dimensions to test')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--single_run', type=int, default=None,
                       help='Run only single latent dimension')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    logger.info("Starting HepG2 Control Cell VAE Experiment")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.num_epochs}")
    
    if args.single_run:
        logger.info(f"Single run mode: latent_dim={args.single_run}")
        latent_dims = [args.single_run]
    else:
        logger.info(f"Latent dimensions to test: {args.latent_dims}")
        latent_dims = args.latent_dims
    
    # Run experiments
    results = run_latent_dim_experiment(
        data_path=args.data_path,
        latent_dims=latent_dims,
        device=args.device,
        num_epochs=args.num_epochs
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    
    for i, latent_dim in enumerate(results['latent_dims']):
        metrics = results['best_metrics'][latent_dim]
        logger.info(f"Latent Dim {latent_dim}:")
        logger.info(f"  Parameters: {results['n_parameters'][i]:,}")
        logger.info(f"  Final Loss: {results['final_losses'][i]:.4f}")
        logger.info(f"  Cell R²: {metrics['cell_r2_mean']:.4f} ± {metrics['cell_r2_std']:.4f}")
        logger.info(f"  Cell Pearson: {metrics['cell_pearson_mean']:.4f} ± {metrics['cell_pearson_std']:.4f}")
        logger.info(f"  Gene R²: {metrics['gene_r2_mean']:.4f} ± {metrics['gene_r2_std']:.4f}")
    
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()
