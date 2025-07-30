#!/usr/bin/env python3
"""
Training script for the Conditional VAE on single-cell RNA-seq data.

This script trains the VAE using paired control/perturbation data, ensuring that
the encoder only sees gene expression + experiment ID, while perturbation information
is only provided to the decoder.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig, vae_loss_function
from dataset.vae_paired_dataloader import create_vae_dataloaders, VAEPairedDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer class for the VAE model."""
    
    def __init__(self, model: ConditionalVAE, config: VAEConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5,
            verbose=True
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move batch to device
            expression = batch['expression'].to(self.device)
            experiment_ids = batch['experiment_id'].to(self.device)
            perturbation_ids = batch['perturbation_id'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(expression, experiment_ids, perturbation_ids)
            
            # Compute loss
            loss_dict = vae_loss_function(outputs, expression, self.config)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kld_loss += loss_dict['kld_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f"{loss_dict['total_loss'].item():.3f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.3f}",
                'KLD': f"{loss_dict['kld_loss'].item():.3f}"
            })
        
        n_batches = len(train_loader)
        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_kld_loss': total_kld_loss / n_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                expression = batch['expression'].to(self.device)
                experiment_ids = batch['experiment_id'].to(self.device)
                perturbation_ids = batch['perturbation_id'].to(self.device)
                
                # Forward pass
                outputs = self.model(expression, experiment_ids, perturbation_ids)
                
                # Compute loss
                loss_dict = vae_loss_function(outputs, expression, self.config)
                
                # Track losses
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_kld_loss += loss_dict['kld_loss'].item()
        
        n_batches = len(val_loader)
        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
            'val_kld_loss': total_kld_loss / n_batches
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: str,
              log_interval: int = 1,
              save_interval: int = 10):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            log_interval: Epochs between logging
            save_interval: Epochs between saving checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if available
        use_wandb = wandb.run is not None
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}")
                logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                logger.info(f"  Train Recon: {train_metrics['train_recon_loss']:.4f}")
                logger.info(f"  Val Recon: {val_metrics['val_recon_loss']:.4f}")
                logger.info(f"  Train KLD: {train_metrics['train_kld_loss']:.4f}")
                logger.info(f"  Val KLD: {val_metrics['val_kld_loss']:.4f}")
                
                if use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(save_dir / 'best_model.pt', epoch, val_metrics['val_loss'])
                logger.info(f"  New best model saved (val_loss: {val_metrics['val_loss']:.4f})")
            
            if epoch % save_interval == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics['val_loss'])
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_metadata_vocabularies(data_path: str) -> Dict[str, Dict]:
    """
    Create vocabularies for categorical metadata from the data file.
    
    Args:
        data_path: Path to the data file (.npz format)
        
    Returns:
        Dictionary containing vocabularies for experiments and perturbations
    """
    data = np.load(data_path, allow_pickle=True)
    obs = data['obs'].item()
    
    # Create experiment vocabulary (SRX accessions)
    unique_experiments = np.unique(obs['SRX_accession'])
    exp_to_id = {exp: i for i, exp in enumerate(unique_experiments)}
    
    # Create perturbation vocabulary
    unique_perturbations = np.unique(obs.get('perturbation', ['control']))
    pert_to_id = {pert: i for i, pert in enumerate(unique_perturbations)}
    
    return {
        'experiments': {
            'vocab': exp_to_id,
            'size': len(exp_to_id)
        },
        'perturbations': {
            'vocab': pert_to_id,
            'size': len(pert_to_id)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train VAE on single-cell RNA-seq data')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the data file (.npz format from download.py)')
    parser.add_argument('--min_cells_per_pair', type=int, default=5,
                       help='Minimum cells required per control/perturbation pair')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=2000,
                       help='Dimension of latent space')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden layer dimensions')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--kld_weight', type=float, default=1.0,
                       help='Weight for KL divergence loss')
    
    # Infrastructure arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='vae_scrnaseq',
                       help='Wandb project name')
    parser.add_argument('--log_interval', type=int, default=1,
                       help='Epochs between logging')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Epochs between saving checkpoints')
    
    args = parser.parse_args()
    
    # Initialize wandb
    run_name = f"vae_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    logger.info(f"Starting VAE training run: {run_name}")
    logger.info(f"Using device: {args.device}")
    
    # Create metadata vocabularies
    logger.info("Creating metadata vocabularies...")
    vocab_info = create_metadata_vocabularies(args.data_path)
    
    # Update config with vocabulary sizes
    config = VAEConfig(
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        n_perturbations=vocab_info['perturbations']['size'],
        n_experiments=vocab_info['experiments']['size'],
        kld_weight=args.kld_weight,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    logger.info(f"Model config:")
    logger.info(f"  Input dim: {config.input_dim}")
    logger.info(f"  Latent dim: {config.latent_dim}")
    logger.info(f"  Hidden dims: {config.hidden_dims}")
    logger.info(f"  N experiments: {config.n_experiments}")
    logger.info(f"  N perturbations: {config.n_perturbations}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_vae_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        min_cells_per_pair=args.min_cells_per_pair,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = ConditionalVAE(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Initialize trainer
    trainer = VAETrainer(model, config, args.device)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    logger.info("Training completed!")
    
    # Test final model
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics:")
    logger.info(f"  Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"  Test Recon: {test_metrics['val_recon_loss']:.4f}")
    logger.info(f"  Test KLD: {test_metrics['val_kld_loss']:.4f}")
    
    wandb.log({
        'test_loss': test_metrics['val_loss'],
        'test_recon_loss': test_metrics['val_recon_loss'],
        'test_kld_loss': test_metrics['val_kld_loss']
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()
