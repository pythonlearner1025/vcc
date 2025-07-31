#!/usr/bin/env python3
"""
Unified training script for the Flexible VAE.

This script supports both training phases:
1. Phase 1: Self-supervised pretraining on large scRNA-seq datasets
2. Phase 2: Perturbation fine-tuning with paired control/perturbation data

Features:
- Flexible configuration via command line or config file
- Support for different gene embedding types (learned, ESM2, Gene2Vec)
- Robust data handling with phase-aware loading
- Comprehensive logging and checkpointing
- Easy model swapping and experimentation

Usage:
    # Phase 1: Pretraining
    python train_flexible_vae.py --phase 1 --data_path data/pretraining.npz --config configs/phase1.yaml
    
    # Phase 2: Fine-tuning
    python train_flexible_vae.py --phase 2 --data_path data/paired.npz --pretrained_model checkpoints/phase1_best.pt
    
    # Custom configuration
    python train_flexible_vae.py --latent_dim 256 --learning_rate 1e-4 --batch_size 512
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flexible_vae import (
    FlexibleVAE, VAEConfig, LearnedGeneEmbedding, 
    PretrainedGeneEmbedding, flexible_vae_loss
)
from dataset.vae_paired_dataloader import VAEPairedDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple dataset for Phase 1 pretraining without perturbation labels."""
    
    def __init__(self, expression_data: np.ndarray, experiment_ids: np.ndarray):
        self.expression_data = torch.FloatTensor(expression_data)
        self.experiment_ids = torch.LongTensor(experiment_ids)
    
    def __len__(self):
        return len(self.expression_data)
    
    def __getitem__(self, idx):
        return {
            'expression': self.expression_data[idx],
            'experiment_id': self.experiment_ids[idx]
        }


class FlexibleVAETrainer:
    """
    Trainer class for the Flexible VAE supporting both training phases.
    """
    
    def __init__(self, config: VAEConfig, device: str = 'cuda', log_dir: str = None):
        self.config = config
        self.device = device
        self.log_dir = log_dir or f"logs/flexible_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/checkpoints", exist_ok=True)
        
        # Initialize model (will be created later with proper dimensions)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Tensorboard logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"Trainer initialized. Logging to: {self.log_dir}")
    
    def create_model(self, gene_embedding: Union[str, torch.Tensor] = "learned"):
        """
        Create the FlexibleVAE model with appropriate gene embeddings.
        
        Args:
            gene_embedding: Either "learned" for learned embeddings or a tensor for pretrained
        """
        # Create gene embedding
        if isinstance(gene_embedding, str) and gene_embedding == "learned":
            gene_emb = LearnedGeneEmbedding(
                n_genes=self.config.n_genes,
                embed_dim=self.config.target_gene_embed_dim
            )
        elif isinstance(gene_embedding, torch.Tensor):
            gene_emb = PretrainedGeneEmbedding(gene_embedding, freeze=True)
        else:
            raise ValueError(f"Unsupported gene embedding type: {gene_embedding}")
        
        # Create model
        self.model = FlexibleVAE(self.config, gene_emb).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        return self.model
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained model weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if 'optimizer_state_dict' in checkpoint and strict:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint, strict=strict)
        
        logger.info(f"Loaded pretrained model from: {checkpoint_path}")
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"
        
        checkpoint_path = os.path.join(self.log_dir, "checkpoints", filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.log_dir, "checkpoints", "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to: {best_path}")
        
        logger.info(f"Saved checkpoint to: {checkpoint_path}")
    
    def train_phase1(self, train_loader: DataLoader, val_loader: DataLoader, 
                     num_epochs: int, save_every: int = 10):
        """
        Phase 1 training: Self-supervised pretraining without perturbation labels.
        """
        logger.info("Starting Phase 1 training (pretraining)...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self._train_epoch_phase1(train_loader)
            
            # Validation
            val_loss = self._validate_epoch_phase1(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoints
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
    
    def train_phase2(self, train_loader: DataLoader, val_loader: DataLoader, 
                     num_epochs: int, save_every: int = 10):
        """
        Phase 2 training: Fine-tuning with perturbation labels.
        """
        logger.info("Starting Phase 2 training (fine-tuning)...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self._train_epoch_phase2(train_loader)
            
            # Validation
            val_loss = self._validate_epoch_phase2(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoints
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
    
    def _train_epoch_phase1(self, train_loader: DataLoader) -> float:
        """Train one epoch for Phase 1."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        for batch in pbar:
            expression = batch['expression'].to(self.device)
            experiment_ids = batch['experiment_id'].to(self.device)
            
            # Forward pass (no perturbation IDs for Phase 1)
            outputs = self.model(expression, experiment_ids)
            
            # Compute loss
            loss_dict = flexible_vae_loss(outputs, expression, self.config)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train_Detailed/{key}', value, self.global_step)
        
        return total_loss / len(train_loader)
    
    def _validate_epoch_phase1(self, val_loader: DataLoader) -> float:
        """Validate one epoch for Phase 1."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                expression = batch['expression'].to(self.device)
                experiment_ids = batch['experiment_id'].to(self.device)
                
                outputs = self.model(expression, experiment_ids)
                loss_dict = flexible_vae_loss(outputs, expression, self.config)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(val_loader)
    
    def _train_epoch_phase2(self, train_loader: DataLoader) -> float:
        """Train one epoch for Phase 2."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        for batch in pbar:
            control_expression = batch['control_expression'].to(self.device)
            perturbed_expression = batch['perturbed_expression'].to(self.device)
            experiment_ids = batch['experiment_id'].to(self.device)
            target_gene_ids = batch['target_gene_id'].to(self.device)
            
            # Forward pass with perturbation
            outputs = self.model(
                control_expression, 
                experiment_ids, 
                target_gene_ids=target_gene_ids
            )
            
            # Compute loss against perturbed expression
            loss_dict = flexible_vae_loss(outputs, perturbed_expression, self.config)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train_Detailed/{key}', value, self.global_step)
        
        return total_loss / len(train_loader)
    
    def _validate_epoch_phase2(self, val_loader: DataLoader) -> float:
        """Validate one epoch for Phase 2."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                control_expression = batch['control_expression'].to(self.device)
                perturbed_expression = batch['perturbed_expression'].to(self.device)
                experiment_ids = batch['experiment_id'].to(self.device)
                target_gene_ids = batch['target_gene_id'].to(self.device)
                
                outputs = self.model(
                    control_expression, 
                    experiment_ids, 
                    target_gene_ids=target_gene_ids
                )
                loss_dict = flexible_vae_loss(outputs, perturbed_expression, self.config)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(val_loader)


def load_config(config_path: str) -> VAEConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return VAEConfig(**config_dict)


def create_data_loaders(data_path: str, phase: int, batch_size: int, 
                       val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for the specified training phase."""
    
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        expression_data = data['expression']
        experiment_ids = data['experiment_ids']
        
        if phase == 1:
            # Phase 1: Simple dataset without perturbation labels
            dataset = SimpleDataset(expression_data, experiment_ids)
            
            # Split into train/val
            n_val = int(len(dataset) * val_split)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
            
        elif phase == 2:
            # Phase 2: Paired dataset with perturbation labels
            perturbation_ids = data['perturbation_ids']
            target_gene_ids = data.get('target_gene_ids', perturbation_ids)  # Fallback
            
            dataset = VAEPairedDataset(
                expression_data, experiment_ids, perturbation_ids,
                target_gene_ids=target_gene_ids
            )
            
            # Split into train/val
            n_val = int(len(dataset) * val_split)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
        
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Flexible VAE')
    
    # Phase and data
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                       help='Training phase: 1=pretraining, 2=fine-tuning')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (.npz format)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent space dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Pretrained model (for Phase 2)
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model (for Phase 2)')
    
    # Gene embeddings
    parser.add_argument('--gene_embedding', type=str, default='learned',
                       choices=['learned', 'esm2', 'gene2vec'],
                       help='Type of gene embedding to use')
    parser.add_argument('--gene_embedding_path', type=str, default=None,
                       help='Path to pretrained gene embeddings')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for logs and checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = VAEConfig()
    
    # Override config with command line arguments
    config.latent_dim = args.latent_dim
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    
    # Load data to get dimensions
    data = np.load(args.data_path)
    config.input_dim = data['expression'].shape[1]
    config.n_experiments = len(np.unique(data['experiment_ids']))
    if 'perturbation_ids' in data:
        config.n_genes = len(np.unique(data.get('target_gene_ids', data['perturbation_ids'])))
    
    logger.info(f"Data dimensions: {data['expression'].shape}")
    logger.info(f"Config: input_dim={config.input_dim}, latent_dim={config.latent_dim}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_path, args.phase, args.batch_size
    )
    
    # Initialize trainer
    trainer = FlexibleVAETrainer(config, args.device, args.log_dir)
    
    # Create model with appropriate gene embeddings
    if args.gene_embedding == 'learned':
        gene_embedding = 'learned'
    else:
        # Load pretrained embeddings
        if not args.gene_embedding_path:
            raise ValueError(f"--gene_embedding_path required for {args.gene_embedding}")
        gene_embedding = torch.load(args.gene_embedding_path)
    
    trainer.create_model(gene_embedding)
    
    # Load pretrained model if specified
    if args.pretrained_model:
        trainer.load_pretrained(args.pretrained_model, strict=(args.phase == 2))
    
    # Save configuration
    config_path = os.path.join(trainer.log_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    # Train model
    try:
        if args.phase == 1:
            trainer.train_phase1(train_loader, val_loader, args.num_epochs, args.save_every)
        else:
            trainer.train_phase2(train_loader, val_loader, args.num_epochs, args.save_every)
            
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        trainer.save_checkpoint("error_checkpoint.pt")
        raise
    
    finally:
        trainer.writer.close()


if __name__ == '__main__':
    main()
