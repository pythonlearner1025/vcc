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
        """Build index mapping global cell indices to (batch_file, local_index)."""
        self.cell_index = []
        self.batch_starts = [0]
        total_cells = 0
        
        for batch_file in self.batch_files:
            with h5py.File(batch_file, 'r') as f:
                n_cells = f['X'].shape[0]
                
                # Limit cells per batch if specified
                if self.max_cells_per_batch:
                    n_cells = min(n_cells, self.max_cells_per_batch)
                
                for local_idx in range(n_cells):
                    self.cell_index.append((batch_file, local_idx))
                
                total_cells += n_cells
                self.batch_starts.append(total_cells)
    
    def _build_metadata_vocabularies(self):
        """Build vocabularies for categorical metadata across all batches."""
        all_metadata = []
        
        # Collect all metadata
        for batch_file in self.batch_files:
            with h5py.File(batch_file, 'r') as f:
                n_cells = f['X'].shape[0]
                if self.max_cells_per_batch:
                    n_cells = min(n_cells, self.max_cells_per_batch)
                
                batch_metadata = {
                    'srx_accession': [f['obs/SRX_accession'][i].decode() for i in range(n_cells)],
                    'gene_count': f['obs/gene_count'][:n_cells].tolist(),
                    'umi_count': f['obs/umi_count'][:n_cells].tolist()
                }
                all_metadata.append(pd.DataFrame(batch_metadata))
        
        # Combine all metadata
        metadata_df = pd.concat(all_metadata, ignore_index=True)
        
        # In practice, you would have this information in your metadata
        np.random.seed(42)
        n_cells = len(metadata_df)
        
        # Simulate cell types (could be inferred from clustering) # Cell type = transcriptomic profile
        # metadata_df['cell_type'] = np.random.choice(
        #     ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'Dendritic', 'Other'], 
        #     size=n_cells, 
        #     p=[0.3, 0.2, 0.1, 0.2, 0.1, 0.1]
        # )
        
        # Simulate perturbations (control vs various treatments)
        metadata_df['perturbation'] = np.random.choice(
            ['control', 'treatment_A', 'treatment_B', 'knockout', 'overexpression'],
            size=n_cells,
            p=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Create vocabularies
        self.vocabularies = create_metadata_vocabularies(metadata_df)
        self.encoded_metadata = encode_metadata(metadata_df, self.vocabularies)
        
        logger.info(f"Metadata vocabularies created:")
        for key, vocab in self.vocabularies.items():
            logger.info(f"  {key}: {len(vocab)} unique values")
    
    def __len__(self):
        return len(self.cell_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell's data."""
        batch_file, local_idx = self.cell_index[idx]
        
        # Load data from file
        with h5py.File(batch_file, 'r') as f:
            # Gene expression (filter to HVG genes)
            x = f['X'][local_idx][self.gene_indices]
            
            # Metadata
            # gene_count = f['obs/gene_count'][local_idx]
            umi_count = f['obs/umi_count'][local_idx]
            srx_accession = f['obs/SRX_accession'][local_idx].decode()
        
        # Convert to tensors
        x = torch.from_numpy(x).float()
        
        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
        
        # Get encoded metadata
        # cell_type_id = self.encoded_metadata['cell_type_ids'][idx]
        perturbation_id = self.encoded_metadata['perturbation_ids'][idx]
        experiment_id = self.encoded_metadata['srx_accession_ids'][idx]
        
        return {
            'x': x,
            # 'cell_type_ids': cell_type_id,
            'perturbation_ids': perturbation_id,
            'experiment_ids': experiment_id,
            # 'gene_count': torch.tensor(gene_count, dtype=torch.float),
            'umi_count': torch.tensor(umi_count, dtype=torch.float),
            'srx_accession': srx_accession
        }


def load_hvg_genes(hvg_file: str) -> List[str]:
    """Load HVG gene names from file."""
    hvg_genes = []
    with open(hvg_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene:
                hvg_genes.append(gene)
    return hvg_genes


def create_data_loaders(data_dir: str, 
                       hvg_file: Optional[str] = None,
                       batch_size: int = 256,
                       val_split: float = 0.1,
                       max_cells_per_batch: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Load HVG genes if provided
    hvg_genes = None
    if hvg_file and os.path.exists(hvg_file):
        hvg_genes = load_hvg_genes(hvg_file)
        logger.info(f"Loaded {len(hvg_genes)} HVG genes from {hvg_file}")
    
    # Create full dataset
    full_dataset = ScRNAVAEDataset(
        data_dir=data_dir,
        hvg_genes=hvg_genes,
        max_cells_per_batch=max_cells_per_batch
    )
    
    # Split into train/val
    n_cells = len(full_dataset)
    n_val = int(n_cells * val_split)
    n_train = n_cells - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_loader, val_loader, full_dataset.vocabularies


def train_vae(config: VAEConfig,
              data_dir: str,
              output_dir: str,
              hvg_file: Optional[str] = None,
              n_epochs: int = 100,
              batch_size: int = 256,
              device: str = 'cuda',
              val_split: float = 0.1,
              max_cells_per_batch: Optional[int] = None) -> ConditionalVAE:
    """
    Train the VAE model.
    
    Args:
        config: VAE configuration
        data_dir: Directory with batch files from download.py
        output_dir: Directory to save outputs
        hvg_file: Path to HVG gene list file
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
        val_split: Fraction of data for validation
        max_cells_per_batch: Limit cells per batch file
        
    Returns:
        Trained model
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Create data loaders
    train_loader, val_loader, vocabularies = create_data_loaders(
        data_dir=data_dir,
        hvg_file=hvg_file,
        batch_size=batch_size,
        val_split=val_split,
        max_cells_per_batch=max_cells_per_batch
    )
    
    # Update config with actual vocabulary sizes
    config.n_perturbations = len(vocabularies['perturbation'])
    config.n_experiments = len(vocabularies['srx_accession'])
    
    logger.info(f"Updated config with vocabulary sizes:")
    logger.info(f"  Perturbations: {config.n_perturbations}")
    logger.info(f"  Experiments: {config.n_experiments}")
    
    logger.info(f"Training configuration: {config.__dict__}")
    wdblog.config.update(config.__dict__)
    # Log config to wandb
    wdblog.config.update({
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'val_split': val_split,
        'max_cells_per_batch': max_cells_per_batch
    })
    wdblog.config.update(vocabularies)
    wdblog.log({"vocabularies": vocabularies})

    # Create model and trainer
    model = ConditionalVAE(config)
    trainer = VAETrainer(model, config, device)
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Save vocabularies and config
    with open(output_dir / 'vocabularies.json', 'w') as f:
        json.dump(vocabularies, f, indent=2)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for batch in train_pbar:
            losses = trainer.train_step(batch)
            train_losses.append(losses)
            
            # Log to tensorboard
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value, global_step)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'recon': f"{losses['reconstruction_loss']:.4f}",
                'kld': f"{losses['kld_loss']:.4f}"
            })

            # Log to wandb
            wdblog.log({
                'train_loss': losses['total_loss'],
                'reconstruction_loss': losses['reconstruction_loss'],
                'kld_loss': losses['kld_loss'],
                # 'epoch': epoch,
                'global_step': global_step
            })
            
            global_step += 1
        
        # Validation
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]")
        for batch in val_pbar:
            losses = trainer.validate_step(batch)
            val_losses.append(losses)
            
            val_pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'recon': f"{losses['reconstruction_loss']:.4f}",
                'kld': f"{losses['kld_loss']:.4f}"
            })
            wdblog.log({
                'val_loss': losses['total_loss'],
                'val_reconstruction_loss': losses['reconstruction_loss'],
                'val_kld_loss': losses['kld_loss'],
            })

        # Compute epoch averages
        avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
        avg_val_loss = np.mean([l['total_loss'] for l in val_losses])
        
        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('epoch/learning_rate', trainer.optimizer.param_groups[0]['lr'], epoch)

        wdblog.log({
            'epoch': epoch,
            'epoch_avg_train_loss': avg_train_loss,
            'epoch_avg_val_loss': avg_val_loss,
            'epoch_learning_rate': trainer.optimizer.param_groups[0]['lr']
        })
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        trainer.scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config.__dict__,
                'vocabularies': vocabularies
            }, output_dir / 'best_model.pt')
            logger.info(f"Saved new best model with val loss {avg_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config.__dict__,
                'vocabularies': vocabularies
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    writer.close()
    logger.info("Training completed!")
    
    return model


def evaluate_model(model: ConditionalVAE,
                  data_loader: DataLoader,
                  vocabularies: Dict,
                  device: str = 'cuda') -> Dict[str, float]:
    """Evaluate the trained model."""
    model.eval()
    trainer = VAETrainer(model, VAEConfig(), device)
    
    all_losses = []
    
    # Standard reconstruction evaluation
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            losses = trainer.validate_step(batch)
            all_losses.append(losses)
    
    avg_losses = {
        key: np.mean([l[key] for l in all_losses])
        for key in all_losses[0].keys()
    }
    
    # # Cell type prediction evaluation
    # correct_predictions = 0
    # total_predictions = 0
    
    # with torch.no_grad():
    #     for batch in tqdm(data_loader, desc="Evaluating cell type prediction"):
    #         x = batch['x'].to(device)
    #         true_cell_types = batch['cell_type_ids'].to(device)
    #         perturbation_ids = batch['perturbation_ids'].to(device)
    #         experiment_ids = batch['experiment_ids'].to(device)
            
    #         predicted_cell_types = trainer.predict_cell_type(x, perturbation_ids, experiment_ids)
            
    #         correct_predictions += (predicted_cell_types == true_cell_types).sum().item()
    #         total_predictions += len(true_cell_types)
    
    # cell_type_accuracy = correct_predictions / total_predictions
    
    results = {
        **avg_losses,
        # 'cell_type_accuracy': cell_type_accuracy
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Conditional VAE on single-cell RNA-seq data")
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing batch_*.h5 files from download.py')
    parser.add_argument('--output-dir', type=str, default='output/vae_training',
                       help='Output directory for models and logs')
    parser.add_argument('--hvg-file', type=str, default=None,
                       help='File containing HVG gene names (one per line)')
    
    # Model parameters
    parser.add_argument('--input-dim', type=int, default=2000,
                       help='Number of input genes (HVGs)')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='Latent space dimension')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[512, 256],
                       help='Hidden layer dimensions')
    
    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max-cells-per-batch', type=int, default=None,
                       help='Limit cells per batch file (for memory management)')
    
    args = parser.parse_args()
    
    # Create VAE config
    config = VAEConfig(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Train model
    model = train_vae(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hvg_file=args.hvg_file,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        val_split=args.val_split,
        max_cells_per_batch=args.max_cells_per_batch
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
