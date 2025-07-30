#!/usr/bin/env python3
"""
Training script for the Conditional VAE on single-cell RNA-seq data.

This script demonstrates how to train the VAE using the batch file format
produced by download.py, with proper handling of metadata and conditioning.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig, VAETrainer, create_metadata_vocabularies, encode_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScRNAVAEDataset(Dataset):
    """
    PyTorch Dataset for VAE training using the batch file format from download.py.
    """
    
    def __init__(self, 
                 data_dir: str,
                 hvg_genes: Optional[List[str]] = None,
                 max_cells_per_batch: Optional[int] = None,
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing batch_*.h5 files from download.py
            hvg_genes: List of HVG gene names to filter to
            max_cells_per_batch: Limit cells per batch file (for memory management)
            transform: Optional transform to apply to gene expression data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_cells_per_batch = max_cells_per_batch
        
        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.h5"))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        logger.info(f"Found {len(self.batch_files)} batch files")
        
        # Load gene information from first batch
        with h5py.File(self.batch_files[0], 'r') as f:
            all_genes = [g.decode() for g in f['genes'][:]]
        
        # Filter to HVG genes if provided
        if hvg_genes is not None:
            self.gene_indices = [i for i, gene in enumerate(all_genes) if gene in hvg_genes]
            self.genes = [all_genes[i] for i in self.gene_indices]
            logger.info(f"Filtered to {len(self.genes)} HVG genes")
        else:
            self.gene_indices = list(range(len(all_genes)))
            self.genes = all_genes
            logger.info(f"Using all {len(self.genes)} genes")
        
        # Build index of all cells across batches
        self._build_cell_index()
        
        # Load and process metadata to create vocabularies
        self._build_metadata_vocabularies()
        
        logger.info(f"Dataset initialized with {len(self)} cells")
    
    def _build_cell_index(self):
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
        
        # Create synthetic cell types and perturbations for demonstration
        # In practice, you would have this information in your metadata
        np.random.seed(42)
        n_cells = len(metadata_df)
        
        # Simulate cell types (could be inferred from clustering)
        metadata_df['cell_type'] = np.random.choice(
            ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'Dendritic', 'Other'], 
            size=n_cells, 
            p=[0.3, 0.2, 0.1, 0.2, 0.1, 0.1]
        )
        
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
            gene_count = f['obs/gene_count'][local_idx]
            umi_count = f['obs/umi_count'][local_idx]
            srx_accession = f['obs/SRX_accession'][local_idx].decode()
        
        # Convert to tensors
        x = torch.from_numpy(x).float()
        
        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
        
        # Get encoded metadata
        cell_type_id = self.encoded_metadata['cell_type_ids'][idx]
        perturbation_id = self.encoded_metadata['perturbation_ids'][idx]
        experiment_id = self.encoded_metadata['srx_accession_ids'][idx]
        
        return {
            'x': x,
            'cell_type_ids': cell_type_id,
            'perturbation_ids': perturbation_id,
            'experiment_ids': experiment_id,
            'gene_count': torch.tensor(gene_count, dtype=torch.float),
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
    config.n_cell_types = len(vocabularies['cell_type'])
    config.n_perturbations = len(vocabularies['perturbation'])
    config.n_experiments = len(vocabularies['srx_accession'])
    
    logger.info(f"Updated config with vocabulary sizes:")
    logger.info(f"  Cell types: {config.n_cell_types}")
    logger.info(f"  Perturbations: {config.n_perturbations}")
    logger.info(f"  Experiments: {config.n_experiments}")
    
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
        
        # Compute epoch averages
        avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
        avg_val_loss = np.mean([l['total_loss'] for l in val_losses])
        
        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('epoch/learning_rate', trainer.optimizer.param_groups[0]['lr'], epoch)
        
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
    
    # Cell type prediction evaluation
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating cell type prediction"):
            x = batch['x'].to(device)
            true_cell_types = batch['cell_type_ids'].to(device)
            perturbation_ids = batch['perturbation_ids'].to(device)
            experiment_ids = batch['experiment_ids'].to(device)
            
            predicted_cell_types = trainer.predict_cell_type(x, perturbation_ids, experiment_ids)
            
            correct_predictions += (predicted_cell_types == true_cell_types).sum().item()
            total_predictions += len(true_cell_types)
    
    cell_type_accuracy = correct_predictions / total_predictions
    
    results = {
        **avg_losses,
        'cell_type_accuracy': cell_type_accuracy
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
