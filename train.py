#!/usr/bin/env python3
"""
Example usage of the downloaded scRNA-seq data for discrete diffusion transformer training.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from dataset import ScRNADataset, create_dataloader


def compute_hvgs(data_dir: str, n_hvgs: int = 2000, cache_file: str = "hvg_info.json") -> Dict:
    """
    Compute highly variable genes across the entire dataset.
    
    Returns:
        Dict containing:
        - hvg_indices: List of gene indices for HVGs
        - hvg_names: List of gene names for HVGs
        - gene_to_hvg_idx: Dict mapping gene index to HVG index
        - statistics: Dict with variance/mean info
    """
    cache_path = Path(data_dir).parent / cache_file
    
    # Check if already computed
    if cache_path.exists():
        print(f"Loading precomputed HVGs from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"Computing top {n_hvgs} highly variable genes...")
    
    # Load gene list
    with open(Path(data_dir).parent / "config.json", 'r') as f:
        config = json.load(f)
    gene_list = config['gene_list']
    n_genes = len(gene_list)
    
    # Accumulate statistics across all batches
    gene_means = np.zeros(n_genes)
    gene_vars = np.zeros(n_genes)
    gene_counts = np.zeros(n_genes)  # Number of non-zero cells per gene
    total_cells = 0
    
    # Process each batch
    batch_files = sorted(Path(data_dir).glob("batch_*.h5"))
    for batch_file in tqdm(batch_files, desc="Computing gene statistics"):
        with h5py.File(batch_file, 'r') as f:
            expression = f['X'][:]  # Changed from 'expression' to 'X'
            
            # Update statistics
            batch_mean = expression.mean(axis=0)
            batch_var = expression.var(axis=0)
            batch_nonzero = (expression > 0).sum(axis=0)
            
            # Online update for mean and variance
            n = expression.shape[0]
            gene_means = (gene_means * total_cells + batch_mean * n) / (total_cells + n)
            gene_vars = (gene_vars * total_cells + batch_var * n) / (total_cells + n)
            gene_counts += batch_nonzero
            total_cells += n
    
    # Compute dispersion (variance/mean ratio) - standard HVG metric
    # Avoid division by zero
    dispersions = np.divide(gene_vars, gene_means, 
                           out=np.zeros_like(gene_vars), 
                           where=gene_means > 0)
    
    # Filter genes expressed in at least 0.1% of cells
    min_cells = int(0.001 * total_cells)
    expressed_mask = gene_counts >= min_cells
    
    # Set dispersion to -1 for non-expressed genes
    dispersions[~expressed_mask] = -1
    
    # Get top HVGs by dispersion
    hvg_indices = np.argsort(dispersions)[-n_hvgs:].tolist()
    hvg_indices.reverse()  # Highest dispersion first
    
    # Create mappings
    hvg_names = [gene_list[i] for i in hvg_indices]
    gene_to_hvg_idx = {gene_idx: hvg_idx for hvg_idx, gene_idx in enumerate(hvg_indices)}
    
    # Prepare result
    hvg_info = {
        'hvg_indices': hvg_indices,
        'hvg_names': hvg_names,
        'gene_to_hvg_idx': gene_to_hvg_idx,
        'statistics': {
            'total_cells_analyzed': total_cells,
            'mean_dispersion': float(dispersions[hvg_indices].mean()),
            'min_dispersion': float(dispersions[hvg_indices].min()),
            'max_dispersion': float(dispersions[hvg_indices].max()),
            'mean_expression': [float(gene_means[i]) for i in hvg_indices[:10]],  # First 10
            'percent_cells_expressed': [float(100 * gene_counts[i] / total_cells) 
                                      for i in hvg_indices[:10]]
        }
    }
    
    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(hvg_info, f, indent=2)
    
    print(f"Selected {n_hvgs} HVGs from {len(expressed_mask[expressed_mask])} expressed genes")
    print(f"Top 5 HVGs: {hvg_names[:5]}")
    print(f"Dispersion range: {hvg_info['statistics']['min_dispersion']:.2f} - "
          f"{hvg_info['statistics']['max_dispersion']:.2f}")
    
    return hvg_info


class HVGDataset(ScRNADataset):
    """Dataset that returns only HVG features with smart gene selection."""
    
    def __init__(self, data_dir: str, n_hvgs: int = 2000, transform=None):
        super().__init__(data_dir, transform)
        
        # Compute or load HVGs
        self.hvg_info = compute_hvgs(data_dir, n_hvgs)
        self.hvg_indices = set(self.hvg_info['hvg_indices'])
        self.gene_to_hvg = self.hvg_info['gene_to_hvg_idx']
        self.n_hvgs = n_hvgs
        
        print(f"\nHVG Dataset initialized:")
        print(f"  Original genes: {self.n_genes:,}")
        print(f"  HVG genes: {self.n_hvgs:,}")
        print(f"  Reduction: {100 * (1 - self.n_hvgs/self.n_genes):.1f}%")
    
    def __getitem__(self, idx):
        """Get a cell with HVG selection."""
        # Find which batch this index belongs to
        batch_idx = np.searchsorted(self.batch_starts[1:], idx, side='right')
        local_idx = idx - self.batch_starts[batch_idx]
        
        with h5py.File(self.batch_files[batch_idx], 'r') as f:
            # Get full expression vector
            full_expression = f['X'][local_idx]  # Changed from 'expression' to 'X'
            
            # Get metadata
            cell_metadata = {
                'cell_id': f['cells'][local_idx].decode('utf-8') if 'cells' in f else f'cell_{idx}',
                'experiment_id': f['obs/SRX_accession'][local_idx].decode('utf-8') if 'obs/SRX_accession' in f else 'unknown',
                'batch_idx': batch_idx,
                'local_idx': local_idx
            }
        
        # Convert to tensor
        full_expression = torch.from_numpy(full_expression).float()
        
        # Smart HVG selection
        hvg_expression = self._select_hvg_features(full_expression)
        
        if self.transform:
            hvg_expression = self.transform(hvg_expression)
        
        return hvg_expression, cell_metadata
    
    def _select_hvg_features(self, full_expression: torch.Tensor) -> torch.Tensor:
        """
        Select HVG features with smart fallback for cells with few HVGs expressed.
        
        Strategy:
        1. Select all HVGs that are expressed in this cell
        2. If < n_hvgs, fill remaining slots with top expressed non-HVG genes
        """
        hvg_expression = torch.zeros(self.n_hvgs)
        
        # Get expressed genes in this cell
        nonzero_indices = torch.nonzero(full_expression).squeeze(-1).tolist()
        
        # Separate into HVG and non-HVG
        hvg_in_cell = []
        non_hvg_in_cell = []
        
        for gene_idx in nonzero_indices:
            if gene_idx in self.gene_to_hvg:
                hvg_idx = self.gene_to_hvg[gene_idx]
                hvg_in_cell.append((hvg_idx, gene_idx))
            else:
                non_hvg_in_cell.append(gene_idx)
        
        # Fill HVG values
        for hvg_idx, gene_idx in hvg_in_cell:
            hvg_expression[hvg_idx] = full_expression[gene_idx]
        
        # If we have fewer than n_hvgs expressed, fill with top non-HVG genes
        n_filled = len(hvg_in_cell)
        if n_filled < self.n_hvgs and non_hvg_in_cell:
            # Sort non-HVG genes by expression value
            non_hvg_values = [(idx, full_expression[idx].item()) for idx in non_hvg_in_cell]
            non_hvg_values.sort(key=lambda x: x[1], reverse=True)
            
            # Fill remaining slots
            n_to_fill = min(len(non_hvg_values), self.n_hvgs - n_filled)
            for i in range(n_to_fill):
                gene_idx, value = non_hvg_values[i]
                # Place in the next available slot
                hvg_expression[n_filled + i] = value
        
        return hvg_expression


def create_hvg_dataloader(data_dir: str, batch_size: int = 32, 
                         n_hvgs: int = 2000, shuffle: bool = True, 
                         num_workers: int = 0):
    """Create a dataloader for HVG features."""
    dataset = HVGDataset(data_dir, n_hvgs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_config(data_dir: str):
    """Load dataset configuration."""
    with open(Path(data_dir).parent / "config.json", 'r') as f:
        return json.load(f)


def create_tokenizer(max_value: int, strategy: str = "direct"):
    """Create a tokenizer for gene expression values."""
    
    if strategy == "direct":
        # Direct mapping: each count is a token
        def tokenize(x):
            return x.long()
        
        def detokenize(tokens):
            return tokens.float()
        
        vocab_size = max_value + 1
        
    elif strategy == "binned":
        # Bin counts into discrete levels
        bins = torch.tensor([0, 1, 5, 10, 50, 100, 500, 1000, 5000])
        
        def tokenize(x):
            return torch.bucketize(x, bins)
        
        def detokenize(tokens):
            # Return bin centers
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_centers = torch.cat([torch.tensor([0.]), bin_centers, torch.tensor([bins[-1].float()])])
            return bin_centers[tokens]
        
        vocab_size = len(bins) + 1
        
    elif strategy == "log":
        # Log-scale tokenization
        def tokenize(x):
            return torch.log1p(x).long()
        
        def detokenize(tokens):
            return torch.expm1(tokens.float())
        
        vocab_size = int(np.log1p(max_value)) + 1
    
    else:
        raise ValueError(f"Unknown tokenization strategy: {strategy}")
    
    return tokenize, detokenize, vocab_size


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""
    
    def __init__(self, vocab_size: int, n_genes: int, d_model: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_genes = n_genes
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Position embedding for genes
        self.pos_embed = nn.Parameter(torch.randn(1, n_genes, d_model))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=256),
            num_layers=1
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens):
        # tokens: [batch_size, n_genes]
        x = self.token_embed(tokens)  # [batch_size, n_genes, d_model]
        x = x + self.pos_embed
        
        # Transformer expects [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # [batch_size, n_genes, vocab_size]
        
        return logits


def train_step(model, batch, tokenize, criterion, optimizer):
    """Single training step."""
    x, meta = batch
    
    # Tokenize
    tokens = tokenize(x)
    
    # Forward pass
    logits = model(tokens)
    
    # Compute loss (predict each gene)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    # Configuration
    data_dir = "data/scRNA/processed"
    batch_size = 32
    learning_rate = 1e-4
    n_epochs = 10
    n_hvgs = 2000  # Number of highly variable genes
    tokenization_strategy = "binned"  # "direct", "binned", or "log"
    
    # Load config
    config = load_config(data_dir)
    print(f"Dataset info:")
    print(f"  Total cells: {config['total_cells_downloaded']:,}")
    print(f"  Total genes: {len(config['gene_list']):,}")
    print(f"  Max expression: {config['max_expression_value']}")
    print(f"  Sparsity: {config['sparsity']:.1%}")
    
    # Create tokenizer
    tokenize, detokenize, vocab_size = create_tokenizer(
        config['max_expression_value'], 
        tokenization_strategy
    )
    print(f"\nTokenization: {tokenization_strategy}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create HVG dataset and dataloader
    dataloader = create_hvg_dataloader(data_dir, batch_size=batch_size, n_hvgs=n_hvgs)
    dataset = dataloader.dataset
    print(f"\nDataset size: {len(dataset):,} cells")
    print(f"Input dimension: {n_hvgs} genes (reduced from {len(config['gene_list']):,})")
    
    # Create model with HVG dimension
    model = SimpleTransformer(
        vocab_size=vocab_size,
        n_genes=n_hvgs,  # Use HVG count instead of all genes
        d_model=128
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(model, batch, tokenize, criterion, optimizer)
            total_loss += loss
            n_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            # Just train on a few batches for demonstration
            if batch_idx >= 10:
                break
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Example: Generate synthetic cells
    print("\nGenerating synthetic cells...")
    model.eval()
    with torch.no_grad():
        # Start with zeros (most genes are not expressed)
        synthetic_tokens = torch.zeros(4, n_hvgs, dtype=torch.long)
        
        # You would implement proper sampling here
        # For now, just get the model predictions
        logits = model(synthetic_tokens)
        predicted_tokens = logits.argmax(dim=-1)
        
        # Detokenize
        synthetic_expression = detokenize(predicted_tokens)
        
        print(f"Generated {synthetic_expression.shape[0]} synthetic cells")
        print(f"Average expression: {synthetic_expression.mean():.2f}")
        print(f"Sparsity: {(synthetic_expression == 0).float().mean():.1%}")
        
    # Show HVG statistics
    print("\nHVG Statistics:")
    hvg_info = dataset.hvg_info
    print(f"Top 10 HVGs: {hvg_info['hvg_names'][:10]}")
    print(f"Mean percent cells expressed (top 10): "
          f"{np.mean(hvg_info['statistics']['percent_cells_expressed']):.1f}%")


if __name__ == "__main__":
    main() 