#!/usr/bin/env python3
"""
VCC Paired DataLoader for creating matched control-perturbation cell sets.

This module creates paired sets of perturbed and control cells from the VCC dataset,
with batch matching and HVG gene filtering.
"""

import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import warnings
from dataset.vcc_collator import VCCCollator

class VCCPairedDataset(Dataset):
    """
    Dataset for VCC paired control-perturbation experiments.
    
    Creates matched sets of perturbed and control cells with batch 
    matching between control and perturbed cells
    """
    
    def __init__(
        self,
        adata_path: str = "data/vcc_data/adata_Training.h5ad",
        hvg_gene_ids: List[str] = None,
        set_size: int = 16,
        n_samples_per_gene: int = 10,
        train_split: float = 0.8,
        is_train: bool = True, # Train or val
        random_seed: int = 42,
        normalize: bool = True,
    ):
  
        self.set_size = set_size
        self.n_samples_per_gene = n_samples_per_gene
        self.random_seed = random_seed
        self.is_train = is_train
        self.normalize = normalize
        
        np.random.seed(random_seed)
        
        # Load data
        print(f"Loading VCC data from {adata_path}...")
        self.adata = sc.read_h5ad(adata_path)

              # Apply normalization if requested
        if self.normalize:
            print("Applying CP10K normalization + log1p transformation...")
            # Make a copy to avoid modifying the original data
            self.adata = self.adata.copy()
            # Normalize each cell to 10,000 total counts
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            # Log1p transformation
            sc.pp.log1p(self.adata)
            print(f"  Normalization complete. Expression values now in range [0, {self.adata.X.max():.2f}]")
        
        # Create gene name <-> ID mappings
        self.gene_name_to_id = dict(zip(self.adata.var.index, self.adata.var['gene_id']))
        self.gene_id_to_name = dict(zip(self.adata.var['gene_id'], self.adata.var.index))
        
        # Filter to HVG genes if provided
        if hvg_gene_ids is not None:
            self._filter_to_hvgs(hvg_gene_ids)
        
        # Get all perturbed genes (excluding non-targeting)
        all_perturbed_genes = [g for g in self.adata.obs['target_gene'].unique() 
                               if g != 'non-targeting']
        
        # Perform 80:20 split on all perturbed genes
        np.random.seed(random_seed)
        np.random.shuffle(all_perturbed_genes)
        split_idx = int(len(all_perturbed_genes) * train_split)
        
        if is_train:
            self.target_genes = all_perturbed_genes[:split_idx]
        else:
            self.target_genes = all_perturbed_genes[split_idx:]
        
        print(f"{'Training' if is_train else 'Validation'} dataset: {len(self.target_genes)} genes")
        
        # Create mapping from genes to their HVG indices
        self.gene_to_hvg_idx = {}
        for idx, gene_id in enumerate(self.hvg_gene_ids):
            gene_name = self.gene_id_to_name.get(gene_id, None)
            if gene_name:
                self.gene_to_hvg_idx[gene_name] = idx
        
        # Debug: Check how many target genes are in HVG list
        genes_in_hvg = [g for g in self.target_genes if g in self.gene_to_hvg_idx]
        print(f"  -> {len(genes_in_hvg)}/{len(self.target_genes)} target genes are in HVG list")
        
        # Prepare samples
        self._prepare_samples()
        print(f"Created {len(self.samples)} samples from {len(self.target_genes)} genes")
        # Build batch name → index mapping once so collate_fn can convert strings fast
        self.unique_batches = sorted(list(set(self.adata.obs['batch'].values)))
        self.batch_to_idx = {name: idx for idx, name in enumerate(self.unique_batches)}
        
    def _filter_to_hvgs(self, hvg_gene_ids: List[str]):
        """Filter adata to only include HVG genes."""
        # Find which HVG genes are in our data
        hvg_in_data = []
        hvg_names = []
        
        for gene_id in hvg_gene_ids:
            if gene_id in self.gene_id_to_name:
                gene_name = self.gene_id_to_name[gene_id]
                hvg_in_data.append(gene_id)
                hvg_names.append(gene_name)
        
        print(f"Found {len(hvg_in_data)}/{len(hvg_gene_ids)} HVG genes in data")
        
        # Filter adata to only these genes
        self.adata = self.adata[:, hvg_names]
        self.hvg_gene_ids = hvg_in_data
        self.hvg_gene_names = hvg_names
        
        # Update gene mappings
        self.gene_name_to_id = dict(zip(self.adata.var.index, self.adata.var['gene_id']))
        self.gene_id_to_name = dict(zip(self.adata.var['gene_id'], self.adata.var.index))
        
    def _prepare_samples(self):
        """Pre-compute sample indices for each gene."""
        self.samples = []
        skipped_genes = []
        
        for gene in self.target_genes:
            # Skip if gene not in our HVG list
            if gene not in self.gene_to_hvg_idx:
                skipped_genes.append((gene, "not in HVG"))
                continue
                
            # Get perturbed cells for this gene
            pert_mask = self.adata.obs['target_gene'] == gene
            pert_cells = self.adata.obs.index[pert_mask].values
            
            if len(pert_cells) < self.set_size:
                skipped_genes.append((gene, f"only {len(pert_cells)} perturbed cells"))
                warnings.warn(f"Gene {gene} has only {len(pert_cells)} cells, skipping")
                continue
            
            # Get batches where this gene was perturbed
            pert_batches = self.adata.obs.loc[pert_mask, 'batch'].unique()
            
            # Get control cells from same batches
            ctrl_mask = (self.adata.obs['target_gene'] == 'non-targeting') & \
                        (self.adata.obs['batch'].isin(pert_batches))
            ctrl_cells = self.adata.obs.index[ctrl_mask].values
            
            if len(ctrl_cells) < self.set_size:
                skipped_genes.append((gene, f"only {len(ctrl_cells)} control cells"))
                warnings.warn(f"Not enough control cells for gene {gene}, skipping")
                continue
            
            # Create n_samples_per_gene different sets
            for _ in range(self.n_samples_per_gene):
                # Sample cells
                pert_sample = np.random.choice(pert_cells, self.set_size, replace=False)
                ctrl_sample = np.random.choice(ctrl_cells, self.set_size, replace=False)
                
                self.samples.append({
                    'gene': gene,
                    'gene_idx': self.gene_to_hvg_idx[gene],
                    'pert_cells': pert_sample,
                    'ctrl_cells': ctrl_sample,
                    'pert_batches': self.adata.obs.loc[pert_sample, 'batch'].values,
                    'ctrl_batches': self.adata.obs.loc[ctrl_sample, 'batch'].values,
                })
        
        if skipped_genes:
            print(f"  -> Skipped {len(skipped_genes)} genes:")
            for gene, reason in skipped_genes[:5]:  # Show first 5
                print(f"     - {gene}: {reason}")
            if len(skipped_genes) > 5:
                print(f"     ... and {len(skipped_genes) - 5} more")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get expression data
        pert_expr = self.adata[sample['pert_cells'], :].X
        ctrl_expr = self.adata[sample['ctrl_cells'], :].X
        
        # Convert to dense if sparse
        if hasattr(pert_expr, 'toarray'):
            pert_expr = pert_expr.toarray()
        if hasattr(ctrl_expr, 'toarray'):
            ctrl_expr = ctrl_expr.toarray()
        
        # Convert to torch tensors
        pert_expr = torch.from_numpy(pert_expr).float()
        ctrl_expr = torch.from_numpy(ctrl_expr).float()
        
        # Calculate log2 fold change for the target gene
        gene_idx_in_adata = self.adata.var.index.get_loc(sample['gene'])

            
        if self.normalize:
            # For normalized data, calculate fold change in log space
            # Since data is already log-transformed, we can subtract
            pert_gene_expr = pert_expr[:, gene_idx_in_adata].mean()
            ctrl_gene_expr = ctrl_expr[:, gene_idx_in_adata].mean()
            # Convert from natural log to log2: log2(x/y) = (ln(x) - ln(y)) / ln(2)
            log2fc = (pert_gene_expr - ctrl_gene_expr) / np.log(2)
        else:
            pert_gene_expr = pert_expr[:, gene_idx_in_adata].mean() + 1e-6
            ctrl_gene_expr = ctrl_expr[:, gene_idx_in_adata].mean() + 1e-6
            log2fc = np.log2(pert_gene_expr / ctrl_gene_expr)
        return {
            'perturbed_expr': pert_expr,
            'control_expr': ctrl_expr,
            'target_gene': sample['gene'],
            'target_gene_idx': sample['gene_idx'],
            'log2fc': torch.tensor(float(log2fc), dtype=torch.float32),
            'pert_batches': sample['pert_batches'].tolist(),
            'ctrl_batches': sample['ctrl_batches'].tolist(),
        }


def create_vcc_paired_dataloader(
    adata_path: str = "data/vcc_data/adata_Training.h5ad",
    hvg_gene_ids: List[str] = None,
    set_size: int = 16,
    batch_size: int = 4,
    n_samples_per_gene: int = 10,
    train_split: float = 0.8,
    is_train: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
    random_seed: int = 42,
    tokenizer=None,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> Tuple[VCCPairedDataset, DataLoader]:
   
    dataset = VCCPairedDataset(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
        n_samples_per_gene=n_samples_per_gene,
        train_split=train_split,
        is_train=is_train,
        random_seed=random_seed,
    )
    
    # Build a collate function that performs tokenisation and other CPU-heavy work
    collate_fn = VCCCollator(tokenizer, dataset.batch_to_idx, set_size) if tokenizer else None

    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle and is_train,  # Only shuffle training data
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    if collate_fn is not None:
        dataloader_kwargs['collate_fn'] = collate_fn
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    return dataset, dataloader


def create_train_val_dataloaders(
    *,
    adata_path: str = "data/vcc_data/adata_Training.h5ad",
    hvg_gene_ids: List[str] = None,
    set_size: int = 16,
    batch_size: int = 4,
    n_samples_per_gene_train: int = 10,
    n_samples_per_gene_val: int = 1,
    train_split: float = 0.8,
    num_workers: int = 4,
    tokenizer=None,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    random_seed: int = 42,
    normalize: bool = True,
) -> Tuple[Tuple[VCCPairedDataset, DataLoader], Tuple[VCCPairedDataset, DataLoader]]:

    # Training dataloader
    train_dataset, train_dataloader = create_vcc_paired_dataloader(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
        batch_size=batch_size,
        n_samples_per_gene=n_samples_per_gene_train,
        train_split=train_split,
        is_train=True,
        num_workers=num_workers,
        shuffle=True,
        random_seed=random_seed,
        tokenizer=tokenizer,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        normalize=normalize,
    )
    
    # Validation dataloader
    val_dataset, val_dataloader = create_vcc_paired_dataloader(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
        batch_size=batch_size,
        n_samples_per_gene=n_samples_per_gene_val,
        train_split=train_split,
        is_train=False,
        num_workers=0,  # Disable multiprocessing for validation
        shuffle=False,
        random_seed=random_seed,
        tokenizer=tokenizer,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        normalize=normalize,
    )
    
    return (train_dataset, train_dataloader), (val_dataset, val_dataloader)
