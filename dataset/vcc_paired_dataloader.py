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
from dataset.collators import VCCCollator

class VCCPairedDataset(Dataset):
    """
    Dataset for VCC paired control-perturbation experiments.
    
    Creates matched sets of perturbed and control cells with batch 
    matching between control and perturbed cells
    """
    
    def __init__(
        self,
        adata_path: str = "data/competition_support/competition_train.h5",
        hvg_gene_ids: List[str] = None,
        set_size: int = 16,
        n_samples_per_gene: int = 10,
        train_split: float = 0.8,
        is_train: bool = True,  # Train or val
        random_seed: int = 42,
        normalize: bool = True,
        blacklist_path: str = None,
    ):
        # Save key construction arguments for downstream evaluation utilities
        self.adata_path = adata_path
        self.set_size = set_size
        self.n_samples_per_gene = n_samples_per_gene
        self.random_seed = random_seed
        self.is_train = is_train
        self.normalize = normalize
        
        # Load blacklist genes if provided
        self.blacklist_genes = set()
        if blacklist_path is not None:
            try:
                with open(blacklist_path, 'r') as fh:
                    self.blacklist_genes = {line.strip() for line in fh if line.strip()}
                print(f"Loaded {len(self.blacklist_genes)} genes from blacklist file {blacklist_path}")
            except FileNotFoundError:
                warnings.warn(f"Blacklist file {blacklist_path} not found. Continuing without blacklist.")
        
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
        else:
            # Sanity check for log1p normalization using the same approach as scrna_hvg_dataset.py
            pass
            '''
            X = self.adata.X
            
            # Basic range check - should be non-negative with reasonable max
            X_min, X_max = (X.min(), X.max()) if not hasattr(X, 'toarray') else (X.min(), X.max())
            print(f"  Expression value range: [{X_min:.6f}, {X_max:.2f}]")
            
            # Print mean statistics across the cells
            mean_values = self.adata.X.mean(axis=0)
            print(f"Mean expression values across cells: {mean_values}")
            print(f"Overall mean expression value: {mean_values.mean():.2f}")
            '''

        # Create gene name <-> ID mappings
        self.gene_name_to_id = dict(zip(self.adata.var.index, self.adata.var['gene_id']))
        self.gene_id_to_name = dict(zip(self.adata.var['gene_id'], self.adata.var.index))
        
        # Filter to HVG genes if provided, otherwise keep all genes
        if hvg_gene_ids is not None:
            self._filter_to_hvgs(hvg_gene_ids)
        else:
            self.hvg_gene_ids = list(self.adata.var['gene_id'])
            self.hvg_gene_names = list(self.adata.var.index)
        
        # Get all perturbed genes (excluding non-targeting)
        all_perturbed_genes = [g for g in self.adata.obs['target_gene'].unique() 
                               if g != 'non-targeting']
        
        # Determine train/validation gene sets
        if self.blacklist_genes:
            if is_train:
                # Training uses all genes except blacklist for H1 (handled later per cell type)
                self.target_genes = all_perturbed_genes
            else:
                # Validation focuses only on blacklist genes
                self.target_genes = [g for g in all_perturbed_genes if g in self.blacklist_genes]
        else:
            # Fallback to random 80:20 split if no blacklist provided
            np.random.seed(random_seed)
            np.random.shuffle(all_perturbed_genes)
            split_idx = int(len(all_perturbed_genes) * train_split)
            self.target_genes = all_perturbed_genes[:split_idx] if is_train else all_perturbed_genes[split_idx:]
        
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
        """Pre-compute sample indices for each gene / cell-type combination."""
        self.samples = []
        skipped_entries = []

        obs = self.adata.obs  # shorthand
        rng = np.random.default_rng(self.random_seed)

        for gene in self.target_genes:
            # Ignore genes that are not in HVG list
            if gene not in self.gene_to_hvg_idx:
                skipped_entries.append((gene, "not in HVG"))
                continue

            gene_mask = obs['target_gene'] == gene
            if gene_mask.sum() == 0:
                skipped_entries.append((gene, "no perturbed cells"))
                continue

            # iterate over each cell_type separately so that control / perturb pairs share the same type
            for cell_type in obs.loc[gene_mask, 'cell_type'].unique():
                ct_mask = gene_mask & (obs['cell_type'] == cell_type)
                in_blacklist = gene in self.blacklist_genes

                # Apply blacklist logic (only affects H1)
                if self.is_train and in_blacklist and cell_type == 'ARC_H1':
                    continue  # skip H1 blacklist genes in training
                if (not self.is_train):
                    # Validation must be H1 + blacklist genes only
                    if not (in_blacklist and cell_type == 'ARC_H1'):
                        continue

                pert_cells = obs.index[ct_mask].values
                # If there are no perturbed cells for this gene / cell-type combo, skip it entirely
                if len(pert_cells) == 0:
                    skipped_entries.append((f"{gene}/{cell_type}", "no perturbed cells"))
                    continue

                # batches where this gene was perturbed within this cell_type
                pert_batches = obs.loc[ct_mask, 'batch'].unique()

                # control cells: non-targeting, same batches, same cell_type
                ctrl_mask = (
                    (obs['target_gene'] == 'non-targeting') &
                    (obs['batch'].isin(pert_batches)) &
                    (obs['cell_type'] == cell_type)
                )
                ctrl_cells = obs.index[ctrl_mask].values
                # If there are no control cells available, skip this entry
                if len(ctrl_cells) == 0:
                    skipped_entries.append((f"{gene}/{cell_type}", "no control cells"))
                    continue

                for _ in range(self.n_samples_per_gene):
                    # Dynamically allow sampling **with replacement** when the available cells are fewer than the
                    # desired set size. This guarantees that we still create a set of exactly ``self.set_size``
                    # cells while making use of every unique cell that exists.
                    replace_pert = len(pert_cells) < self.set_size
                    replace_ctrl = len(ctrl_cells) < self.set_size
                    pert_sample = rng.choice(pert_cells, self.set_size, replace=replace_pert)
                    ctrl_sample = rng.choice(ctrl_cells, self.set_size, replace=replace_ctrl)
                    self.samples.append({
                        'gene': gene,
                        'gene_idx': self.gene_to_hvg_idx[gene],
                        'cell_type': cell_type,
                        'pert_cells': pert_sample,
                        'ctrl_cells': ctrl_sample,
                        'pert_batches': obs.loc[pert_sample, 'batch'].values,
                        'ctrl_batches': obs.loc[ctrl_sample, 'batch'].values,
                    })

        if skipped_entries:
            print(f"  -> Skipped {len(skipped_entries)} gene/cell-type combos:")
            for entry, reason in skipped_entries[:5]:
                print(f"     - {entry}: {reason}")
        
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
        
        # Calculate log2 fold change for the target gene BEFORE exponentiating
        gene_idx_in_adata = self.adata.var.index.get_loc(sample['gene'])
        pert_gene_expr_log = pert_expr[:, gene_idx_in_adata].mean()
        ctrl_gene_expr_log = ctrl_expr[:, gene_idx_in_adata].mean()
        log2fc = (pert_gene_expr_log - ctrl_gene_expr_log) / np.log(2)

        pert_expr = torch.from_numpy(pert_expr).float()
        ctrl_expr = torch.from_numpy(ctrl_expr).float()

        return {
            'perturbed_expr': pert_expr,
            'control_expr': ctrl_expr,
            'target_gene': sample['gene'],
            'target_gene_idx': sample['gene_idx'],
            'log2fc': torch.tensor(float(log2fc), dtype=torch.float32),
            'pert_batches': sample['pert_batches'].tolist(),
            'ctrl_batches': sample['ctrl_batches'].tolist(),
            'cell_type': sample['cell_type'],
        }


def create_vcc_paired_dataloader(
    adata_path: str = "data/competition_support/competition_train.h5",
    hvg_gene_ids: List[str] = None,
    set_size: int = 16,
    n_samples_per_gene: int = 10,
    train_split: float = 0.8,
    is_train: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
    random_seed: int = 42,
    tokenizer=None,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    normalize: bool = True,
    blacklist_path: str = None
) -> Tuple[VCCPairedDataset, DataLoader]:
   
    dataset = VCCPairedDataset(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
        n_samples_per_gene=n_samples_per_gene,
        train_split=train_split,
        is_train=is_train,
        random_seed=random_seed,
        normalize=normalize,
        blacklist_path=blacklist_path
    )
    
    # Build a collate function that performs tokenisation and other CPU-heavy work
    collate_fn = VCCCollator(tokenizer, dataset.batch_to_idx, set_size) if tokenizer else None

    dataloader_kwargs = dict(
        batch_size=1,
        shuffle=shuffle and is_train,  # Only shuffle training data
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    assert collate_fn is not None
    dataloader_kwargs['collate_fn'] = collate_fn
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    return dataset, dataloader


def create_vcc_train_val_dataloaders(
    *,
    adata_path: str = "data/competition_support/competition_train.h5",
    hvg_gene_ids: List[str] = None,
    set_size: int = 16,
    n_samples_per_gene_train: int = 10,
    n_samples_per_gene_val: int = 1,
    train_split: float = 0.8,
    num_workers: int = 4,
    tokenizer=None,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    random_seed: int = 42,
    normalize: bool = True,
    blacklist_path: str = None,
) -> Tuple[Tuple[VCCPairedDataset, DataLoader], Tuple[VCCPairedDataset, DataLoader]]:

    # Training dataloader
    train_dataset, train_dataloader = create_vcc_paired_dataloader(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
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
        blacklist_path=blacklist_path,
    )
    
    # Validation dataloader
    val_dataset, val_dataloader = create_vcc_paired_dataloader(
        adata_path=adata_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=set_size,
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
        blacklist_path=blacklist_path,
    )
    
    return (train_dataset, train_dataloader), (val_dataset, val_dataloader)
