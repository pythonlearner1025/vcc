#!/usr/bin/env python3
"""
Training script for ST-style Conditional Discrete Diffusion Transformer.

This script implements the ST architecture with:
- Paired control-perturbed cell sets
- Control set cross-attention
- Adaptive masking based on conditioning
"""

import torch
import wandb
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

from models.diffusion import (
    ConditionalModelConfig, 
    ConditionalDiffusionTransformer,
    PartialMaskingDiffusion,
    create_optimizer,
    cosine_lr_schedule,
)
# Updated imports for cross-dataset HVGs
from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs, create_scrna_hvg_dataloader
from dataset.vcc_paired_dataloader import create_train_val_dataloaders

class TokenizedScRNADataset(Dataset):
    """Wrapper around ScRNADatasetWithHVGs that applies tokenization."""
    
    def __init__(self, scrna_dataset: ScRNADatasetWithHVGs, tokenizer):
        self.dataset = scrna_dataset
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]  # Already returns tensor with HVG genes
        # Apply tokenizer to convert continuous expression to discrete tokens
        tokens = self.tokenizer(x)
        return tokens

def create_simple_tokenizer(vocab_size: int = 64, max_value: float = 10000.0):
    """
    Create a simple binning tokenizer for gene expression values.
    
    Args:
        vocab_size: Number of discrete bins (should be 63 + 1 mask token)
        max_value: Maximum expression value to handle
    
    Returns:
        tokenizer: A callable that discretizes expression values
        detokenizer: A callable that converts tokens back to approximate expression values
    """
    class SimpleTokenizer:
        def __init__(self, vocab_size, max_value):
            self.vocab_size = vocab_size - 1  # Reserve last token for [MASK]
            self.max_value = max_value
            self.mask_token = vocab_size - 1
            
            # Define bins for expression values - use log-scale for better distribution
            # Bin 0: exactly 0 (very common in scRNA-seq)
            # Bins 1-(vocab_size-2): log-scale from 0.1 to max_value
            self.bins = torch.zeros(self.vocab_size)
            self.bins[1:] = torch.logspace(
                np.log10(0.1), 
                np.log10(max_value), 
                self.vocab_size - 1
            )
            
        def __call__(self, x):
            """Tokenize expression values into discrete bins."""
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            
            # Handle zero values explicitly
            zero_mask = (x == 0)
            
            # Clip values to max range
            x_clipped = torch.clamp(x, 0, self.max_value)
            
            # Bucketize into bins
            tokens = torch.bucketize(x_clipped, self.bins)
            
            # Ensure zero values map to token 0
            tokens[zero_mask] = 0
            
            return tokens.clamp(0, self.vocab_size - 1)
        
        def detokenize(self, tokens):
            """Convert tokens back to approximate expression values."""
            # Use bin centers for reconstruction
            bin_centers = torch.zeros(self.vocab_size)
            bin_centers[0] = 0.0  # Zero bin
            
            for i in range(1, self.vocab_size - 1):
                bin_centers[i] = (self.bins[i] + self.bins[i+1]) / 2
            bin_centers[-1] = self.bins[-1]  # Last bin uses upper bound
            
            return bin_centers[tokens]
    
    tokenizer = SimpleTokenizer(vocab_size, max_value)
    return tokenizer, tokenizer.detokenize


def train_epoch_st(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ConditionalModelConfig,
    epoch: int,
    global_step: int,
    total_training_steps: int,
    tokenizer = None,
    batch_to_idx: Optional[Dict[str, int]] = None,
    use_control_sets: bool = True,
    mixed_training: bool = True,
    max_steps: Optional[int] = None,
) -> int:
    """
    Train for one epoch using ST-style conditioning.
    
    Args:
        model: The diffusion model
        diffusion: The diffusion process handler
        dataloader: Training data loader (paired sets)
        optimizer: Optimizer
        config: Model configuration
        epoch: Current epoch number
        global_step: Global step counter
        total_training_steps: Total training steps for LR scheduling
        tokenizer: Tokenizer function to convert expression values to tokens (needed for VCC data)
        batch_to_idx: Mapping from batch names to indices for batch conditioning
        use_control_sets: Whether to use control set conditioning
        mixed_training: Whether to mix conditioned and unconditioned batches
        max_steps: Maximum number of steps to run (for debugging), None for full epoch
        
    Returns:
        Updated global step
    """
    model.train()
    epoch_start = time.time()
    epoch_losses = []
    
    total_steps = len(dataloader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        print(f"Debug mode: limiting epoch to {max_steps} steps")
    
    for batch_idx, batch in enumerate(dataloader):
        # Early cutoff for debugging
        if max_steps is not None and batch_idx >= max_steps:
            print(f"Debug cutoff: stopping after {max_steps} steps")
            break
        
        if isinstance(batch, dict) and 'perturbed_expr' in batch:
            # Processing VCC paired data
            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda() if use_control_sets else None
            
            B, S, N = X_pert.shape  # (batch, set_size, n_genes)
            
            # Tokenize the expression data
            # Note: tokenizer is expected to be passed to train_epoch_st
            # Move to CPU for tokenization to avoid device mismatch
            X_pert_cpu = X_pert.cpu()
            X_pert_tokens = torch.zeros_like(X_pert_cpu, dtype=torch.long)
            for b in range(B):
                for s in range(S):
                    X_pert_tokens[b, s] = tokenizer(X_pert_cpu[b, s])
            X_pert_tokens = X_pert_tokens.cuda()
                    
            if X_ctrl is not None:
                X_ctrl_cpu = X_ctrl.cpu()
                X_ctrl_tokens = torch.zeros_like(X_ctrl_cpu, dtype=torch.long)
                for b in range(B):
                    for s in range(S):
                        X_ctrl_tokens[b, s] = tokenizer(X_ctrl_cpu[b, s])
                X_ctrl = X_ctrl_tokens.cuda()
            
            X_pert_flat = X_pert_tokens.view(B * S, N)
            
            # Process conditioning information
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            
            # Compute perturbation signs
            log2fc = batch['log2fc'].cuda()
            if log2fc.dim() > 1:
                log2fc_avg = log2fc.mean(dim=-1)
            else:
                log2fc_avg = log2fc
            perturb_sign = torch.sign(log2fc_avg).long()
            
            # Expand conditioning to match batch
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            log2fc_avg = log2fc_avg.unsqueeze(1).expand(-1, S).reshape(-1)
            perturb_sign = perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1)
            
            # Process batch information
            if batch_to_idx is not None and 'pert_batches' in batch:
                pert_batch_names = batch['pert_batches']  # List of lists: [batch_per_sample][cell_in_set]
                batch_indices = []
                for sample_batches in pert_batch_names:
                    for batch_name in sample_batches:
                        batch_idx = batch_to_idx.get(batch_name, 0)  # Default to 0 if not found
                        batch_indices.append(batch_idx)
                batch_indices = torch.tensor(batch_indices, device='cuda')
            else:
                batch_indices = None
            
            if mixed_training and torch.rand(1).item() < 0.5:
                # Unconditioned batch
                X_ctrl = None
                target_gene_idx = None
                log2fc = None
                perturb_sign = None
                batch_indices = None  # No batch conditioning for unconditioned training
                tokens = X_pert_flat
            else:
                # Conditioned batch
                tokens = X_pert_flat
                log2fc = log2fc_avg
                # batch_indices already computed above
                
                if X_ctrl is not None:
                    # X_ctrl shape: (B, S, N)
                    # We need to repeat each control set S times to match flattened perturbed cells
                    # Each of the S perturbed cells from a batch should see the same control set
                    X_ctrl_expanded = X_ctrl.unsqueeze(1).expand(-1, S, -1, -1)  # (B, S, S, N)
                    X_ctrl = X_ctrl_expanded.reshape(B * S, S, N)  # (B*S, S, N)
        else:
            # Processing regular pretraining data
            tokens = batch.cuda()
            B = tokens.shape[0]
            X_ctrl = None
            target_gene_idx = None
            log2fc = None
            perturb_sign = None
            batch_indices = None  # No batch conditioning for pretraining data
        
        loss = diffusion.compute_loss(
            model, 
            tokens,
            control_set=X_ctrl,
            target_gene_idx=target_gene_idx,
            perturb_magnitude=log2fc,
            perturb_sign=perturb_sign,
            batch_idx=batch_indices,
            step=global_step
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_training_steps = config.pretrain_epochs * len(dataloader) + config.finetune_epochs * len(dataloader)
        lr = cosine_lr_schedule(optimizer, global_step, total_training_steps, config)
        
        epoch_losses.append(loss.item())
        
        # Log every 10 steps
        if global_step % 10 == 0:
            avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) > 100 else np.mean(epoch_losses)
            print(f"Epoch {epoch:3d} [{batch_idx+1:4d}/{total_steps:4d}] | "
                  f"Step {global_step:6d} | Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            wandb.log({
                'train_loss': loss.item(),
                'avg_train_loss': avg_loss,
                'learning_rate': lr,
                'epoch': epoch,
                'global_step': global_step,
                'use_control_sets': X_ctrl is not None,
            })
        
        if global_step % config.save_every == 0 and global_step > 0:
            checkpoint_path = f'checkpoint_st_epoch_{epoch}_step_{global_step}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        global_step += 1
    
    epoch_time = time.time() - epoch_start
    avg_epoch_loss = np.mean(epoch_losses)
    steps_completed = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    print(f"Epoch {epoch} completed in {epoch_time:.1f}s | Steps: {steps_completed} | Avg Loss: {avg_epoch_loss:.4f}")
    
    return global_step

def evaluate_validation(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ConditionalModelConfig,
    epoch: int,
    tokenizer = None,
    batch_to_idx: Optional[Dict[str, int]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate validation loss on held-out genes (20% split).
    
    Args:
        model: The diffusion model
        diffusion: The diffusion process handler
        val_dataloader: Validation data loader
        optimizer: Optimizer (not used, kept for consistency)
        config: Model configuration
        epoch: Current epoch number
        tokenizer: Tokenizer function to convert expression values to tokens
        batch_to_idx: Mapping from batch names to indices for batch conditioning
        max_steps: Maximum number of batches to evaluate (for debugging), None for full dataset
    """
    model.eval()
    val_losses = []
    
    total_batches = len(val_dataloader)
    if max_steps is not None:
        total_batches = min(total_batches, max_steps)
        print(f"Debug mode: limiting evaluation to {max_steps} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Early cutoff for debugging
            if max_steps is not None and batch_idx >= max_steps:
                print(f"Debug cutoff: stopping evaluation after {max_steps} batches")
                break
            
            # Process batch data
            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda()
            
            B, S, N = X_pert.shape  # (batch, set_size, n_genes)
            
            # Tokenize the expression data
            X_pert_tokens = torch.zeros_like(X_pert, dtype=torch.long)
            X_ctrl_tokens = torch.zeros_like(X_ctrl, dtype=torch.long)
            for b in range(B):
                for s in range(S):
                    X_pert_tokens[b, s] = tokenizer(X_pert[b, s])
                    X_ctrl_tokens[b, s] = tokenizer(X_ctrl[b, s])
            
            X_pert_flat = X_pert_tokens.view(B * S, N)
            
            # Process conditioning information
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            
            # Compute perturbation signs
            log2fc = batch['log2fc'].cuda()
            if log2fc.dim() > 1:
                log2fc_avg = log2fc.mean(dim=-1)
            else:
                log2fc_avg = log2fc
            perturb_sign = torch.sign(log2fc_avg).long()
            
            # Expand conditioning to match batch
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            log2fc_avg = log2fc_avg.unsqueeze(1).expand(-1, S).reshape(-1)
            perturb_sign = perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1)
            
            # Process batch information
            if batch_to_idx is not None and 'pert_batches' in batch:
                pert_batch_names = batch['pert_batches']
                batch_indices = []
                for sample_batches in pert_batch_names:
                    for batch_name in sample_batches:
                        batch_idx = batch_to_idx.get(batch_name, 0)
                        batch_indices.append(batch_idx)
                batch_indices = torch.tensor(batch_indices, device='cuda')
            else:
                batch_indices = None
            
            # Expand control sets to match flattened perturbed cells
            X_ctrl_expanded = X_ctrl_tokens.unsqueeze(1).expand(-1, S, -1, -1)  # (B, S, S, N)
            X_ctrl_flat = X_ctrl_expanded.reshape(B * S, S, N)  # (B*S, S, N)
            
            # Compute loss
            loss = diffusion.compute_loss(
                model, 
                X_pert_flat,
                control_set=X_ctrl_flat,
                target_gene_idx=target_gene_idx,
                perturb_magnitude=log2fc_avg,
                perturb_sign=perturb_sign,
                batch_idx=batch_indices,
                step=epoch  # Use epoch as step for validation
            )
            
            val_losses.append(loss.item())
    
    # Compute metrics
    avg_val_loss = np.mean(val_losses) if val_losses else 0.0
    metrics = {
        'val_loss': avg_val_loss,
        'val_batches_evaluated': len(val_losses),
        'epoch': epoch
    }
    return metrics


def main():
    """
    MEMORY OPTIMIZATION STRATEGY:
    
    This model faces unique memory challenges due to:
    1. Large sequence length: 2000 genes per cell (vs typical 512-2048 in language models)
    2. Set-based training: Each batch contains batch_size × set_size sequences
    3. Cross-attention: Control sets add additional memory overhead
    
    Memory scaling for self-attention: O(batch_size × set_size × n_heads × seq_len²)
    With seq_len=2000: each sequence pair requires 4M attention operations per head
    
    Current settings:
    - batch_size=2, set_size=16 → 32 total sequences per batch
    - 8 attention heads, 2000 sequence length
    - Memory per batch: ~1GB for attention matrices (manageable)
    
    If you get OOM errors:
    1. Reduce batch_size further (to 1 if needed)
    2. Reduce set_size (but may hurt model performance)
    3. Consider gradient checkpointing
    4. Use multi-query attention (already implemented)
    
    DEBUG MODE:
    For quick testing, set config.debug_max_steps to a small number (e.g., 10-50)
    This will limit both training and evaluation to the specified number of steps.
    """
    # Configuration
    config = ConditionalModelConfig(
        # Model architecture
        dim=256,
        n_head=8,
        n_layer=8,
        ffn_mult=8,
        vocab_size=128,
        n_genes=2000,
        
        # Conditioning
        n_total_genes=2000,  # Using HVG genes only
        gene_embed_dim=128,
        perturb_sign_dim=16,
        perturb_magnitude_dim=64,
        magnitude_clip=5.0,
        
        # Batch conditioning - ENABLED with correct number of batches
        use_batch_conditioning=True,
        n_batches=48,  # From VCC data exploration: 48 unique batches
        batch_embed_dim=64,
        
        # Control set encoder
        control_set_encoder_layers=2,
        control_set_dim_hidden=128,
        # LEGACY, this is now just config.dim
        #control_set_dim_out=128,  # Must match model dim for cross-attention
        
        # Diffusion
        n_timesteps=10,
        mask_ratio=0.30,
        schedule="cosine",
        
        # Training - MEMORY OPTIMIZED
        # Note: With 2000-gene sequences, attention is O(N²) = 4M operations per head per sequence
        # Each batch processes batch_size × set_size = total sequences
        # Memory scales as: total_sequences × n_heads × sequence_length²
        pretrain_batch_size=32,  # Very small due to 2000-token sequences and cross-attention
        vcc_batch_size=1,
        vcc_set_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5000,

        # Pretrain data dir  
        pretrain_data_dir = "/scRNA/processed",
        finetune_data_path = "/vcc_data/adata_Training.h5ad",
        hvg_info_path = "hvg_seuratv3_2000.txt",
        
        # Epochs
        pretrain_epochs=0,
        finetune_epochs=1,
        
        # Logging
        log_every=10,  # Updated to match hardcoded logging frequency
        eval_every=1000,
        save_every=5000,
        vcc_eval_interval=5000,
        
        # Debug - set to None for full training, or number of steps for quick debugging
        debug_pretrain_max_steps = 10,
        debug_finetune_max_steps = 10,
        debug_eval_max_steps = 10 # equivalent to validation set's target perturb genes
    )
    
    # Initialize wandb
    wandb.init(
        project="vcc-st-diffusion",
        config=config.__dict__,
        name=f"st_diffusion_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create model
    model = ConditionalDiffusionTransformer(config).cuda()
    diffusion = PartialMaskingDiffusion(config)
    optimizer = create_optimizer(model, config)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create tokenizer
    tokenizer, detokenizer = create_simple_tokenizer(config.vocab_size)

    # TODO the hvg should be a single .txt file of ENSEMBLIDs
    # TODO change scran_hvg.py to output ENSEMBLIDs
    # TODO normalize counts by UMI before tokenizing in dataloader!

        # Load cross-dataset HVG info to get gene names
    with open(config.hvg_info_path, 'r') as f:
        hvg_gene_ensemble = [line.strip() for line in f.readlines()]

    print("\n=== Creating Dataloaders ===")
    # Create pretrain dataloader using cross-dataset HVGs
    print("Creating scRNA pretrain dataloader with cross-dataset HVGs...")
    scrna_dataset, _ = create_scrna_hvg_dataloader(
        data_dir=config.pretrain_data_dir,
        hvg_genes=hvg_gene_ensemble,
        batch_size=1,  # We'll batch in the wrapper
        shuffle=False,
        num_workers=0,
        use_cache=True
    )
    
    # Wrap with tokenizer
    pretrain_dataset = TokenizedScRNADataset(scrna_dataset, tokenizer)
    
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Pretrain dataset: {len(pretrain_dataset):,} cells, {scrna_dataset.n_hvgs} HVG genes")
    
    # Create VCC train and validation dataloaders
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_train_val_dataloaders(
        adata_path=config.finetune_data_path,
        hvg_gene_ids=hvg_gene_ensemble,
        set_size=config.vcc_set_size,
        batch_size=config.vcc_batch_size,
        n_samples_per_gene_train=10,  # Multiple samples per gene for training
        n_samples_per_gene_val=1,      # Single sample per gene for validation
        train_split=0.8,
        num_workers=4,
        random_seed=42
    )
    
    # Create gene to index mapping
    vcc_gene_to_idx = {gene: idx for idx, gene in enumerate(hvg_gene_ensemble)}
    
    # Create batch name to index mapping for conditioning
    # Get unique batch names from the VCC dataset
    unique_batches = sorted(list(set(vcc_dataset.adata.obs['batch'].values)))
    batch_to_idx = {batch_name: idx for idx, batch_name in enumerate(unique_batches)}
    print(f"Found {len(unique_batches)} unique batches for batch conditioning")
    
    global_step = 0
    
    # Calculate total training steps for learning rate schedule
    pretrain_steps = config.pretrain_epochs * len(pretrain_dataloader) if config.pretrain_epochs > 0 else 0
    finetune_steps = config.finetune_epochs * len(vcc_dataloader)
    total_training_steps = pretrain_steps + finetune_steps
    print(f"\nTotal training steps: {total_training_steps:,} (pretrain: {pretrain_steps:,}, finetune: {finetune_steps:,})")
    
    # Create unique checkpoint directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"checkpoints/run_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"\nSaved config to {config_path}")
    
    # Phase 1: Pretraining on single cells
    if config.pretrain_epochs > 0:
        print(f"\n=== Phase 1: Pretraining on scRNA data ({config.pretrain_epochs} epochs) ===")
        print(f"Using {len(pretrain_dataset):,} cells for pretraining")
        
        for epoch in range(config.pretrain_epochs):
            global_step = train_epoch_st(
                model, diffusion, pretrain_dataloader, optimizer, config,
                epoch, global_step, total_training_steps,
                tokenizer=None,  # Pretrain data is already tokenized
                batch_to_idx=None,  # No batch conditioning for pretraining
                use_control_sets=False,  # No control sets in pretraining
                mixed_training=False,
                max_steps=config.debug_pretrain_max_steps
            )
            
        # Save checkpoint after pretraining
        pretrain_checkpoint = checkpoint_dir / "checkpoint_st_pretrained.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'epoch': config.pretrain_epochs,
            'global_step': global_step,
        }, pretrain_checkpoint)
        print(f"Saved pretrained model to {pretrain_checkpoint}")
    else:
        print("\n=== Skipping Phase 1: Pretraining (pretrain_epochs=0) ===")
    
    # Phase 2: Fine-tuning with control sets
    print("\n=== Phase 2: Fine-tuning with Control Sets ===")
    print(f"Using batch_size={config.pretrain_batch_size} optimized for 2000-gene sequences")
    
    for epoch in range(config.finetune_epochs):
        global_step = train_epoch_st(
            model, diffusion, vcc_dataloader, optimizer, config,
            epoch, global_step, total_training_steps,
            tokenizer=tokenizer,  # VCC data needs tokenization
            batch_to_idx=batch_to_idx,  # Pass batch mapping for conditioning
            use_control_sets=True,
            mixed_training=True,  # Mix conditioned and unconditioned batches
            max_steps=config.debug_finetune_max_steps
        )
        
        # Evaluate on validation set (20% of training genes)
        if epoch % 5 == 0:
            print("\n=== Validation Set Evaluation ===")
            val_metrics = evaluate_validation(
                model, diffusion, val_dataloader, optimizer, config,
                epoch, tokenizer, batch_to_idx,
                max_steps=config.debug_eval_max_steps
            )
            print(f"Validation loss: {val_metrics['val_loss']:.4f} ({val_metrics['val_batches_evaluated']} batches)")
            wandb.log(val_metrics)
            
            # TODO Evaluate true zero-shot performance on unseen genes
    
    # Final save
    final_checkpoint = checkpoint_dir / "checkpoint_st_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': config.pretrain_epochs + config.finetune_epochs,
        'global_step': global_step,
    }, final_checkpoint)
    print(f"\nTraining complete! Final model saved to {final_checkpoint}")
    
    wandb.finish()


if __name__ == "__main__":
    main()