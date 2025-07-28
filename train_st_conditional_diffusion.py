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
    prepare_perturbation_conditioning,
    create_gene_mapping
)
# Updated imports for cross-dataset HVGs
from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs, create_scrna_hvg_dataloader
from dataset import (
    create_vcc_paired_dataloader,
    create_vcc_validation_dataloader,
    VCCDataset
)
from eval import evaluate_on_vcc_validation, log_vcc_metrics, create_vcc_evaluator



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
    use_control_sets: bool = True,
    mixed_training: bool = True,
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
        use_control_sets: Whether to use control set conditioning
        mixed_training: Whether to mix conditioned and unconditioned batches
        
    Returns:
        Updated global step
    """
    print("Setting model to train mode")
    model.train()
    epoch_start = time.time()
    epoch_losses = []
    
    total_steps = len(dataloader)
    print(f"Starting epoch with {total_steps} total steps")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nProcessing batch {batch_idx+1}/{total_steps}")
        
        if isinstance(batch, dict) and 'perturbed_expr' in batch:
            print("Processing VCC paired data")
            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda() if use_control_sets else None
            
            B, S, N = X_pert.shape  # (batch, set_size, n_genes)
            print(f"Batch shape: {B} batches, {S} cells per set, {N} genes")
            
            print("Flattening perturbed expression")
            X_pert_flat = X_pert.view(B * S, N)
            
            print("Processing conditioning information")
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            
            print("Computing perturbation signs")
            log2fc = batch['log2fc'].cuda()
            if log2fc.dim() > 1:
                print("Averaging log2fc across genes")
                log2fc_avg = log2fc.mean(dim=-1)
            else:
                log2fc_avg = log2fc
            perturb_sign = torch.sign(log2fc_avg).long()
            
            print("Expanding conditioning to match batch")
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            log2fc_avg = log2fc_avg.unsqueeze(1).expand(-1, S).reshape(-1)
            perturb_sign = perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1)
            
            if mixed_training and torch.rand(1).item() < 0.5:
                print("Using unconditioned batch")
                X_ctrl = None
                target_gene_idx = None
                log2fc = None
                perturb_sign = None
                tokens = X_pert_flat
            else:
                print("Using conditioned batch")
                tokens = X_pert_flat
                log2fc = log2fc_avg
                
                if X_ctrl is not None:
                    print("Using control sets for cross-attention")
                    # X_ctrl shape: (B, S, N)
                    # We need to repeat each control set S times to match flattened perturbed cells
                    # Each of the S perturbed cells from a batch should see the same control set
                    X_ctrl_expanded = X_ctrl.unsqueeze(1).expand(-1, S, -1, -1)  # (B, S, S, N)
                    X_ctrl = X_ctrl_expanded.reshape(B * S, S, N)  # (B*S, S, N)
        else:
            print("Processing regular pretraining data")
            tokens = batch.cuda()
            B = tokens.shape[0]
            X_ctrl = None
            target_gene_idx = None
            log2fc = None
            perturb_sign = None
        
        print("Computing diffusion loss")
        loss = diffusion.compute_loss(
            model, 
            tokens,
            control_set=X_ctrl,
            target_gene_idx=target_gene_idx,
            perturb_magnitude=log2fc,
            perturb_sign=perturb_sign,
            step=global_step
        )
        
        print("Performing backward pass")
        optimizer.zero_grad()
        loss.backward()
        print("Clipping gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print("Optimizer step")
        optimizer.step()
        
        print("Updating learning rate")
        total_training_steps = config.pretrain_epochs * len(dataloader) + config.finetune_epochs * len(dataloader)
        lr = cosine_lr_schedule(optimizer, global_step, total_training_steps, config)
        
        epoch_losses.append(loss.item())
        
        if global_step % config.log_every == 0:
            print("Logging metrics")
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
            print("Saving checkpoint")
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
    print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_epoch_loss:.4f}")
    
    return global_step


def evaluate_zero_shot(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    val_dataloader: torch.utils.data.DataLoader,
    config: ConditionalModelConfig,
    gene_to_idx: Dict[str, int],
    epoch: int,
) -> Dict[str, float]:
    """
    Evaluate zero-shot perturbation prediction on validation genes.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_genes = []
    with torch.no_grad():
        for batch in val_dataloader:
            # Get control sets
            print(batch.keys())
            X_ctrl = batch['control_expr'].cuda()  # Fixed: was 'X_ctrl_tokens'
            target_genes = batch['target_gene']
            target_gene_idx = batch['target_gene_idx'].cuda()
            log2fc = batch['log2fc'].cuda()
            
            # Compute perturb_sign from log2fc like in training
            if log2fc.dim() > 1:
                log2fc_avg = log2fc.mean(dim=-1)
            else:
                log2fc_avg = log2fc
            perturb_sign = torch.sign(log2fc_avg).long()
            
            B, S, N = X_ctrl.shape
            
            # Generate perturbed cells
            generated = diffusion.p_sample_loop(
                model,
                shape=(B * S, N),  # Generate S cells per perturbation
                control_set=X_ctrl,
                target_gene_idx=target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1),
                perturb_magnitude=log2fc_avg.unsqueeze(1).expand(-1, S).reshape(-1),  # Fixed: use log2fc_avg
                perturb_sign=perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1),
                temperature=1.0,
                device='cuda'
            )
            
            # Reshape back to sets
            generated = generated.view(B, S, N)
            
            # Store results
            all_predictions.append(generated.cpu())
            all_genes.extend(target_genes)
    
    # Compute metrics
    metrics = {
        'zero_shot_genes_evaluated': len(set(all_genes)),
        'epoch': epoch
    }
    
    return metrics


def main():
    # Configuration
    config = ConditionalModelConfig(
        # Model architecture
        dim=256,
        n_head=8,
        n_layer=8,
        ffn_mult=8,
        vocab_size=64,
        n_genes=2000,
        
        # Conditioning
        n_total_genes=2000,  # Using HVG genes only
        gene_embed_dim=128,
        perturb_sign_dim=16,
        perturb_magnitude_dim=64,
        magnitude_clip=5.0,
        
        # Control set encoder
        control_set_encoder_layers=2,
        control_set_dim_hidden=128,
        control_set_dim_out=128,  # Must match model dim for cross-attention
        
        # Diffusion
        n_timesteps=10,
        mask_ratio=0.30,
        schedule="cosine",
        
        # Training
        batch_size=32,  # Smaller batch size due to sets
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5000,

        # Pretrain data dir  
        pretrain_data_dir = "data/scRNA_1e5/processed",
        
        # Epochs
        pretrain_epochs=1,
        finetune_epochs=1,
        
        # Logging
        log_every=50,
        eval_every=1000,
        save_every=5000,
        vcc_eval_interval=5000,
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
    
    # Path to cross-dataset HVG info
    hvg_info_path = "data/vcc_data/cross_dataset_hvg_info.json"
    
    # Check if cross-dataset HVG info exists
    if not Path(hvg_info_path).exists():
        print(f"\nERROR: Cross-dataset HVG info not found at {hvg_info_path}")
        print("Please run: python scripts/compute_cross_dataset_hvgs.py")
        return
    
    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    
    # Create pretrain dataloader using cross-dataset HVGs
    print("Creating scRNA pretrain dataloader with cross-dataset HVGs...")
    scrna_dataset, _ = create_scrna_hvg_dataloader(
        data_dir=config.pretrain_data_dir,
        hvg_info_path=hvg_info_path,
        batch_size=1,  # We'll batch in the wrapper
        shuffle=False,
        num_workers=0,
        use_cache=True
    )
    
    # Wrap with tokenizer
    pretrain_dataset = TokenizedScRNADataset(scrna_dataset, tokenizer)
    
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Pretrain dataset: {len(pretrain_dataset):,} cells, {scrna_dataset.n_hvgs} HVG genes")
    
    # VCC paired dataloader for fine-tuning (already uses HVGs from hvg_info.json)
    vcc_dataset, vcc_dataloader = create_vcc_paired_dataloader(
        set_size=16,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        num_workers=4,
        match_by_batch=True,
        use_hvgs=True  # Uses hvg_info.json which should now be cross-dataset
    )
    
    # VCC validation dataloader
    val_dataset, val_dataloader = create_vcc_validation_dataloader(
        set_size=16,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        n_samples_per_gene=10,
        use_hvgs=True,  # Uses hvg_info.json which should now be cross-dataset
        num_workers=0  # Disable multiprocessing to avoid worker issues
    )
    
    # Load cross-dataset HVG info to get gene names
    with open(hvg_info_path, 'r') as f:
        hvg_info = json.load(f)
    hvg_genes = hvg_info['hvg_names']
    
    # Create gene to index mapping
    vcc_gene_to_idx = {gene: idx for idx, gene in enumerate(hvg_genes)}
    
    global_step = 0
    
    # Calculate total training steps for learning rate schedule
    pretrain_steps = config.pretrain_epochs * len(pretrain_dataloader) if config.pretrain_epochs > 0 else 0
    finetune_steps = config.finetune_epochs * len(vcc_dataloader)
    total_training_steps = pretrain_steps + finetune_steps
    print(f"\nTotal training steps: {total_training_steps:,} (pretrain: {pretrain_steps:,}, finetune: {finetune_steps:,})")
    
    # Phase 1: Pretraining on single cells
    if config.pretrain_epochs > 0:
        print(f"\n=== Phase 1: Pretraining on scRNA data ({config.pretrain_epochs} epochs) ===")
        print(f"Using {len(pretrain_dataset):,} cells for pretraining")
        print("Gene indices are now consistent with VCC fine-tuning!")
        
        for epoch in range(config.pretrain_epochs):
            global_step = train_epoch_st(
                model, diffusion, pretrain_dataloader, optimizer, config,
                epoch, global_step, total_training_steps,
                use_control_sets=False,  # No control sets in pretraining
                mixed_training=False
            )
            
        # Save checkpoint after pretraining
        pretrain_checkpoint = 'checkpoint_st_pretrained.pt'
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
    print("Using same HVG gene indices as pretraining - transfer learning enabled!")
    
    for epoch in range(config.pretrain_epochs, config.pretrain_epochs + config.finetune_epochs):
        global_step = train_epoch_st(
            model, diffusion, vcc_dataloader, optimizer, config,
            epoch, global_step, total_training_steps,
            use_control_sets=True,
            mixed_training=True  # Mix conditioned and unconditioned batches
        )
        
        # Evaluate zero-shot performance
        if epoch % 5 == 0:
            print("\n=== Zero-shot Evaluation ===")
            metrics = evaluate_zero_shot(
                model, diffusion, val_dataloader, config,
                vcc_gene_to_idx, epoch
            )
            
            print(f"Zero-shot genes evaluated: {metrics['zero_shot_genes_evaluated']}")
            wandb.log(metrics)
    
    # Final save
    final_checkpoint = 'checkpoint_st_final.pt'
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