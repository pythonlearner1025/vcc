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
from dataset import ScRNADataset, create_dataloader
from vcc_paired_dataloader import (
    create_vcc_paired_dataloader,
    create_vcc_validation_dataloader
)
from vcc_dataloader import VCCDataset
from eval import evaluate_on_vcc_validation, log_vcc_metrics, create_vcc_evaluator


class TokenizedScRNADataset(Dataset):
    """Wrapper around ScRNADataset that applies tokenization."""
    
    def __init__(self, data_dir: str, tokenizer, max_genes: Optional[int] = None):
        self.dataset = ScRNADataset(data_dir, max_genes=max_genes)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, meta = self.dataset[idx]
        # Apply tokenizer to convert continuous expression to discrete tokens
        tokens = self.tokenizer(x)
        return tokens


def create_simple_tokenizer(vocab_size: int = 64):
    """
    Create a simple binning tokenizer for gene expression values.
    
    Returns:
        tokenizer: A callable that discretizes expression values
    """
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            # Define bins for expression values
            # We'll use log-scale bins
            self.bins = torch.logspace(-2, 4, vocab_size - 1)  # From 0.01 to 10000
            self.bins = torch.cat([torch.tensor([0.0]), self.bins])
            
        def __call__(self, x):
            """Tokenize expression values into discrete bins."""
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            # Bucketize into bins
            tokens = torch.bucketize(x, self.bins)
            return tokens.clamp(0, self.vocab_size - 1)
    
    return SimpleTokenizer(vocab_size)


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
    model.train()
    epoch_start = time.time()
    epoch_losses = []
    
    total_steps = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to GPU
        if 'perturbed_expr' in batch:
            # VCC paired data
            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda() if use_control_sets else None
            
            # Get batch size and set size
            B, S, N = X_pert.shape  # (batch, set_size, n_genes)
            
            # Flatten for processing
            X_pert_flat = X_pert.view(B * S, N)
            
            # Conditioning
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            
            # Get log2fc and compute sign
            log2fc = batch['log2fc'].cuda()
            # Average log2fc across genes to get a single value per sample
            if log2fc.dim() > 1:
                log2fc_avg = log2fc.mean(dim=-1)
            else:
                log2fc_avg = log2fc
            perturb_sign = torch.sign(log2fc_avg).long()
            
            # Expand conditioning to match flattened batch
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            log2fc_avg = log2fc_avg.unsqueeze(1).expand(-1, S).reshape(-1)
            perturb_sign = perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1)
            
            # Decide if this batch should be conditioned
            if mixed_training and torch.rand(1).item() < 0.5:
                # 50% of batches are unconditioned (control-only)
                X_ctrl = None
                target_gene_idx = None
                log2fc = None
                perturb_sign = None
                tokens = X_pert_flat  # Still use perturbed cells but without conditioning
            else:
                tokens = X_pert_flat
                log2fc = log2fc_avg
                
                # Prepare control sets if using
                if X_ctrl is not None:
                    # Keep control sets in (B, S, N) shape for cross-attention
                    X_ctrl = X_ctrl  # Already in correct shape
        else:
            # Regular pretraining data (not paired)
            tokens = batch.cuda()
            B = tokens.shape[0]
            X_ctrl = None
            target_gene_idx = None
            log2fc = None
            perturb_sign = None
        
        # Compute loss with ST-style conditioning
        loss = diffusion.compute_loss(
            model, 
            tokens,
            control_set=X_ctrl,
            target_gene_idx=target_gene_idx,
            perturb_magnitude=log2fc,
            perturb_sign=perturb_sign,
            step=global_step
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update learning rate
        lr = cosine_lr_schedule(optimizer, global_step, total_training_steps, config)
        
        epoch_losses.append(loss.item())
        
        # Logging
        if global_step % config.log_every == 0:
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
        
        # Save checkpoint
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
            X_ctrl = batch['X_ctrl_tokens'].cuda()
            target_genes = batch['target_gene']
            target_gene_idx = batch['target_gene_idx'].cuda()
            log2fc = batch['log2fc'].cuda()
            perturb_sign = batch['perturb_sign'].cuda()
            
            B, S, N = X_ctrl.shape
            
            # Generate perturbed cells
            generated = diffusion.p_sample_loop(
                model,
                shape=(B * S, N),  # Generate S cells per perturbation
                control_set=X_ctrl,
                target_gene_idx=target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1),
                perturb_magnitude=log2fc.unsqueeze(1).expand(-1, S).reshape(-1),
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
        dim=128,
        n_head=4,
        n_layer=4,
        ffn_mult=4,
        vocab_size=64,
        n_genes=2000,
        
        # Conditioning
        n_total_genes=18080,  # VCC has 18080 genes
        gene_embed_dim=128,
        perturb_sign_dim=16,
        perturb_magnitude_dim=64,
        magnitude_clip=5.0,
        
        # Control set encoder
        control_set_encoder_layers=2,
        control_set_dim_hidden=128,
        control_set_dim_out=512,
        
        # Diffusion
        n_timesteps=10,
        mask_ratio=0.30,
        schedule="cosine",
        
        # Training
        batch_size=4,  # Smaller batch size for paired sets
        pretrain_batch_size=128,  # Larger batch size for single-cell pretraining
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5000,
        
        # Epochs
        pretrain_epochs=20,
        finetune_epochs=30,
        
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
    tokenizer = create_simple_tokenizer(config.vocab_size)
    
    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    
    # Create pretrain dataloader from HVG-filtered scRNA data
    print("Creating scRNA pretrain dataloader...")
    pretrain_dataset = TokenizedScRNADataset(
        data_dir="data/scRNA/processed",
        tokenizer=tokenizer,
        max_genes=config.n_genes  # Use 2000 HVG genes
    )
    
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Pretrain dataset: {len(pretrain_dataset):,} cells, {pretrain_dataset.dataset.n_genes} genes")
    
    # VCC paired dataloader for fine-tuning
    vcc_dataset, vcc_dataloader = create_vcc_paired_dataloader(
        set_size=16,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        num_workers=4,
        match_by_batch=True,
        use_hvgs=True  # Use HVG genes
    )
    
    # VCC validation dataloader
    val_dataset, val_dataloader = create_vcc_validation_dataloader(
        set_size=16,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        n_samples_per_gene=10,
        use_hvgs=True  # Use HVG genes
    )
    
    # Load HVG info to get gene names
    hvg_info_path = Path("data/vcc_data/hvg_info.json")
    with open(hvg_info_path, 'r') as f:
        hvg_info = json.load(f)
    hvg_genes = hvg_info['hvg_names']
    
    # VCC dataset already uses HVG genes, so mapping is identity
    hvg_to_vcc_mapping = {i: i for i in range(len(hvg_genes))}
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