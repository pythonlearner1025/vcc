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
import argparse
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
from delta_tokenizer import create_delta_tokenizer

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
            
            # Ensure bins are on the same device as input
            if x.device != self.bins.device:
                self.bins = self.bins.to(x.device)
            
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

def _tokenize_batch(data: torch.Tensor, tokenizer) -> torch.Tensor:
    """Tokenize a batch of expression data."""
    if tokenizer is None:
        return data
    
    # Vectorised tokenisation using the tokenizer which can bucketise the whole tensor
    data_cpu = data.cpu()
    tokens = tokenizer(data_cpu)
    return tokens.cuda()

def _process_batch_indices(batch: dict, batch_to_idx: Optional[Dict[str, int]]) -> Optional[torch.Tensor]:
    """Process batch indices from batch names."""
    if batch_to_idx is None or 'pert_batches' not in batch:
        return None
    
    batch_indices = []
    for sample_batches in batch['pert_batches']:
        for batch_name in sample_batches:
            batch_indices.append(batch_to_idx.get(batch_name, 0))
    
    return torch.tensor(batch_indices, device='cuda')

def _expand_control_sets(X_ctrl: torch.Tensor, S: int) -> torch.Tensor:
    """Expand control sets to match flattened perturbed cells."""
    B, _, N = X_ctrl.shape
    X_ctrl_expanded = X_ctrl.unsqueeze(1).expand(-1, S, -1, -1)
    return X_ctrl_expanded.reshape(B * S, S, N)

def train_epoch_st(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ConditionalModelConfig,
    epoch: int,
    global_step: int,
    total_training_steps: int,
    checkpoint_dir: Path,
    tokenizer=None,
    batch_to_idx: Optional[Dict[str, int]] = None,
    use_control_sets: bool = True,
    max_cells: Optional[int] = None,
) -> int:
    """Train for one epoch using ST-style conditioning."""
    model.train()
    epoch_start = time.time()
    epoch_losses = []
    cells_processed = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if max_cells and cells_processed >= max_cells:
            break
        
        if isinstance(batch, dict) and 'tokens' in batch:
            tokens = batch['tokens'].cuda()
            X_ctrl = batch.get('control', None)
            if X_ctrl is not None:
                X_ctrl = X_ctrl.cuda()
            target_gene_idx = batch.get('target_gene_idx', None)
            if target_gene_idx is not None:
                target_gene_idx = target_gene_idx.cuda()
            batch_indices = batch.get('batch_idx', None)
            if batch_indices is not None:
                batch_indices = batch_indices.cuda()
            cells_processed += tokens.shape[0]
        elif isinstance(batch, dict) and 'perturbed_expr' in batch:
            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda() if 'control_expr' in batch else None
            B, S, N = X_pert.shape

            # Build per-cell Δ = perturbed − control (no averaging)
            if X_ctrl is None:
                raise ValueError("control_expr is required for delta computation")
            delta_expr = X_pert - X_ctrl

            delta_tokens = _tokenize_batch(delta_expr, tokenizer)
            tokens = delta_tokens.view(B * S, N)

            # Prepare control-set tokens for contextual conditioning (unchanged)
            X_ctrl_expanded = None
            if use_control_sets and X_ctrl is not None:
                X_ctrl_tokens = _tokenize_batch(X_ctrl, tokenizer)
                X_ctrl_expanded = _expand_control_sets(X_ctrl_tokens, S)

            # Use expanded tokenised control set for the loss (can be None)
            X_ctrl = X_ctrl_expanded
            
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            
            batch_indices = _process_batch_indices(batch, batch_to_idx)
            cells_processed += B * S
        else:
            tokens = batch.cuda()
            X_ctrl = None
            target_gene_idx = None
            batch_indices = None
            cells_processed += tokens.shape[0]
        
        loss = diffusion.compute_loss(
            model,
            tokens,
            control_set=X_ctrl,
            target_gene_idx=target_gene_idx,
            batch_idx=batch_indices,
            step=global_step
        )
        
        optimizer.zero_grad()
        lr = cosine_lr_schedule(optimizer, global_step, total_training_steps, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
        if global_step % 10 == 0:
            avg_loss = np.mean(epoch_losses[-100:] if len(epoch_losses) > 100 else epoch_losses)
            print(f"Epoch {epoch:3d} [{batch_idx+1:4d}/{len(dataloader):4d}] | "
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
            checkpoint_path = checkpoint_dir / f'checkpoint_st_epoch_{epoch}_step_{global_step}.pt'
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
    print(f"Epoch {epoch} completed in {epoch_time:.1f}s | "
          f"Steps: {batch_idx + 1} | Cells: {cells_processed} | "
          f"Avg Loss: {avg_epoch_loss:.4f}")
    
    return global_step

def val_epoch_st(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    val_dataloader: torch.utils.data.DataLoader,
    epoch: int,
    tokenizer=None,
    batch_to_idx: Optional[Dict[str, int]] = None,
    max_cells: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate validation loss on held-out data."""
    model.eval()
    val_losses = []
    cells_evaluated = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_cells and cells_evaluated >= max_cells:
                break
            
            # Fast path when batch is already tokenised by the DataLoader collator
            if isinstance(batch, dict) and 'tokens' in batch:
                tokens = batch['tokens'].cuda()
                X_ctrl = batch.get('control', None)
                if X_ctrl is not None:
                    X_ctrl = X_ctrl.cuda()
                target_gene_idx = batch.get('target_gene_idx', None)
                if target_gene_idx is not None:
                    target_gene_idx = target_gene_idx.cuda()
                batch_indices = batch.get('batch_idx', None)
                if batch_indices is not None:
                    batch_indices = batch_indices.cuda()

                loss = diffusion.compute_loss(
                    model,
                    tokens,
                    control_set=X_ctrl,
                    target_gene_idx=target_gene_idx,
                    batch_idx=batch_indices,
                    step=epoch
                )
                val_losses.append(loss.item())
                cells_evaluated += tokens.shape[0]
                continue

            X_pert = batch['perturbed_expr'].cuda()
            X_ctrl = batch['control_expr'].cuda() if 'control_expr' in batch else None
            B, S, N = X_pert.shape

            # Δ computation for validation (per-cell)
            if X_ctrl is None:
                raise ValueError("control_expr is required for delta computation in validation")
            delta_expr = X_pert - X_ctrl

            delta_tokens = _tokenize_batch(delta_expr, tokenizer)
            X_ctrl_expanded = None
            if X_ctrl is not None:
                X_ctrl_tokens = _tokenize_batch(X_ctrl, tokenizer)
                X_ctrl_expanded = _expand_control_sets(X_ctrl_tokens, S)

            tokens = delta_tokens.view(B * S, N)
            X_ctrl = X_ctrl_expanded
            
            target_gene_idx = batch['target_gene_idx']
            if isinstance(target_gene_idx, list):
                target_gene_idx = torch.tensor(target_gene_idx)
            target_gene_idx = target_gene_idx.cuda()
            target_gene_idx = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            
            batch_indices = _process_batch_indices(batch, batch_to_idx)
            
            loss = diffusion.compute_loss(
                model,
                tokens,
                control_set=X_ctrl,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_indices,
                step=epoch
            )
            
            val_losses.append(loss.item())
            cells_evaluated += B * S
    
    avg_val_loss = np.mean(val_losses) if val_losses else 0.0
    return {
        'val_loss': avg_val_loss,
        'val_batches_evaluated': len(val_losses),
        'epoch': epoch
    }


def main():
    """Main training function for ST-style Conditional Discrete Diffusion Transformer."""
    # -----------------------------
    # Parse command line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Train or continue training ST-style Conditional Diffusion Transformer")
    parser.add_argument("--ckpt_pt", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--continue_from", type=str, choices=["pretrain", "finetune"], default="pretrain",
                        help="Stage to continue from: 'pretrain' or 'finetune'")
    parser.add_argument("--continue_pretrain_epochs", type=int, default=None,
                        help="Override the number of pretrain epochs from the saved config")
    parser.add_argument("--continue_finetune_epochs", type=int, default=None,
                        help="Override the number of finetune epochs from the saved config")
    parser.add_argument("--continue_bs", type=int, default=None,
                        help="Override the batch size for both pretraining and finetuning")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Existing wandb run ID to continue logging to (e.g., from restored run)")
    args = parser.parse_args()

    # -----------------------------
    # Load config
    # -----------------------------
    if args.ckpt_pt:
        ckpt_path = Path(args.ckpt_pt)
        ckpt_dir = ckpt_path.parent
        config_json_path = ckpt_dir / "config.json"
        if not config_json_path.exists():
            raise FileNotFoundError(f"Could not find config.json in {ckpt_dir}")
        with open(config_json_path, "r") as f:
            cfg_dict = json.load(f)
        config = ConditionalModelConfig(**cfg_dict)
    else:
        # Fresh training run – create default config
        config = ConditionalModelConfig(
            train_notes="100M Norm NO ESM2",
            full_eval=True,
            target_is_delta=True,

            dim=512,
            n_head=8,
            n_layer=16,
            ffn_mult=8,

            vocab_size=256,
            n_genes=3000,
            n_total_genes=3000,
            gene_embed_dim=256,

            use_batch_conditioning=True,
            n_batches=48,
            batch_embed_dim=40,
            control_set_encoder_layers=2,
            control_set_dim_hidden=512,

            # BS
            pretrain_batch_size=8,
            vcc_set_size=8,

            # Diffusion
            n_timesteps=16,
            schedule="cosine",
            # MASK
            pretrain_mask_ratio=0.4,
            finetune_mask_ratio_start=0.4,
            finetune_mask_ratio_end=0.9,
            finetune_mask_ratio_steps=10000,
            finetune_full_mask_prob=0.05,
            vcc_batch_size=1,
            
            # LR
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=500,
            finetune_learning_rate=3e-5,
            finetune_warmup_steps=250,

            # DATA
            pretrain_data_dir="/data_normalized/scRNA/processed",
            finetune_data_path="/competition_train.h5",
            hvg_info_path="/workspace/vcc/hvg_seuratv3_3000.txt",
            esm_matrix_path="/esm_all.pt",
            blacklist_path="data/blacklist.txt",
            token_distribution_json="/token_distribution.json",

            token_weighting_annealing_steps=None,
            esm_proj_dim=512,
            pretrain_epochs=1,
            finetune_epochs=10,

            save_every=5000,
            eval_every=1,
            vcc_eval_interval=5000,

            # DEBUG
            debug_pretrain_max_cells=None,
            debug_finetune_max_cells=None,
            debug_eval_max_cells=1000
        )

    # -----------------------------
    # Apply overrides from CLI args
    # -----------------------------
    if args.continue_pretrain_epochs is not None:
        config.pretrain_epochs = args.continue_pretrain_epochs
    if args.continue_finetune_epochs is not None:
        config.finetune_epochs = args.continue_finetune_epochs
    if args.continue_bs is not None:
        config.pretrain_batch_size = args.continue_bs
        config.vcc_set_size = args.continue_bs
    if args.continue_from == "finetune":
        config.pretrain_epochs = 0

    # Create tokenizer – use delta-aware tokenizer when fine-tuning on perturbations
    if getattr(config, 'target_is_delta', False):
        ft_tokenizer, ft_detokenizer = create_delta_tokenizer(config.vocab_size)
    else:
        ft_tokenizer, ft_detokenizer = create_simple_tokenizer(config.vocab_size)

    pt_tokenizer, pt_detokenizer = create_simple_tokenizer(config.vocab_size)

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
        use_cache=True,
        normalize=False
    )
    
    # Wrap with tokenizer
    pretrain_dataset = TokenizedScRNADataset(scrna_dataset, pt_tokenizer)
    
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Pretrain dataset: {len(pretrain_dataset):,} cells, {scrna_dataset.n_hvgs} HVG genes")

    # ------------------------------------------------------------------
    #  Update config & create model now that we know the final HVG count
    # ------------------------------------------------------------------
    config.n_genes = scrna_dataset.n_hvgs
    hvg_gene_ensemble = scrna_dataset.get_gene_names()

    # Initialise wandb AFTER we have the final config
    if args.wandb_run_id:
        # Resume existing run - use the project from the existing run
        print(f"Resuming wandb logging to existing run: {args.wandb_run_id}")
        wandb.init(
            project="vcc-st-diffusion",  # Use same project as restored run
            config=config.__dict__,
            id=args.wandb_run_id,
            resume="must"  # Must resume the specified run
        )
    else:
        # Create new run
        wandb.init(
            project="vcc-st-diffusion",
            config=config.__dict__,
            name=f"st_diffusion_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    # -----------------------------
    # Create model / diffusion / optimiser
    # -----------------------------
    model = ConditionalDiffusionTransformer(config)
    if args.ckpt_pt:
        print(f"Loading weights from checkpoint: {args.ckpt_pt}")
        ckpt_state = torch.load(args.ckpt_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_state["model_state_dict"])
    else:
        ckpt_state = {}
    # Move model to the appropriate device
    model = model.cuda()
    diffusion = PartialMaskingDiffusion(config)
    optimizer = create_optimizer(model, config)
    # If continuing from a checkpoint try to restore optimizer state
    if args.ckpt_pt and "optimizer_state_dict" in ckpt_state:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create VCC train and validation dataloaders
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_train_val_dataloaders(
        tokenizer=ft_tokenizer,
        adata_path=config.finetune_data_path,
        hvg_gene_ids=hvg_gene_ensemble,
        set_size=config.vcc_set_size,
        batch_size=config.vcc_batch_size,
        n_samples_per_gene_train=10,  # Multiple samples per gene for training
        n_samples_per_gene_val=1,      # Single sample per gene for validation
        train_split=0.8,
        num_workers=4,
        random_seed=42,
        normalize=False,
        blacklist_path=config.blacklist_path
    )
    
    # Create gene to index mapping
    vcc_gene_to_idx = {gene: idx for idx, gene in enumerate(hvg_gene_ensemble)}
    
    # Create batch name to index mapping for conditioning
    # Get unique batch names from the VCC dataset
    unique_batches = sorted(list(set(vcc_dataset.adata.obs['batch'].values)))
    batch_to_idx = {batch_name: idx for idx, batch_name in enumerate(unique_batches)}
    print(f"Found {len(unique_batches)} unique batches for batch conditioning")
    # Prepare gene mappings for full evaluation (optional)
    if getattr(config, "full_eval", False):
        from evaluate import _prepare_gene_mappings, _run_validation_merged
        print("Preparing gene mappings for full evaluation …")
        gene_names, hvg_to_full, hvg_gene_ids = _prepare_gene_mappings(
            config.finetune_data_path,
            config.hvg_info_path,
        )
    
    global_step = 0
    
    # Calculate total training steps for learning rate schedule
    pretrain_steps = config.pretrain_epochs * len(pretrain_dataloader) if config.pretrain_epochs > 0 else 0
    finetune_steps = config.finetune_epochs * len(vcc_dataloader)
    print(f"\nTotal training steps: {pretrain_steps+finetune_steps:,} (pretrain: {pretrain_steps:,}, finetune: {finetune_steps:,})")
    
    # Create unique checkpoint directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"checkpoints/run_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    
    # Convert config to dict and ensure all fields are serializable
    config_dict = {k: str(v) if isinstance(v, Path) else v 
                  for k, v in config.__dict__.items()}
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved config to {config_path}")
    
    # Phase 1: Pretraining on single cells
    if config.pretrain_epochs > 0:
        print(f"\n=== Phase 1: Pretraining on scRNA data ({config.pretrain_epochs} epochs) ===")
        print(f"Using {len(pretrain_dataset):,} cells for pretraining")
        
        for epoch in range(config.pretrain_epochs):
            global_step = train_epoch_st(
                model, diffusion, pretrain_dataloader, optimizer, config,
                epoch, global_step, pretrain_steps, checkpoint_dir,
                tokenizer=None,  # Pretrain data is already tokenized
                batch_to_idx=None,  # No batch conditioning for pretraining
                use_control_sets=False,  # No control sets in pretraining
                max_cells=config.debug_pretrain_max_cells
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
    
    # ------------------------------------------------------------------
    # Re-initialise token embeddings if we switch objective to Δ (fine-tune)
    # ------------------------------------------------------------------
    if getattr(config, 'target_is_delta', False):
        print("Re-initialising token embeddings for Δ objective (fine-tune phase)")
        torch.nn.init.normal_(model.token_emb.weight, mean=0.0, std=0.02)
        # Re-initialise output head as well so logits match new embedding
        if hasattr(model, 'head'):
            torch.nn.init.normal_(model.head.weight, mean=0.0, std=0.02)
    # Phase 2: Fine-tuning with control sets
    print("\n=== Phase 2: Fine-tuning with Control Sets ===")
    print(f"Using batch_size={config.pretrain_batch_size} optimized for {config.n_genes}-gene sequences")
    global_step = 0
    config_ft = config
    config_ft.learning_rate = config.finetune_learning_rate
    config_ft.warmup_steps  = config.finetune_warmup_steps
    optimizer = create_optimizer(model, config_ft)
    
    for epoch in range(config.finetune_epochs):
        global_step = train_epoch_st(
            model, diffusion, vcc_dataloader, optimizer, config,
            epoch, global_step, finetune_steps, checkpoint_dir,
            tokenizer=ft_tokenizer,  # VCC data needs tokenization
            batch_to_idx=batch_to_idx,  # Pass batch mapping for conditioning
            use_control_sets=True,
            max_cells=config.debug_finetune_max_cells
        )
        
        # Evaluate on validation set
        if epoch % config.eval_every == 0:
            print("\n=== Validation Set Evaluation ===")
            val_metrics = val_epoch_st(
                model, diffusion, val_dataloader,
                epoch, ft_tokenizer, batch_to_idx,
                max_cells=config.debug_eval_max_cells
            )
            print(f"Validation loss: {val_metrics['val_loss']:.4f} ({val_metrics['val_batches_evaluated']} batches)")
            wandb.log(val_metrics)
            # ------------------------------------------------------------------
            # Full evaluation (cell-eval metrics) – optional
            # ------------------------------------------------------------------
            if getattr(config, "full_eval", False):
                eval_dir = checkpoint_dir / f"eval_ft_epoch{epoch}"
                eval_dir.mkdir(exist_ok=True)
                from types import SimpleNamespace
                args_eval = SimpleNamespace(
                    val_genes_number=10,
                    val_set_size=config.vcc_set_size,
                    blacklist_path=config.blacklist_path,
                )
                device_eval = next(model.parameters()).device
                _run_validation_merged(
                    config,
                    model,
                    diffusion,
                    ft_tokenizer,
                    ft_detokenizer,
                    gene_names,
                    hvg_to_full,
                    hvg_gene_ids,
                    args_eval,
                    device_eval,
                    eval_dir,
                )
    
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
