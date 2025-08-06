import torch
import wandb

# ---------------------------------------------------------------------
# Enable memory-efficient attention kernels and higher matmul precision
# ---------------------------------------------------------------------
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

# ------------------------------------------------------------------
# Transformer-Engine FP8 support
# ------------------------------------------------------------------
import os
USE_TE = int(os.getenv("TE"))
HAS_TE = False
print(USE_TE)
if USE_TE:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch import fp8_autocast
        HAS_TE = True
    except ImportError:
        from contextlib import nullcontext as fp8_autocast
        HAS_TE = False

import time
import math
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import json
import argparse
from torch.profiler import profile, ProfilerActivity
import contextlib
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

from models.diffusion import (
    ConditionalModelConfig, 
    ConditionalDiffusionTransformer,
    PartialMaskingDiffusion,
    create_muon_optimizer,
    get_lr,
)
# Updated imports for cross-dataset HVGs
from dataset.orion_paired import create_orion_train_val_dataloaders
from dataset.vcc_paired_dataloader import create_vcc_train_val_dataloaders
from tokenizer import (
    create_delta_tokenizer,
    create_logbin_tokenizer
)
from dataset.collators import OrionCollator

# ------------------------------------------------------------------
# Helper: choose the correct autocast context (FP8 vs BF16)
# ------------------------------------------------------------------
from contextlib import nullcontext as _nullctx

def _autocast(model):
    """Return FP8 or BF16 autocast context depending on model settings."""
    if HAS_TE and hasattr(model, 'config') and model.config.use_fp8:
        # FP8 kernels – use Transformer-Engine's context manager
        return fp8_autocast(enabled=True)
    # Fallback to BF16 mixed precision
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)

def train_epoch_st(
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    dataloader: torch.utils.data.DataLoader,
    opts: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
    config: ConditionalModelConfig,
    epoch: int,
    global_step: int,
    total_training_steps: int,
    checkpoint_dir: Path,
    tokenizer=None,
    batch_to_idx: Optional[Dict[str, int]] = None,
    use_control_sets: bool = True,
    max_cells: Optional[int] = None,
    profile_steps: int = 0,  # Number of training batches to profile (0 = no profiling)
) -> int:
    """Train for one epoch using ST-style conditioning."""
    model.train()
    epoch_start = time.time()
    epoch_losses = []

    # ------------------------------------------------------------------
    # Optional PyTorch profiler setup – profile the first `profile_steps`
    # batches (after a 1-step warm-up) for CUDA time & memory.
    # ------------------------------------------------------------------
    profiler_ctx: contextlib.AbstractContextManager
    device = next(model.parameters()).device  # target device for batches
    
    if profile_steps > 0:
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
        )
    else:
        profiler_ctx = contextlib.nullcontext()

    # Reset counter
    cells_processed = 0

    # ------------------------------------------------------------------
    # Enter profiler context (no-op if profiling disabled)
    # ------------------------------------------------------------------
    with profiler_ctx as prof:
        s = time.time()
        for batch_idx, batch in enumerate(dataloader):
            if max_cells and cells_processed >= max_cells:
                break
            e = time.time()
            print(f'dataload time: {(e-s)*1000:.0f}ms')

            tokens = batch['tokens'].to(device, non_blocking=True)
            X_ctrl = batch.get('control', None)
            if X_ctrl is not None:
                X_ctrl = X_ctrl.to(device, non_blocking=True)
            target_gene_idx = batch.get('target_gene_idx', None)
            if target_gene_idx is not None:
                target_gene_idx = target_gene_idx.to(device, non_blocking=True)
            batch_indices = batch.get('batch_idx', None)
            if batch_indices is not None:
                batch_indices = batch_indices.to(device, non_blocking=True)
            cells_processed += tokens.shape[0]
            e1 = time.time()
            print(f'move to device time: {(e1-e)*1000:.0f}ms')

            #print(f"X_ctrl shape: {X_ctrl.shape}")

            with _autocast(model):
                loss = diffusion.compute_loss(
                    model,
                    tokens,
                    control_set=X_ctrl,
                    target_gene_idx=target_gene_idx,
                    batch_idx=batch_indices,
                    step=global_step
                )
            e2 = time.time()
            print(f'step time: {(e2-e1)*1000:.0f}ms')

            optimizer, scheduler = opts 
            optimizer.zero_grad()
            loss.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) no grad when using Muon
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            e3 = time.time()
            print(f'backward time: {(e3-e2)*1000:.0f}ms')
            
            if global_step % 1 == 0:
                # Get current learning rate from scheduler
                lr = scheduler.get_last_lr()[0]
                
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
            s = time.time()
            
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
    device = next(model.parameters()).device  # target device for batches
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_cells and cells_evaluated >= max_cells:
                break
            
            # Fast path when batch is already tokenised by the DataLoader collator
            # Always use collator 
            tokens = batch['tokens'].to(device, non_blocking=True)
            X_ctrl = batch.get('control', None)
            if X_ctrl is not None:
                X_ctrl = X_ctrl.to(device, non_blocking=True)
            target_gene_idx = batch.get('target_gene_idx', None)
            if target_gene_idx is not None:
                target_gene_idx = target_gene_idx.to(device, non_blocking=True)
            batch_indices = batch.get('batch_idx', None)
            if batch_indices is not None:
                batch_indices = batch_indices.to(device, non_blocking=True)

            with _autocast(model):
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
    parser.add_argument("--reinit_weights", type=int, default=0) 
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
            # DATA
            pretrain_data_dir="data/batched2",
            finetune_data_path="/competition_train.h5",
            esm_matrix_path="/esm_all.pt",
            hvg_info_path="assets/hvg_seuratv3_6288.txt",
            blacklist_path="assets/blacklist.txt",
            token_distribution_json="assets/token_distribution.json",

            full_eval=True,
            target_is_delta=True,

            dim=648,
            n_head=12,
            n_layer=12,
            ffn_mult=4,

            # tokenizer
            vocab_size=128,
            # max log1p value
            # TODO - while Orion was normalized to 10,000, VCC competition_train.h5 
            # was normalized to 50,000
            token_max_value=round(math.log1p(50000), 1),
            
            n_genes=6288,
            n_total_genes=6288,
            gene_embed_dim=648,
            # Ensure gene_embed_dim + batch_embed_dim is divisible by 16 for FP8
            # 512 + 64 = 576 which is divisible by 16
            batch_embed_dim=64,

            use_batch_conditioning=True,
            control_set_encoder_layers=2,
            control_set_dim_hidden=648,

            # BS
            pretrain_batch_size=128,
            vcc_set_size=128,

            # Diffusion
            n_timesteps=16,
            schedule="cosine",
            # MASK
            pretrain_mask_ratio=0.20,
            finetune_mask_ratio_start=0.2,
            finetune_mask_ratio_end=0.9,
            finetune_mask_ratio_steps=10000,
            finetune_full_mask_prob=0.05,
            vcc_batch_size=1,
            
            # LR
            adam_lr=2e-4,
            muon_lr=2e-2,
            warmup_steps=1000,
            finetune_learning_rate=5e-5,
            finetune_warmup_steps=500,

            token_weighting_annealing_steps=None,
            esm_proj_dim=648,
            
            # 1e6 cells x 5
            pretrain_epochs=1,
            finetune_epochs=3,

            save_every=1000,
            eval_every=1,
            max_eval_genes=25,

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

    # FP8 validation
    if config.use_fp8 and HAS_TE:
        cond_dim = config.gene_embed_dim + (config.batch_embed_dim if config.use_batch_conditioning else 0)
        assert cond_dim % 16 == 0, f"FP8 requires cond_dim ({cond_dim}) divisible by 16. Adjust batch_embed_dim."
        assert config.dim % 16 == 0, f"FP8 requires model dim ({config.dim}) divisible by 16"

    tokenizer, detokenizer = create_delta_tokenizer(
        config.vocab_size, max_abs=config.token_max_value, min_abs=1e-3)

    with open(config.hvg_info_path, 'r') as f:
        hvg_gene_ensemble = [line.strip() for line in f.readlines()]

    print("\n=== Creating Dataloaders ===")

    # ------------------------------------------------------------------
    #  Update config & create model now that we know the final HVG count
    # ------------------------------------------------------------------
    config.n_genes 
    hvg_gene_ensemble

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

    # Create Orion pretrain and validation dataloaders
    (orion_train_ds, orion_train_dl), (orion_val_ds, orion_val_dl) = create_orion_train_val_dataloaders(
        batches_dir=config.pretrain_data_dir,
        hvg_gene_ids=hvg_gene_ensemble,
        tokenizer=tokenizer,
        set_size=config.vcc_set_size,
        train_split=0.9,
        num_workers=4,
        random_seed=42
    )
    # Backward-compatibility aliases for existing variable names
    orion_dataloader = orion_train_dl
    orion_dataset = orion_train_ds
    
     # Create VCC train and validation dataloaders
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_vcc_train_val_dataloaders(
        adata_path=config.finetune_data_path,
        hvg_gene_ids=hvg_gene_ensemble,
        tokenizer=tokenizer,
        set_size=config.vcc_set_size,
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
    # -- Complicated global batch index creating logic below --- 
    # Merge unique batch names from both VCC and Orion datasets
    vcc_batches = sorted(list(set(vcc_dataset.adata.obs['batch'].values)))

    # Extract unique batch / sample names from Orion dataset
    def _extract_unique_batches(ds):
        base_ds = ds.dataset if hasattr(ds, 'dataset') else ds
        return getattr(base_ds, 'unique_batches', [])

    orion_batches = _extract_unique_batches(orion_dataloader.dataset)

    unique_batches = sorted(set(vcc_batches) | set(orion_batches))
    batch_to_idx = {batch_name: idx for idx, batch_name in enumerate(unique_batches)}

    # Update collate functions to use global batch mapping
    if orion_dataloader is not None and hasattr(orion_dataloader, 'collate_fn'):
        orion_dataloader.collate_fn = OrionCollator(tokenizer, config.vcc_set_size, batch_to_idx)
    if vcc_dataloader is not None and hasattr(vcc_dataloader, 'collate_fn'):
        if hasattr(vcc_dataloader.collate_fn, 'batch_to_idx'):
            vcc_dataloader.collate_fn.batch_to_idx = batch_to_idx
    # Also update the dataset attribute so other components (e.g., evaluation) can access it
    if vcc_dataset is not None:
        vcc_dataset.batch_to_idx = batch_to_idx
    if val_dataset is not None:
        val_dataset.batch_to_idx = batch_to_idx
    if val_dataloader is not None and hasattr(val_dataloader, 'collate_fn'):
        if hasattr(val_dataloader.collate_fn, 'batch_to_idx'):
            val_dataloader.collate_fn.batch_to_idx = batch_to_idx
    print(f"Found {len(unique_batches)} unique batches for batch conditioning")
    # -- end creating global batches --

    # -----------------------------
    # Create model / diffusion / optimiser
    # -----------------------------
    n_batches = len(unique_batches)
    while n_batches % 16 != 0:
        n_batches += 1 
    print(f'New batch size: {n_batches}')
    config.n_technical_batches = n_batches
    model = ConditionalDiffusionTransformer(config)
    if args.ckpt_pt:
        print(f"Loading weights from checkpoint: {args.ckpt_pt}")
        ckpt_state = torch.load(args.ckpt_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_state["model_state_dict"])
    else:
        ckpt_state = {}
    # Move model to the appropriate device
    model = model.cuda()
    
    # Torch compilation settings - can be disabled via environment variable if issues occur
    compile_mode = os.getenv("TORCH_COMPILE_MODE", "disable")
    if compile_mode != "disable":
        # Use reduce-overhead mode by default to avoid Triton compilation issues
        # max-autotune can generate invalid Triton kernels for certain operations
        print(f"Compiling model with torch.compile (mode={compile_mode})")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    else:
        print("Torch compilation disabled (TORCH_COMPILE_MODE=disable)")
    diffusion = PartialMaskingDiffusion(config)
    optimizer = create_muon_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr) 
    # If continuing from a checkpoint try to restore optimizer state
    if args.ckpt_pt and "optimizer_state_dict" in ckpt_state:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Prepare gene mappings for full evaluation (optional)
    if getattr(config, "full_eval", False):
        from evaluate import _prepare_gene_mappings, _run_validation_merged
        print("Preparing gene mappings for full evaluation …")
        gene_names, hvg_to_full, hvg_gene_ids = _prepare_gene_mappings(
            config.finetune_data_path,
            config.hvg_info_path,
        )
    # Initialise global_step based on checkpoint to continue LR schedule smoothly
    global_step = ckpt_state.get('global_step', 0) if ckpt_state else 0
    
    # Calculate total training steps for learning rate schedule
    pretrain_steps = config.pretrain_epochs * len(orion_dataloader) if config.pretrain_epochs > 0 else 0
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
        print(f"\n=== Phase 1: Pretraining on Orion data ({config.pretrain_epochs} epochs) ===")
        print(f"Using {len(orion_train_ds):,} sets for Orion pretraining")
        
        for epoch in range(config.pretrain_epochs):
            global_step = train_epoch_st(
                model, diffusion, orion_dataloader, (optimizer, scheduler), config,
                epoch, global_step, pretrain_steps, checkpoint_dir,
                tokenizer=tokenizer,  # TODO 
                batch_to_idx=batch_to_idx,  # Use global batch mapping for pretraining
                use_control_sets=False,  # No control sets in pretraining
                max_cells=config.debug_pretrain_max_cells
            )

            # Evaluate on training set
            if epoch % config.eval_every == 0:
                print("\n=== Pretraining Validation ===")
                val_metrics = val_epoch_st(
                    model, diffusion, val_dataloader,
                    epoch, tokenizer, batch_to_idx,
                    max_cells=config.debug_eval_max_cells
                )
                print(f"Pretraining validation loss: {val_metrics['val_loss']:.4f} ({val_metrics['val_batches_evaluated']} batches)")
                wandb.log(val_metrics)
                if config.full_eval:
                    eval_dir = checkpoint_dir / f"eval_pt_epoch{epoch}"
                    eval_dir.mkdir(exist_ok=True)
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
                        tokenizer,
                        detokenizer,
                        gene_names,
                        hvg_to_full,
                        hvg_gene_ids,
                        args_eval,
                        device_eval,
                        eval_dir,
                        batch_to_idx=batch_to_idx,
                        max_genes=config.max_eval_genes,
                        val_ds=orion_val_ds,
                        val_dl=orion_val_dl,
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
    
    print("\n=== Phase 2: Fine-tuning with Control Sets ===")
    global_step = 0
    config_ft = config
    config_ft.learning_rate = config.finetune_learning_rate
    config_ft.warmup_steps  = config.finetune_warmup_steps
    optimizer = create_muon_optimizer(model, config_ft)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr) 
    
    for epoch in range(config.finetune_epochs):
        global_step = train_epoch_st(
            model, diffusion, vcc_dataloader, (optimizer, scheduler), config,
            epoch, global_step, finetune_steps, checkpoint_dir,
            tokenizer=tokenizer,  # VCC data needs tokenization
            batch_to_idx=batch_to_idx,  # Pass batch mapping for conditioning
            use_control_sets=True,
            max_cells=config.debug_finetune_max_cells
        )
        
        # Evaluate on validation set
        if epoch % config.eval_every == 0:
            print("\n=== Finetuning Validation ===")
            val_metrics = val_epoch_st(
                model, diffusion, val_dataloader,
                epoch, tokenizer, batch_to_idx,
                max_cells=config.debug_eval_max_cells
            )
            print(f"Finetuning validation loss: {val_metrics['val_loss']:.4f} ({val_metrics['val_batches_evaluated']} batches)")
            wandb.log(val_metrics)
            # ------------------------------------------------------------------
            # Full evaluation (cell-eval metrics) – optional
            # ------------------------------------------------------------------
            if getattr(config, "full_eval", False):
                eval_dir = checkpoint_dir / f"eval_ft_epoch{epoch}"
                eval_dir.mkdir(exist_ok=True)
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
                    tokenizer,
                    detokenizer,
                    gene_names,
                    hvg_to_full,
                    hvg_gene_ids,
                    args_eval,
                    device_eval,
                    eval_dir,
                    batch_to_idx=batch_to_idx,
                    max_genes=config.max_eval_genes,
                    val_ds=val_dataset,
                    val_dl=val_dataloader
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
