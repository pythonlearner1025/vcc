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
    # Older PyTorch versions will not expose these flags – skip silently.
    pass
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

from models.diffusion import (
    ConditionalModelConfig, 
    ConditionalDiffusionTransformer,
    PartialMaskingDiffusion,
    create_optimizer,
    cosine_lr_schedule,
)
# Updated imports for cross-dataset HVGs
from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs, create_scrna_hvg_dataloader
from dataset.vcc_paired_dataloader import create_vcc_train_val_dataloaders
from tokenizer import (
    create_delta_tokenizer,
    create_logbin_tokenizer,
    TokenizedScRNADataset
)

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
    if profile_steps > 0:
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
        )
    else:
        profiler_ctx = contextlib.nullcontext()

        device = next(model.parameters()).device  # target device for batches

    # Reset counter
    cells_processed = 0

    # ------------------------------------------------------------------
    # Enter profiler context (no-op if profiling disabled)
    # ------------------------------------------------------------------
    with profiler_ctx as prof:
        for batch_idx, batch in enumerate(dataloader):
            if max_cells and cells_processed >= max_cells:
                break

            # finetune / validation 
            if isinstance(batch, dict) and 'tokens' in batch:
                tokens = batch['tokens'].to(device, non_blocking=True)
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
            # pretrain
            else:
                tokens = batch.to(device, non_blocking=True)
                X_ctrl = None
                target_gene_idx = None
                batch_indices = None
                cells_processed += tokens.shape[0]
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if global_step % 10 == 0:
                print(f"grad_norm: {grad_norm}")
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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            pretrain_data_dir="/scRNA_norm/processed",
            finetune_data_path="/competition_train.h5",
            esm_matrix_path="/esm_all.pt",
            hvg_info_path="assets/hvg_seuratv3_3000.txt",
            blacklist_path="assets/blacklist.txt",
            token_distribution_json="assets/token_distribution.json",

            full_eval=True,
            target_is_delta=True,

            dim=640,
            n_head=16,
            n_layer=18,
            ffn_mult=4,

            # tokenizer
            vocab_size=128,
            # max log1p value
            token_max_value=round(math.log1p(50000), 1),
            
            n_genes=3000,
            n_total_genes=3000,
            gene_embed_dim=512,

            use_batch_conditioning=True,
            n_batches=48,
            batch_embed_dim=40,
            control_set_encoder_layers=2,
            control_set_dim_hidden=512,

            # BS
            pretrain_batch_size=128,
            vcc_set_size=128,

            # Diffusion
            n_timesteps=16,
            schedule="cosine",
            # MASK
            # TODO - use BERT mask ratio 20%
            # Use MUCH smaller finetune_mask_ratio_steps
            # ~1000, so most of the epochs is in 90% masking regime
            pretrain_mask_ratio=0.40,
            finetune_mask_ratio_start=0.4,
            finetune_mask_ratio_end=0.9,
            finetune_mask_ratio_steps=10000,
            finetune_full_mask_prob=0.05,
            vcc_batch_size=1,
            
            # LR
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=500,
            finetune_learning_rate=1e-4,
            finetune_warmup_steps=500,

            token_weighting_annealing_steps=None,
            esm_proj_dim=512,
            
            # 1e6 cells x 5
            pretrain_epochs=1,
            finetune_epochs=1,

            save_every=1500,
            eval_every=1,
            vcc_eval_interval=5000,

            # DEBUG
            debug_pretrain_max_cells=1,
            debug_finetune_max_cells=1,
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
        ft_tokenizer, ft_detokenizer = create_delta_tokenizer(
            config.vocab_size, max_abs=config.token_max_value, min_abs=1e-3)
    else:
        ft_tokenizer, ft_detokenizer = create_logbin_tokenizer(
            config.vocab_size, max_value=config.token_max_value)

    pt_tokenizer, pt_detokenizer = create_logbin_tokenizer(
        config.vocab_size, max_value=config.token_max_value)

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
        num_workers=0,  # avoid redundant IO / RAM
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
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_vcc_train_val_dataloaders(
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
    
    # Initialise global_step based on checkpoint to continue LR schedule smoothly
    global_step = ckpt_state.get('global_step', 0) if ckpt_state else 0
    
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
        print(f"Using {len(pretrain_dataset):,}  for pretraining")
        
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
    if getattr(config, "target_is_delta", False) and args.reinit_weights:
        print("Re-initialising token embeddings and head for Δ objective (fine-tune phase)")
        torch.nn.init.normal_(model.token_emb.weight, mean=0.0, std=0.02)
        # Re-initialise output head (both weight and bias) to prevent early-step bias toward pre-training bins
        if hasattr(model, "head"):
            torch.nn.init.normal_(model.head.weight, mean=0.0, std=0.02)
            if model.head.bias is not None:
                torch.nn.init.zeros_(model.head.bias)
    # Alternative: To completely remove bias, modify the model architecture in models/diffusion.py:
    # self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
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
