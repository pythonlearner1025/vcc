#!/usr/bin/env python3
"""Training script for BERT-style Conditional Masked Language Model."""

import torch
import wandb
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from dataset import ScRNADataset, ProcessedScRNADataset, create_dataloader
from vcc_paired_dataloader import (
    create_vcc_paired_dataloader,
    create_vcc_validation_dataloader
)
from vcc_dataloader import VCCDataset
from utils import create_simple_tokenizer, load_hvg_info


@dataclass
class TransformerConfig:
    """Configuration for BERT-style transformer model - matching diffusion config."""
    # Model architecture
    dim: int = 128
    n_head: int = 4
    n_layer: int = 4
    ffn_mult: int = 4
    vocab_size: int = 64  # 64 expression bins
    n_genes: int = 2000  # Number of HVGs
    
    # Conditioning
    n_total_genes: int = 18080  # VCC has 18080 genes
    gene_embed_dim: int = 128
    perturb_sign_dim: int = 16
    perturb_magnitude_dim: int = 64
    magnitude_clip: float = 5.0
    
    # Control set encoder
    control_set_encoder_layers: int = 2
    control_set_dim_hidden: int = 128
    control_set_dim_out: int = 512
    
    # MLM parameters (replacing diffusion)
    mask_ratio: float = 0.30  # Fraction of genes to mask (BERT-style)
    mask_token_id: int = 63  # Use last token as [MASK]
    
    # Training parameters
    batch_size: int = 4  # Smaller batch size due to sets
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    
    # Training epochs
    pretrain_epochs: int = 20
    finetune_epochs: int = 30
    
    # Logging
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 5000
    vcc_eval_interval: int = 5000


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) with shared key/value projections."""
    
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        
        # Query projection for all heads
        self.q_proj = nn.Linear(dim, dim)
        
        # Shared key/value projection (single head)
        self.kv_proj = nn.Linear(dim, 2 * self.head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            kv: Optional key/value tensor (B, M, D). If None, uses x.
        
        Returns:
            Output tensor (B, N, D)
        """
        B, N, D = x.shape
        
        # Use x for key/value if not provided
        if kv is None:
            kv = x
        
        # Project queries for all heads
        q = self.q_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, N, HD)
        
        # Project shared key/value
        kv_proj = self.kv_proj(kv)  # (B, M, 2*HD)
        k, v = kv_proj.chunk(2, dim=-1)  # Each is (B, M, HD)
        
        # Expand k, v for all heads
        k = k.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # (B, H, M, HD)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # (B, H, M, HD)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, M)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, N, HD)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with MQA and cross-attention support."""
    
    def __init__(self, dim: int, n_head: int, ffn_mult: int = 4):
        super().__init__()
        # Self-attention
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = MultiQueryAttention(dim, n_head)
        
        # Cross-attention (optional)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = MultiQueryAttention(dim, n_head)
        
        # FFN
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            context: Optional context tensor for cross-attention (B, M, D)
        """
        # Self-attention
        x = x + self.self_attn(self.ln1(x))
        
        # Cross-attention (if context provided)
        if context is not None:
            x = x + self.cross_attn(self.ln2(x), kv=context)
        
        # FFN
        x = x + self.mlp(self.ln3(x))
        
        return x


class ControlSetEncoder(nn.Module):
    """Encoder for control cell sets following ST architecture."""
    
    def __init__(self, n_genes: int, d_hidden: int = 128, d_out: int = 512, n_layers: int = 2):
        super().__init__()
        
        # Cell-level MLP
        self.cell_mlp = nn.Sequential(
            nn.Linear(n_genes, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # Set-level transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=4,
            dim_feedforward=4 * d_hidden,
            batch_first=True,
            activation='gelu'
        )
        self.set_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.out_proj = nn.Linear(d_hidden, d_out)
        
    def forward(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        """
        Encode control set.
        
        Args:
            x_ctrl: Control cells (B, S, G) where S is set size, G is n_genes
            
        Returns:
            Encoded control set (B, S, d_out)
        """
        # Cell-level encoding
        h = self.cell_mlp(x_ctrl)  # (B, S, d_hidden)
        
        # Set-level attention
        h = self.set_attn(h)  # (B, S, d_hidden)
        
        # Project to output dimension
        h = self.out_proj(h)  # (B, S, d_out)
        
        return h


class ConditionalBERTTransformer(nn.Module):
    """BERT-style transformer with continuous perturbation conditioning."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.n_genes, config.dim) * 0.02)
        
        # Control set encoder
        self.control_encoder = ControlSetEncoder(
            n_genes=config.n_genes,
            d_hidden=config.control_set_dim_hidden,
            d_out=config.control_set_dim_out,
            n_layers=config.control_set_encoder_layers
        )
        
        # Conditioning embeddings
        self.gene_embed = nn.Embedding(config.n_total_genes, config.gene_embed_dim)
        
        # Sign embedding: -1 (knockdown), 0 (control), +1 (activation)
        self.perturb_sign_embed = nn.Embedding(3, config.perturb_sign_dim)
        
        # Magnitude processing: continuous log2 fold change
        self.perturb_magnitude = nn.Sequential(
            nn.Linear(1, config.perturb_magnitude_dim),
            nn.GELU(),
            nn.Linear(config.perturb_magnitude_dim, config.perturb_magnitude_dim),
            nn.GELU(),
            nn.Linear(config.perturb_magnitude_dim, config.dim)
        )
        
        # Combine conditioning
        total_cond_dim = config.gene_embed_dim + config.perturb_sign_dim + config.dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond_dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim)
        )
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlock(config.dim, config.n_head, config.ffn_mult)
            for _ in range(config.n_layer)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self, 
        tokens: torch.LongTensor,  # (B, N)
        control_set: Optional[torch.LongTensor] = None,  # (B, S, N) tokenized control cells
        target_gene_idx: Optional[torch.LongTensor] = None,  # (B,)
        perturb_magnitude: Optional[torch.FloatTensor] = None,  # (B,) log2 fold change
        perturb_sign: Optional[torch.LongTensor] = None,  # (B,) -1, 0, +1
    ) -> torch.Tensor:
        """
        Forward pass with continuous perturbation conditioning.
        
        Args:
            tokens: Input tokens of shape (batch_size, n_genes)
            control_set: Tokenized control cells (batch_size, set_size, n_genes)
            target_gene_idx: Target gene indices
            perturb_magnitude: Log2 fold change values (continuous)
            perturb_sign: Direction of perturbation (-1: down, 0: control, +1: up)
            
        Returns:
            Logits of shape (batch_size, n_genes, vocab_size)
        """
        B, N = tokens.shape
        
        # Embed tokens and add position embeddings
        x = self.token_emb(tokens) + self.pos_emb
        
        # Control set encoding (if provided)
        context = None
        if control_set is not None:
            # Control set should be (B, S, N) where S is set size
            context = self.control_encoder(control_set.float())  # (B, S, control_set_dim_out)
        
        # Conditioning embedding
        if target_gene_idx is not None and perturb_magnitude is not None and perturb_sign is not None:
            # Gene embedding
            gene_emb = self.gene_embed(target_gene_idx)  # (B, gene_embed_dim)
            
            # Clip and normalize magnitude
            magnitude_clipped = torch.clamp(perturb_magnitude, -self.config.magnitude_clip, self.config.magnitude_clip)
            magnitude_normalized = magnitude_clipped / self.config.magnitude_clip  # Normalize to [-1, 1]
            magnitude_emb = self.perturb_magnitude(magnitude_normalized.unsqueeze(-1))  # (B, D)
            
            # Sign embedding (shift to 0, 1, 2 for embedding lookup)
            sign_idx = perturb_sign + 1  # Convert from [-1, 0, 1] to [0, 1, 2]
            sign_emb = self.perturb_sign_embed(sign_idx)  # (B, sign_dim)
            
            # Combine embeddings
            cond_emb = torch.cat([gene_emb, sign_emb, magnitude_emb], dim=-1)
            cond_emb = self.cond_proj(cond_emb)  # (B, D)
            
            # Add conditioning to sequence (broadcast to all positions)
            x = x + cond_emb.unsqueeze(1)
        
        # Apply transformer blocks with optional cross-attention to control set
        for block in self.blocks:
            x = block(x, context=context)
            
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


def compute_mlm_loss(
    model: ConditionalBERTTransformer,
    tokens: torch.LongTensor,
    mask_ratio: float = 0.30,
    control_set: Optional[torch.LongTensor] = None,
    target_gene_idx: Optional[torch.LongTensor] = None,
    perturb_magnitude: Optional[torch.FloatTensor] = None,
    perturb_sign: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    Compute masked language modeling loss.
    
    Args:
        model: The transformer model
        tokens: Original token sequence (B, N)
        mask_ratio: Fraction of tokens to mask
        control_set: Optional control set for conditioning
        target_gene_idx: Target gene indices for perturbation
        perturb_magnitude: Perturbation magnitudes
        perturb_sign: Perturbation signs
        
    Returns:
        MLM loss
    """
    B, N = tokens.shape
    device = tokens.device
    
    # Create mask
    mask = torch.rand(B, N, device=device) < mask_ratio
    
    # Save original tokens for loss computation
    labels = tokens.clone()
    labels[~mask] = -100  # Ignore non-masked positions in loss
    
    # Apply masking to input
    masked_tokens = tokens.clone()
    masked_tokens[mask] = model.config.mask_token_id
    
    # Forward pass
    logits = model(
        masked_tokens,
        control_set=control_set,
        target_gene_idx=target_gene_idx,
        perturb_magnitude=perturb_magnitude,
        perturb_sign=perturb_sign
    )
    
    # Compute loss only on masked positions
    loss = F.cross_entropy(
        logits.view(-1, model.config.vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    
    return loss


def create_optimizer(model: nn.Module, config: TransformerConfig):
    """Create AdamW optimizer with weight decay."""
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)
            else:
                decay.add(fpn)
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))


def cosine_lr_schedule(optimizer, step: int, total_steps: int, config: TransformerConfig):
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        lr = config.learning_rate * step / config.warmup_steps
    else:
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch_transformer(
    model: ConditionalBERTTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TransformerConfig,
    epoch: int,
    global_step: int,
    use_control_sets: bool = True,
    mixed_training: bool = True,
) -> int:
    """
    Train for one epoch using MLM objective.
    
    Args:
        model: The transformer model
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
        else:
            # Regular pretraining data (not paired)
            tokens = batch.cuda()
            B = tokens.shape[0]
            X_ctrl = None
            target_gene_idx = None
            log2fc = None
            perturb_sign = None
        
        # Compute MLM loss
        loss = compute_mlm_loss(
            model, 
            tokens,
            mask_ratio=config.mask_ratio,
            control_set=X_ctrl,
            target_gene_idx=target_gene_idx,
            perturb_magnitude=log2fc,
            perturb_sign=perturb_sign,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update learning rate
        total_training_steps = config.pretrain_epochs * len(dataloader) + config.finetune_epochs * len(dataloader)
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
            checkpoint_path = f'checkpoint_transformer_epoch_{epoch}_step_{global_step}.pt'
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
    model: ConditionalBERTTransformer,
    val_dataloader: torch.utils.data.DataLoader,
    config: TransformerConfig,
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
            
            # For transformer, we need to predict perturbed cells
            # We'll use the control cells as input and predict with perturbation conditioning
            X_ctrl_flat = X_ctrl.view(B * S, N)
            
            # Expand conditioning
            target_gene_idx_exp = target_gene_idx.unsqueeze(1).expand(-1, S).reshape(-1)
            log2fc_exp = log2fc.unsqueeze(1).expand(-1, S).reshape(-1)
            perturb_sign_exp = perturb_sign.unsqueeze(1).expand(-1, S).reshape(-1)
            
            # Forward pass to get predictions
            logits = model(
                X_ctrl_flat,
                control_set=X_ctrl,
                target_gene_idx=target_gene_idx_exp,
                perturb_magnitude=log2fc_exp,
                perturb_sign=perturb_sign_exp
            )
            
            # Get predicted tokens
            predictions = logits.argmax(dim=-1)  # (B*S, N)
            predictions = predictions.view(B, S, N)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_genes.extend(target_genes)
    
    # Compute metrics
    metrics = {
        'zero_shot_genes_evaluated': len(set(all_genes)),
        'epoch': epoch
    }
    
    return metrics


def main():
    # Configuration - matching diffusion config
    config = TransformerConfig(
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
        
        # MLM parameters
        mask_ratio=0.30,
        mask_token_id=63,  # Use last token as [MASK]
        
        # Training
        batch_size=4,
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
        project="vcc-bert-transformer",
        config=config.__dict__,
        name=f"bert_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create model
    model = ConditionalBERTTransformer(config).cuda()
    optimizer = create_optimizer(model, config)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create tokenizer
    tokenizer = create_simple_tokenizer(config.vocab_size)
    
    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    
    # Create pretraining dataloader from scRNA data
    scRNA_data_dir = "data/scRNA/processed"
    if Path(scRNA_data_dir).exists():
        print(f"\nCreating pretraining dataloader from {scRNA_data_dir}")
        pretrain_dataset = ProcessedScRNADataset(
            scRNA_data_dir, 
            n_hvgs=config.n_genes,  # Use same number of genes as model
            tokenizer=tokenizer
        )
        pretrain_dataloader = torch.utils.data.DataLoader(
            pretrain_dataset,
            batch_size=config.batch_size * 8,  # Larger batch size for pretraining
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        print(f"Pretraining dataset: {len(pretrain_dataset):,} cells")
    else:
        print(f"\nWarning: scRNA data not found at {scRNA_data_dir}")
        print("Skipping pretraining phase")
        pretrain_dataloader = None
    
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
    
    # Phase 1: Pretraining on single cells (optional, can skip if starting from checkpoint)
    if pretrain_dataloader is not None and config.pretrain_epochs > 0:
        print("\n=== Phase 1: Pretraining ===")
        for epoch in range(config.pretrain_epochs):
            global_step = train_epoch_transformer(
                model, pretrain_dataloader, optimizer, config,
                epoch, global_step,
                use_control_sets=False,  # No control sets in pretraining
                mixed_training=False
            )
    else:
        print("\n=== Skipping Phase 1: Pretraining (no scRNA data) ===")
    
    # Phase 2: Fine-tuning with control sets
    print("\n=== Phase 2: Fine-tuning with Control Sets ===")
    for epoch in range(config.pretrain_epochs, config.pretrain_epochs + config.finetune_epochs):
        global_step = train_epoch_transformer(
            model, vcc_dataloader, optimizer, config,
            epoch, global_step,
            use_control_sets=True,
            mixed_training=True  # Mix conditioned and unconditioned batches
        )
        
        # Evaluate zero-shot performance
        if epoch % 5 == 0:
            print("\n=== Zero-shot Evaluation ===")
            metrics = evaluate_zero_shot(
                model, val_dataloader, config,
                vcc_gene_to_idx, epoch
            )
            
            print(f"Zero-shot genes evaluated: {metrics['zero_shot_genes_evaluated']}")
            wandb.log(metrics)
    
    # Final save
    final_checkpoint = 'checkpoint_transformer_final.pt'
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
