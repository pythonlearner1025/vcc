#!/usr/bin/env python3
"""
Mini Discrete Diffusion Transformer for scRNA-seq data.
Scaled-down version for sanity checking before 17B model training.

Architecture:
- ~50M parameters (vs 17B full model)
- 512 hidden dim (vs 3072)
- 8 attention heads (vs 24)
- 6 transformer layers (vs 24 dense + 12 MoE)
- No MoE for simplicity in mini version
- 65 vocab bins for gene expression levels
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import time

from dataset import ScRNADataset
from train import HVGDataset, create_hvg_dataloader, create_tokenizer


@dataclass
class ModelConfig:
    """Configuration for mini diffusion model."""
    # Model architecture
    dim: int = 512  # Hidden dimension (3072 in 17B)
    n_head: int = 8  # Attention heads (24 in 17B)
    n_layer: int = 6  # Transformer layers (36 total in 17B)
    ffn_mult: int = 4  # FFN multiplier
    vocab_size: int = 65  # 64 expression bins + [MASK]
    n_genes: int = 2000  # Number of HVGs
    
    # Diffusion parameters
    n_timesteps: int = 1000  # Discrete diffusion steps
    schedule: str = "cosine"  # Noise schedule
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000  # ~25 epochs with 100K cells
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    
    @property
    def n_params(self) -> int:
        """Estimate parameter count."""
        # Embeddings
        params = self.vocab_size * self.dim  # Token embedding
        params += self.n_genes * self.dim  # Position embedding
        params += self.n_timesteps * self.dim  # Time embedding
        
        # Transformer layers
        per_layer = (
            4 * self.dim * self.dim +  # QKV + proj
            2 * self.dim * self.dim * self.ffn_mult  # FFN
        )
        params += self.n_layer * per_layer
        
        # Output head
        params += self.dim * self.vocab_size
        
        return params


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (batch_size,) with timesteps
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
    def __init__(self, dim: int, n_head: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed)[0]
        
        # Pre-norm FFN
        x = x + self.mlp(self.ln2(x))
        return x


class MiniDiffusionTransformer(nn.Module):
    """Mini discrete diffusion transformer for scRNA-seq."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.n_genes, config.dim) * 0.02)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim)
        )
        
        # Transformer blocks
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
        timesteps: torch.LongTensor,  # (B,)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tokens: Input tokens of shape (batch_size, n_genes)
            timesteps: Diffusion timesteps of shape (batch_size,)
            
        Returns:
            Logits of shape (batch_size, n_genes, vocab_size)
        """
        B, N = tokens.shape
        
        # Embed tokens and add position embeddings
        x = self.token_emb(tokens) + self.pos_emb
        
        # Add time embeddings
        t_emb = self.time_emb(timesteps.float())  # (B, D)
        x = x + t_emb.unsqueeze(1)  # Broadcast to all positions
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


class DiscreteDiffusion:
    """Discrete diffusion process for training."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_timesteps = config.n_timesteps
        self.vocab_size = config.vocab_size
        
        # Create noise schedule
        if config.schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.n_timesteps)
        elif config.schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal 2021
            s = 0.008
            steps = torch.arange(self.n_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos((steps / self.n_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
            
        # Precompute schedule quantities
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(
        self, 
        x_start: torch.LongTensor, 
        t: torch.LongTensor,
        mask_token: int = 64  # Last token is [MASK]
    ) -> torch.LongTensor:
        """
        Forward diffusion process: corrupt data by replacing with mask tokens.
        
        Args:
            x_start: Original tokens of shape (B, N)
            t: Timesteps of shape (B,)
            
        Returns:
            Corrupted tokens of shape (B, N)
        """
        B, N = x_start.shape
        device = x_start.device
        
        # Get corruption probability for each timestep
        mask_prob = 1 - self.sqrt_alphas_cumprod[t].to(device)  # (B,)
        
        # Sample random mask
        rand = torch.rand(B, N, device=device)
        mask = rand < mask_prob.unsqueeze(1)
        
        # Apply mask
        x_noisy = torch.where(mask, mask_token, x_start)
        
        return x_noisy
    
    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.LongTensor,
        mask_token: int = 64
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            model: The diffusion model
            x_start: Original tokens of shape (B, N)
            
        Returns:
            Scalar loss
        """
        B, N = x_start.shape
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (B,), device=device)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, mask_token)
        
        # Predict original tokens
        logits = model(x_noisy, t)  # (B, N, vocab_size)
        
        # Compute cross-entropy loss only on masked positions
        mask = (x_noisy == mask_token)
        loss = F.cross_entropy(
            logits[mask].view(-1, self.vocab_size),
            x_start[mask].view(-1),
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.LongTensor,
        t: torch.LongTensor,
        mask_token: int = 64,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        """
        Single denoising step.
        
        Args:
            model: The diffusion model
            x: Current tokens of shape (B, N)
            t: Current timestep
            temperature: Sampling temperature
            
        Returns:
            Denoised tokens of shape (B, N)
        """
        # Get model predictions
        logits = model(x, t)  # (B, N, vocab_size)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample from categorical distribution only for masked positions
        mask = (x == mask_token)
        probs = F.softmax(logits, dim=-1)
        
        # Sample new tokens
        new_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(x.shape)
        
        # Replace only masked positions
        x_denoised = torch.where(mask, new_tokens, x)
        
        return x_denoised
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int],
        mask_token: int = 64,
        temperature: float = 1.0,
        device: str = 'cuda'
    ) -> torch.LongTensor:
        """
        Full denoising loop to generate samples.
        
        Args:
            model: The diffusion model
            shape: (batch_size, n_genes)
            temperature: Sampling temperature
            
        Returns:
            Generated tokens of shape (B, N)
        """
        B, N = shape
        
        # Start from pure noise (all mask tokens)
        x = torch.full((B, N), mask_token, device=device, dtype=torch.long)
        
        # Denoise step by step
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, mask_token, temperature)
            
        return x


def create_optimizer(model: nn.Module, config: ModelConfig):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should and shouldn't have weight decay
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)
            elif pn.endswith('weight'):
                decay.add(fpn)
                
    # Create optimizer groups
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))


def cosine_lr_schedule(optimizer, step: int, config: ModelConfig):
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        # Linear warmup
        lr = config.learning_rate * step / config.warmup_steps
    else:
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr


def evaluate_model(
    model: nn.Module,
    diffusion: DiscreteDiffusion,
    dataloader: torch.utils.data.DataLoader,
    config: ModelConfig,
    n_samples: int = 8
) -> Dict[str, float]:
    """Evaluate model quality."""
    model.eval()
    
    # Compute validation loss
    val_losses = []
    for i, (batch, _) in enumerate(dataloader):
        if i >= 10:  # Evaluate on 10 batches
            break
            
        batch = batch.cuda()
        tokens = create_tokenizer(32768, "binned")[0](batch)
        loss = diffusion.compute_loss(model, tokens)
        val_losses.append(loss.item())
        
    # Generate samples
    samples = diffusion.p_sample_loop(
        model,
        shape=(n_samples, config.n_genes),
        device='cuda'
    )
    
    # Compute sample statistics
    sample_stats = {
        'val_loss': np.mean(val_losses),
        'sample_sparsity': (samples == 0).float().mean().item(),
        'sample_max': samples.max().item(),
        'sample_mean': samples.float().mean().item(),
    }
    
    model.train()
    return sample_stats


def main():
    # Configuration
    config = ModelConfig()
    
    # Print configuration
    print("Mini Discrete Diffusion Transformer Configuration:")
    print(f"  Model dimension: {config.dim}")
    print(f"  Attention heads: {config.n_head}")
    print(f"  Transformer layers: {config.n_layer}")
    print(f"  Estimated parameters: {config.n_params:,} (~{config.n_params/1e6:.1f}M)")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Sequence length: {config.n_genes} genes")
    print(f"  Diffusion steps: {config.n_timesteps}")
    print(f"  Training steps: {config.max_steps:,}")
    
    # Initialize wandb
    wandb.init(
        project="VCC-mini-diffusion",
        config=config.__dict__,
        name=f"mini_diffusion_{config.dim}d_{config.n_layer}l"
    )
    
    # Load data
    print("\nLoading data...")
    data_dir = "data/scRNA/processed"
    train_loader = create_hvg_dataloader(
        data_dir, 
        batch_size=config.batch_size,
        n_hvgs=config.n_genes,
        num_workers=4
    )
    
    # Get dataset info
    dataset = train_loader.dataset
    print(f"Dataset size: {len(dataset):,} cells")
    print(f"Batches per epoch: {len(train_loader):,}")
    
    # Create model and diffusion
    print("\nInitializing model...")
    model = MiniDiffusionTransformer(config).cuda()
    diffusion = DiscreteDiffusion(config)
    
    # Count actual parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Actual model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    
    # Create tokenizer
    tokenize, _, _ = create_tokenizer(32768, "binned")
    
    while step < config.max_steps:
        epoch += 1
        epoch_start = time.time()
        
        for batch_idx, (batch, metadata) in enumerate(train_loader):
            # Move to GPU and tokenize
            batch = batch.cuda()
            tokens = tokenize(batch)
            
            # Compute loss
            loss = diffusion.compute_loss(model, tokens)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update learning rate
            lr = cosine_lr_schedule(optimizer, step, config)
            
            # Logging
            if step % config.log_every == 0:
                print(f"Step {step:6d} | Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e}")
                
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': lr,
                    'epoch': epoch,
                    'step': step,
                })
            
            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                print("\nEvaluating...")
                eval_stats = evaluate_model(model, diffusion, train_loader, config)
                
                print(f"  Validation loss: {eval_stats['val_loss']:.4f}")
                print(f"  Sample sparsity: {eval_stats['sample_sparsity']:.1%}")
                print(f"  Sample mean: {eval_stats['sample_mean']:.2f}")
                
                wandb.log({
                    'val_loss': eval_stats['val_loss'],
                    'sample_sparsity': eval_stats['sample_sparsity'],
                    'sample_max': eval_stats['sample_max'],
                    'sample_mean': eval_stats['sample_mean'],
                    'step': step,
                })
                
                # Save best model
                if eval_stats['val_loss'] < best_val_loss:
                    best_val_loss = eval_stats['val_loss']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'step': step,
                        'val_loss': best_val_loss,
                    }, 'best_mini_diffusion.pt')
                    print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save checkpoint
            if step % config.save_every == 0 and step > 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'step': step,
                }, f'checkpoint_step_{step}.pt')
                print(f"Saved checkpoint at step {step}")
            
            step += 1
            if step >= config.max_steps:
                break
                
        # Epoch timing
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s "
              f"({epoch_time/len(train_loader):.2f}s per batch)")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_stats = evaluate_model(model, diffusion, train_loader, config, n_samples=32)
    print(f"Final validation loss: {final_stats['val_loss']:.4f}")
    print(f"Final sample sparsity: {final_stats['sample_sparsity']:.1%}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'final_stats': final_stats,
    }, 'final_mini_diffusion.pt')
    
    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 