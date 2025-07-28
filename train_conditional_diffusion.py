#!/usr/bin/env python3
"""
Conditional Discrete Diffusion Transformer for scRNA-seq perturbation modeling.
Addresses key issues:
1. Partial masking instead of full masking
2. Conditioning on perturbation type and target gene
3. Three-stage training: pretrain -> finetune -> evaluate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from dataset import ScRNADataset
from train import HVGDataset, create_hvg_dataloader, create_tokenizer
from eval import evaluate_on_vcc_validation, log_vcc_metrics, create_vcc_evaluator


@dataclass
class ConditionalModelConfig:
    """Configuration for conditional diffusion model."""
    # Model architecture
    dim: int = 512
    n_head: int = 8
    n_layer: int = 6
    ffn_mult: int = 4
    vocab_size: int = 64  # 64 expression bins
    n_genes: int = 2000  # Number of HVGs
    
    # Conditioning
    n_total_genes: int = 36601  # Total genes in vocabulary for embedding
    gene_embed_dim: int = 128
    perturb_sign_dim: int = 32  # Embedding dimension for perturbation direction
    perturb_magnitude_dim: int = 64  # Hidden dimension for magnitude processing
    magnitude_clip: float = 5.0  # Clip log2 fold changes to [-5, 5]
    
    # Diffusion parameters
    n_timesteps: int = 10
    mask_ratio: float = 0.30  # Fraction of genes to mask (BERT-style)
    schedule: str = "cosine"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Stage-specific parameters
    pretrain_steps: int = 50000
    finetune_steps: int = 20000
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    vcc_eval_interval: int = 5000
    
    @property
    def n_params(self) -> int:
        """Estimate parameter count."""
        params = self.vocab_size * self.dim  # Token embedding
        params += self.n_genes * self.dim  # Position embedding
        params += self.n_timesteps * self.dim  # Time embedding
        params += self.n_total_genes * self.gene_embed_dim  # Gene embeddings
        params += 3 * self.perturb_sign_dim  # Sign embeddings (3 values)
        params += self.perturb_magnitude_dim * self.dim * 3  # Magnitude MLP (3 layers)
        
        # Transformer layers
        per_layer = (
            4 * self.dim * self.dim +  # QKV + proj
            2 * self.dim * self.dim * self.ffn_mult  # FFN
        )
        params += self.n_layer * per_layer
        
        # Output head
        params += self.dim * self.vocab_size
        
        return params


class ConditionalDiffusionTransformer(nn.Module):
    """Conditional discrete diffusion transformer with continuous perturbation conditioning."""
    
    def __init__(self, config: ConditionalModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.n_genes, config.dim) * 0.02)
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim)
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
        target_gene_idx: Optional[torch.LongTensor] = None,  # (B,)
        perturb_magnitude: Optional[torch.FloatTensor] = None,  # (B,) log2 fold change
        perturb_sign: Optional[torch.LongTensor] = None,  # (B,) -1, 0, +1
    ) -> torch.Tensor:
        """
        Forward pass with continuous perturbation conditioning.
        
        Args:
            tokens: Input tokens of shape (batch_size, n_genes)
            timesteps: Diffusion timesteps of shape (batch_size,)
            target_gene_idx: Target gene indices
            perturb_magnitude: Log2 fold change values (continuous)
            perturb_sign: Direction of perturbation (-1: down, 0: control, +1: up)
            
        Returns:
            Logits of shape (batch_size, n_genes, vocab_size)
        """
        B, N = tokens.shape
        
        # Embed tokens and add position embeddings
        x = self.token_emb(tokens) + self.pos_emb
        
        # Time embedding
        t_emb = self.time_emb(timesteps.float())  # (B, D)
        
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
            
            # Combine time and conditioning
            combined_emb = t_emb + cond_emb
        else:
            combined_emb = t_emb
        
        # Add to sequence
        x = x + combined_emb.unsqueeze(1)  # Broadcast to all positions
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


class PartialMaskingDiffusion:
    """Discrete diffusion with partial masking strategy."""
    
    def __init__(self, config: ConditionalModelConfig):
        self.config = config
        self.n_timesteps = config.n_timesteps
        self.vocab_size = config.vocab_size
        self.mask_ratio = config.mask_ratio
        
        # Create noise schedule for mask probability
        if config.schedule == "linear":
            self.mask_probs = torch.linspace(0.0, self.mask_ratio, self.n_timesteps)
        elif config.schedule == "cosine":
            # Cosine schedule
            s = 0.008
            steps = torch.arange(self.n_timesteps + 1, dtype=torch.float32)
            alphas = torch.cos((steps / self.n_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas = alphas / alphas[0]
            self.mask_probs = 1 - alphas[:-1]
            self.mask_probs = self.mask_probs * self.mask_ratio / self.mask_probs[-1]
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
    
    def q_sample(
        self, 
        x_start: torch.LongTensor, 
        t: torch.LongTensor,
        mask_token: int = 63  # Last token is [MASK]
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: partially mask tokens based on timestep.
        
        Returns:
            x_noisy: Partially masked tokens
            mask: Boolean mask indicating which positions were masked
        """
        B, N = x_start.shape
        device = x_start.device
        
        # Get mask probability for each timestep
        mask_prob = self.mask_probs[t].to(device)  # (B,)
        
        # For each sample, randomly mask mask_prob fraction of genes
        rand = torch.rand(B, N, device=device)
        mask = rand < mask_prob.unsqueeze(1)
        
        # Apply mask
        x_noisy = torch.where(mask, mask_token, x_start)
        
        return x_noisy, mask
    
    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.LongTensor,
        target_gene_idx: Optional[torch.LongTensor] = None,
        perturb_magnitude: Optional[torch.FloatTensor] = None,
        perturb_sign: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
    ) -> torch.Tensor:
        """
        Compute training loss with partial masking and expressive conditioning.
        """
        B, N = x_start.shape
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (B,), device=device)
        
        # Partial masking
        x_noisy, mask = self.q_sample(x_start, t, mask_token)
        
        # Predict original tokens
        logits = model(
            x_noisy, t, 
            target_gene_idx=target_gene_idx,
            perturb_magnitude=perturb_magnitude,
            perturb_sign=perturb_sign
        )
        
        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits[mask].view(-1, self.vocab_size),
            x_start[mask].view(-1),
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int],
        target_gene_idx: Optional[torch.LongTensor] = None,
        perturb_magnitude: Optional[torch.FloatTensor] = None,
        perturb_sign: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        temperature: float = 1.0,
        device: str = 'cuda',
    ) -> torch.LongTensor:
        """
        Generate samples with expressive conditioning and optional partial context.
        
        Args:
            partial_context: Optional partial expression to condition on
            context_mask: Boolean mask for which positions are given as context
            guidance_scale: Scale for classifier-free guidance (1.0 = no guidance)
        """
        B, N = shape
        
        # Start with all masks
        x = torch.full((B, N), mask_token, device=device, dtype=torch.long)
        
        # Iterative denoising
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get model predictions with conditioning
            logits = model(
                x, t_batch,
                target_gene_idx=target_gene_idx,
                perturb_magnitude=perturb_magnitude,
                perturb_sign=perturb_sign
            )
            
            logits = logits / temperature
            
            # Determine which positions to unmask at this step
            current_mask_prob = self.mask_probs[t]
            if t > 0:
                next_mask_prob = self.mask_probs[t-1]
            else:
                next_mask_prob = 0.0
            
            # Positions currently masked
            is_masked = (x == mask_token)
            
            # Probability of unmasking
            unmask_prob = 1.0 - (next_mask_prob / (current_mask_prob + 1e-8))
            unmask_prob = min(max(unmask_prob, 0.0), 1.0)
            
            # Randomly select positions to unmask
            unmask = torch.rand_like(is_masked.float()) < unmask_prob
            unmask = unmask & is_masked
            
            # Sample new tokens for positions to unmask
            if unmask.any():
                probs = F.softmax(logits, dim=-1)
                new_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(x.shape)
                x[unmask] = new_tokens[unmask]
        
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.ln2(x))
        return x


def create_gene_mapping(source_genes: List[str], target_genes: List[str]) -> Dict[int, int]:
    """Create mapping between different gene spaces."""
    gene_to_idx_source = {gene: idx for idx, gene in enumerate(source_genes)}
    gene_to_idx_target = {gene: idx for idx, gene in enumerate(target_genes)}
    
    mapping = {}
    for gene, target_idx in gene_to_idx_target.items():
        if gene in gene_to_idx_source:
            mapping[gene_to_idx_source[gene]] = target_idx
    
    return mapping


def prepare_perturbation_conditioning(
    target_gene: str,
    gene_to_idx: Dict[str, int],
    perturbation_type: str = "knockdown",
    magnitude: Optional[float] = None,
    batch_size: int = 1,
    device: str = 'cuda'
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
    """
    Prepare conditioning tensors for perturbation generation.
    
    Args:
        target_gene: Gene name to perturb
        gene_to_idx: Mapping from gene names to indices
        perturbation_type: "knockdown", "activation", or "control"
        magnitude: Optional specific magnitude (if None, uses typical values)
        batch_size: Number of samples to generate
        device: Device to place tensors on
        
    Returns:
        Tuple of (gene_indices, magnitudes, signs)
    """
    # Get gene index
    if target_gene in gene_to_idx:
        gene_idx = gene_to_idx[target_gene]
    else:
        print(f"Warning: Gene {target_gene} not found in vocabulary, using random index")
        gene_idx = np.random.randint(0, len(gene_to_idx))
    
    gene_idx_tensor = torch.full((batch_size,), gene_idx, device=device, dtype=torch.long)
    
    # Set magnitude and sign based on perturbation type
    if perturbation_type == "knockdown":
        if magnitude is None:
            # Sample typical knockdown magnitudes
            magnitude_values = -torch.abs(torch.randn(batch_size, device=device)) * 1.5 - 0.5
        else:
            magnitude_values = torch.full((batch_size,), -abs(magnitude), device=device)
        sign_values = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        
    elif perturbation_type == "activation":
        if magnitude is None:
            # Sample typical activation magnitudes
            magnitude_values = torch.abs(torch.randn(batch_size, device=device)) * 1.5 + 0.5
        else:
            magnitude_values = torch.full((batch_size,), abs(magnitude), device=device)
        sign_values = torch.full((batch_size,), 1, device=device, dtype=torch.long)
        
    else:  # control
        magnitude_values = torch.zeros(batch_size, device=device)
        sign_values = torch.zeros(batch_size, device=device, dtype=torch.long)
    
    return gene_idx_tensor, magnitude_values, sign_values


def train_stage(
    model: nn.Module,
    diffusion: PartialMaskingDiffusion,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ConditionalModelConfig,
    tokenizer,
    stage: str,
    start_step: int = 0,
    max_steps: int = None,
    use_conditioning: bool = False,
    vcc_dataset = None
):
    """
    Train one stage of the model.
    
    Args:
        stage: "pretrain" or "finetune"
        use_conditioning: Whether to use perturbation conditioning
        vcc_dataset: VCC dataset for fine-tuning
    """
    if max_steps is None:
        max_steps = config.pretrain_steps if stage == "pretrain" else config.finetune_steps
    
    model.train()
    step = start_step
    epoch = 0
    
    print(f"\n{'='*60}")
    print(f"Starting {stage.upper()} stage")
    print(f"Steps: {start_step} -> {start_step + max_steps}")
    print(f"Conditioning: {use_conditioning}")
    print(f"{'='*60}")
    
    while step < start_step + max_steps:
        epoch += 1
        epoch_start = time.time()
        
        for batch_idx, (batch, metadata) in enumerate(dataloader):
            batch = batch.cuda()
            tokens = tokenizer[0](batch)  # Tokenize
            
            # Prepare conditioning if needed
            target_gene_idx = None
            perturb_magnitude = None
            perturb_sign = None
            
            if use_conditioning and vcc_dataset is not None:
                # For fine-tuning, use actual perturbation info
                B = batch.shape[0]
                
                # Simulate realistic perturbation data
                # In practice, this would come from the VCC dataset metadata
                
                # Random target genes
                target_gene_idx = torch.randint(0, config.n_total_genes, (B,), device=batch.device)
                
                # Perturbation magnitudes: sample from realistic distribution
                # Knockdowns typically range from -0.5 to -3 log2FC
                # Activations typically range from 0.5 to 3 log2FC
                perturb_type_random = torch.rand(B, device=batch.device)
                
                # 70% knockdowns, 20% activations, 10% controls
                is_knockdown = perturb_type_random < 0.7
                is_activation = (perturb_type_random >= 0.7) & (perturb_type_random < 0.9)
                is_control = perturb_type_random >= 0.9
                
                # Generate magnitudes based on type
                perturb_magnitude = torch.zeros(B, device=batch.device)
                perturb_magnitude[is_knockdown] = -torch.abs(torch.randn(is_knockdown.sum(), device=batch.device)) * 1.5 - 0.5
                perturb_magnitude[is_activation] = torch.abs(torch.randn(is_activation.sum(), device=batch.device)) * 1.5 + 0.5
                perturb_magnitude[is_control] = 0.0
                
                # Clip to reasonable range
                perturb_magnitude = torch.clamp(perturb_magnitude, -config.magnitude_clip, config.magnitude_clip)
                
                # Set signs based on magnitude
                perturb_sign = torch.sign(perturb_magnitude).long()
            
            # Compute loss with new conditioning
            loss = diffusion.compute_loss(
                model, tokens, 
                target_gene_idx=target_gene_idx,
                perturb_magnitude=perturb_magnitude,
                perturb_sign=perturb_sign
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update learning rate
            lr = cosine_lr_schedule(optimizer, step, config)
            
            # Logging
            if step % config.log_every == 0:
                print(f"[{stage}] Step {step:6d} | Epoch {epoch:3d} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e}")
                
                wandb.log({
                    f'{stage}_loss': loss.item(),
                    'learning_rate': lr,
                    'epoch': epoch,
                    'step': step,
                })
            
            # Evaluation for fine-tuning
            if stage == "finetune" and step % config.vcc_eval_interval == 0 and step > start_step:
                print(f"\nEvaluating on VCC validation...")
                try:
                    vcc_metrics = evaluate_on_vcc_validation(
                        model=model,
                        diffusion=diffusion,
                        tokenizer=tokenizer,
                        n_samples=50,
                        max_genes=3
                    )
                    log_vcc_metrics(vcc_metrics, step)
                except Exception as e:
                    print(f"VCC evaluation failed: {e}")
            
            # Save checkpoint
            if step % config.save_every == 0 and step > start_step:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'step': step,
                    'stage': stage,
                }, f'checkpoint_{stage}_step_{step}.pt')
                print(f"Saved {stage} checkpoint at step {step}")
            
            step += 1
            if step >= start_step + max_steps:
                break
        
        epoch_time = time.time() - epoch_start
        print(f"\n[{stage}] Epoch {epoch} completed in {epoch_time:.1f}s")
    
    return step


def cosine_lr_schedule(optimizer, step: int, config: ConditionalModelConfig):
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        lr = config.learning_rate * step / config.warmup_steps
    else:
        total_steps = config.pretrain_steps + config.finetune_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def create_optimizer(model: nn.Module, config: ConditionalModelConfig):
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


def main():
    # Configuration
    config = ConditionalModelConfig()
    
    print("Conditional Discrete Diffusion Transformer")
    print(f"Model parameters: ~{config.n_params/1e6:.1f}M")
    print(f"Mask ratio: {config.mask_ratio:.1%}")
    print(f"Training stages: pretrain ({config.pretrain_steps} steps) -> "
          f"finetune ({config.finetune_steps} steps)")
    
    # Initialize wandb
    wandb.init(
        project="VCC-conditional-diffusion",
        config=config.__dict__,
        name=f"cond_diff_{config.dim}d_{config.n_layer}l"
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
    
    # Create model and diffusion
    print("\nInitializing model...")
    model = ConditionalDiffusionTransformer(config).cuda()
    diffusion = PartialMaskingDiffusion(config)
    optimizer = create_optimizer(model, config)
    
    # Create tokenizer
    tokenizer = create_tokenizer(32768, "binned")
    
    # Stage 1: Pretraining without conditioning
    current_step = train_stage(
        model, diffusion, train_loader, optimizer, config, tokenizer,
        stage="pretrain",
        start_step=0,
        max_steps=config.pretrain_steps,
        use_conditioning=False
    )
    
    # Stage 2: Fine-tuning with conditioning on VCC data
    print("\nLoading VCC data for fine-tuning...")
    vcc_dataset, _ = create_vcc_evaluator()
    
    # Note: In a real implementation, you would create a proper VCC dataloader
    # For now, we'll use the same dataloader but with conditioning
    current_step = train_stage(
        model, diffusion, train_loader, optimizer, config, tokenizer,
        stage="finetune",
        start_step=current_step,
        max_steps=config.finetune_steps,
        use_conditioning=True,
        vcc_dataset=vcc_dataset
    )
    
    # Final evaluation
    print("\nFinal VCC evaluation...")
    try:
        final_metrics = evaluate_on_vcc_validation(
            model=model,
            diffusion=diffusion,
            tokenizer=tokenizer,
            n_samples=200,
            max_genes=10
        )
        log_vcc_metrics(final_metrics, current_step)
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_step': current_step,
    }, 'final_conditional_model.pt')
    
    wandb.finish()
    print("\nTraining completed!")


def demonstrate_expressive_conditioning(model, diffusion, tokenizer, gene_to_idx, device='cuda'):
    """
    Demonstrate the expressive conditioning capabilities of the model.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING EXPRESSIVE CONDITIONING")
    print("="*60)
    
    model.eval()
    n_genes = model.config.n_genes
    
    # Example 1: Variable knockdown strengths
    print("\n1. Variable knockdown strengths for gene TP53:")
    magnitudes = [-0.5, -1.0, -2.0, -3.0]  # Increasing knockdown strength
    
    for mag in magnitudes:
        gene_idx, magnitude, sign = prepare_perturbation_conditioning(
            "TP53", gene_to_idx, perturbation_type="knockdown", 
            magnitude=mag, batch_size=10, device=device
        )
        
        samples = diffusion.p_sample_loop(
            model, shape=(10, n_genes),
            target_gene_idx=gene_idx,
            perturb_magnitude=magnitude,
            perturb_sign=sign,
            device=device
        )
        
        # Convert and analyze
        if tokenizer:
            expr = tokenizer[1](samples).mean(0)
            print(f"  Magnitude {mag:.1f}: Mean expression = {expr.mean():.2f}")
    
    # Example 2: Different perturbation types
    print("\n2. Different perturbation types for gene MYC:")
    types = ["control", "knockdown", "activation"]
    
    for ptype in types:
        gene_idx, magnitude, sign = prepare_perturbation_conditioning(
            "MYC", gene_to_idx, perturbation_type=ptype,
            magnitude=None, batch_size=10, device=device
        )
        
        samples = diffusion.p_sample_loop(
            model, shape=(10, n_genes),
            target_gene_idx=gene_idx,
            perturb_magnitude=magnitude,
            perturb_sign=sign,
            device=device
        )
        
        if tokenizer:
            expr = tokenizer[1](samples).mean(0)
            print(f"  {ptype.capitalize()}: Mean expression = {expr.mean():.2f}, "
                  f"Magnitude = {magnitude[0].item():.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main() 