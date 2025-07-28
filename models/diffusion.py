#!/usr/bin/env python3
"""
Conditional Discrete Diffusion Transformer Model Components.

This module contains all model-related logic for the conditional diffusion transformer,
including the model architecture, diffusion process, and associated utilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


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
    perturb_sign_dim: int = 3  # Embedding dimension for perturbation direction
    perturb_magnitude_dim: int = 64  # Hidden dimension for magnitude processing
    magnitude_clip: float = 5.0  # Clip log2 fold changes to [-5, 5]
    
    # Control set conditioning
    control_set_encoder_layers: int = 2
    control_set_dim_hidden: int = 128
    control_set_dim_out: int = 512
    
    # Diffusion parameters
    n_timesteps: int = 10
    mask_ratio: float = 0.30  # Fraction of genes to mask (BERT-style)
    schedule: str = "cosine"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Training epochs (replacing step-based training)
    pretrain_epochs: int = 50
    finetune_epochs: int = 20
    
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
        
        # Control set encoder
        self.control_encoder = ControlSetEncoder(
            n_genes=config.n_genes,  # Control sets have n_genes dimensions (after tokenization)
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
        timesteps: torch.LongTensor,  # (B,)
        control_set: Optional[torch.LongTensor] = None,  # (B, S, N) tokenized control cells
        target_gene_idx: Optional[torch.LongTensor] = None,  # (B,)
        perturb_magnitude: Optional[torch.FloatTensor] = None,  # (B,) log2 fold change
        perturb_sign: Optional[torch.LongTensor] = None,  # (B,) -1, 0, +1
    ) -> torch.Tensor:
        """
        Forward pass with continuous perturbation conditioning.
        
        Args:
            tokens: Input tokens of shape (batch_size, n_genes)
            timesteps: Diffusion timesteps of shape (batch_size,)
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
        
        # Time embedding
        t_emb = self.time_emb(timesteps.float())  # (B, D)
        
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
            
            # Combine time and conditioning
            combined_emb = t_emb + cond_emb
        else:
            combined_emb = t_emb
        
        # Add to sequence (following ST: add perturbation embedding to input)
        x = x + combined_emb.unsqueeze(1)  # Broadcast to all positions
        
        # Apply transformer blocks with optional cross-attention to control set
        for block in self.blocks:
            x = block(x, context=context)
            
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
    
    def sample_mask_ratio(self, is_conditioned: bool, step: int) -> float:
        """
        Sample masking ratio based on conditioning and training step.
        
        Args:
            is_conditioned: Whether this batch has conditioning
            step: Current training step
            
        Returns:
            Mask ratio to use
        """
        if is_conditioned:
            # Curriculum learning: gradually increase masking difficulty
            # Start with 30% masking, ramp to 90% (not 100% to avoid trivial cases)
            max_mask = min(0.9, 0.3 + 0.6 * step / 100_000)
        else:
            # Pretraining: moderate masking for representation learning
            max_mask = 0.5
        
        # Sample from a beta distribution for more principled randomness
        # Beta(2, 5) gives a right-skewed distribution favoring lower values
        u = torch.distributions.Beta(2, 5).sample().item()
        return max_mask * u
    
    def q_sample(
        self, 
        x_start: torch.LongTensor, 
        t: torch.LongTensor,
        mask_token: int = 63,  # Last token is [MASK]
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: partially mask tokens based on timestep.
        
        Returns:
            x_noisy: Partially masked tokens
            mask: Boolean mask indicating which positions were masked
        """
        B, N = x_start.shape
        device = x_start.device
        
        if mask_ratio is not None:
            # Use provided mask ratio
            mask_prob = torch.full((B,), mask_ratio, device=device)
        else:
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
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        perturb_magnitude: Optional[torch.FloatTensor] = None,
        perturb_sign: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute training loss with partial masking and expressive conditioning.
        """
        B, N = x_start.shape
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (B,), device=device)
        
        # Determine mask ratio based on conditioning
        is_conditioned = control_set is not None
        mask_ratio = self.sample_mask_ratio(is_conditioned, step)
        
        # Partial masking
        x_noisy, mask = self.q_sample(x_start, t, mask_token, mask_ratio=mask_ratio)
        
        # Predict original tokens
        logits = model(
            x_noisy, t,
            control_set=control_set,
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
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        perturb_magnitude: Optional[torch.FloatTensor] = None,
        perturb_sign: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        temperature: float = 1.0,
        device: str = 'cuda',
    ) -> torch.LongTensor:
        """
        Generate samples with expressive conditioning and optional partial context.
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
                control_set=control_set,
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
    magnitude: float,
    batch_size: int = 1,
    device: str = 'cuda'
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
    """
    Prepare conditioning tensors for perturbation generation.
    
    The perturbation type is inferred from the magnitude and sign of the expression change.
    This is more general than using explicit perturbation types.
    
    Args:
        target_gene: Gene name to perturb
        gene_to_idx: Mapping from gene names to indices
        magnitude: Log2 fold change value (negative for knockdown, positive for activation)
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
        gene_idx = torch.randint(0, len(gene_to_idx), (1,)).item()
    
    gene_idx_tensor = torch.full((batch_size,), gene_idx, device=device, dtype=torch.long)
    
    # Infer perturbation type from magnitude
    magnitude_tensor = torch.full((batch_size,), magnitude, device=device)
    
    # Sign is simply the sign of the magnitude
    # -1 for knockdown (negative magnitude)
    # 0 for control (zero magnitude)
    # +1 for activation (positive magnitude)
    sign_tensor = torch.sign(magnitude_tensor).long()
    
    return gene_idx_tensor, magnitude_tensor, sign_tensor


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


def cosine_lr_schedule(optimizer, step: int, total_steps: int, config: ConditionalModelConfig):
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        lr = config.learning_rate * step / config.warmup_steps
    else:
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AuxiliaryLosses:
    """Additional loss functions for diffusion training."""
    
    @staticmethod
    def mmd_loss(x_real: torch.Tensor, x_fake: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Maximum Mean Discrepancy loss for distribution matching.
        
        Args:
            x_real: Real samples (B1, D)
            x_fake: Generated samples (B2, D)
            sigma: Bandwidth parameter for RBF kernel
            
        Returns:
            MMD loss value
        """
        def rbf_kernel(x, y, sigma):
            """RBF kernel for MMD."""
            pairwise_dists = torch.cdist(x, y) ** 2
            return torch.exp(-pairwise_dists / (2 * sigma ** 2))
        
        # Kernel matrices
        k_xx = rbf_kernel(x_real, x_real, sigma)
        k_yy = rbf_kernel(x_fake, x_fake, sigma)
        k_xy = rbf_kernel(x_real, x_fake, sigma)
        
        # MMD^2 estimator
        m, n = x_real.size(0), x_fake.size(0)
        mmd = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
        mmd += (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
        mmd -= 2 * k_xy.sum() / (m * n)
        
        return mmd
    
    @staticmethod
    def wasserstein_loss(x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        """
        Approximate Wasserstein distance using sorted samples.
        
        Args:
            x_real: Real samples (B, D)
            x_fake: Generated samples (B, D)
            
        Returns:
            Wasserstein distance approximation
        """
        # Flatten and sort samples
        real_sorted, _ = torch.sort(x_real.flatten())
        fake_sorted, _ = torch.sort(x_fake.flatten())
        
        # Ensure same length for comparison
        min_len = min(len(real_sorted), len(fake_sorted))
        real_sorted = real_sorted[:min_len]
        fake_sorted = fake_sorted[:min_len]
        
        return torch.mean(torch.abs(real_sorted - fake_sorted))
    
    @staticmethod
    def sparsity_regularization(x: torch.Tensor, target_sparsity: float = 0.9) -> torch.Tensor:
        """
        Regularization term to encourage sparsity in generated samples.
        
        Args:
            x: Generated samples (B, N)
            target_sparsity: Target fraction of zeros
            
        Returns:
            Sparsity loss
        """
        actual_sparsity = (x == 0).float().mean()
        return torch.abs(actual_sparsity - target_sparsity)


class ImprovedPartialMaskingDiffusion(PartialMaskingDiffusion):
    """Enhanced diffusion with auxiliary losses and better sampling."""
    
    def __init__(self, config: ConditionalModelConfig, aux_loss_weight: float = 0.1):
        super().__init__(config)
        self.aux_loss_weight = aux_loss_weight
        self.aux_losses = AuxiliaryLosses()
    
    def compute_loss_with_aux(
        self,
        model: nn.Module,
        x_start: torch.LongTensor,
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        perturb_magnitude: Optional[torch.FloatTensor] = None,
        perturb_sign: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        step: int = 0,
        use_aux_losses: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with optional auxiliary losses.
        
        Returns:
            Dictionary with 'main_loss', 'aux_loss', and 'total_loss'
        """
        # Standard diffusion loss
        main_loss = self.compute_loss(
            model, x_start, control_set, target_gene_idx, 
            perturb_magnitude, perturb_sign, mask_token, step
        )
        
        losses = {'main_loss': main_loss}
        
        if use_aux_losses and step > 1000:  # Start aux losses after warmup
            # Generate samples for auxiliary losses
            with torch.no_grad():
                B, N = x_start.shape
                generated = self.p_sample_loop(
                    model, (B, N),
                    control_set=control_set,
                    target_gene_idx=target_gene_idx,
                    perturb_magnitude=perturb_magnitude,
                    perturb_sign=perturb_sign,
                    device=x_start.device
                )
            
            # Convert to continuous values for comparison
            x_real_continuous = x_start.float()
            x_fake_continuous = generated.float()
            
            # Compute auxiliary losses
            aux_loss = 0.0
            
            # Distribution matching
            if B > 1:  # Need multiple samples for MMD
                mmd_loss = self.aux_losses.mmd_loss(x_real_continuous, x_fake_continuous)
                aux_loss += mmd_loss
                losses['mmd_loss'] = mmd_loss
            
            # Sparsity regularization (important for scRNA-seq)
            sparsity_loss = self.aux_losses.sparsity_regularization(x_fake_continuous)
            aux_loss += sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
            
            aux_loss *= self.aux_loss_weight
            losses['aux_loss'] = aux_loss
        else:
            losses['aux_loss'] = torch.tensor(0.0, device=main_loss.device)
        
        losses['total_loss'] = losses['main_loss'] + losses['aux_loss']
        return losses
class DiffusionEvaluator:
    """Comprehensive evaluation utilities for diffusion models."""
    
    def __init__(self, tokenizer, detokenizer):
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
    
    def evaluate_generation_quality(
        self, 
        generated_tokens: torch.Tensor, 
        real_tokens: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate quality of generated samples.
        
        Args:
            generated_tokens: Generated token sequences (B, N)
            real_tokens: Real token sequences (B, N)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to continuous values
        generated_expr = self.detokenizer(generated_tokens)
        real_expr = self.detokenizer(real_tokens)
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_expression_real'] = real_expr.mean().item()
        metrics['mean_expression_gen'] = generated_expr.mean().item()
        metrics['std_expression_real'] = real_expr.std().item()
        metrics['std_expression_gen'] = generated_expr.std().item()
        
        # Sparsity (important for scRNA-seq)
        metrics['sparsity_real'] = (real_expr == 0).float().mean().item()
        metrics['sparsity_gen'] = (generated_expr == 0).float().mean().item()
        
        # Distribution comparison
        if generated_tokens.numel() > 0 and real_tokens.numel() > 0:
            # KL divergence between token distributions
            real_dist = torch.bincount(real_tokens.flatten(), minlength=64).float()
            gen_dist = torch.bincount(generated_tokens.flatten(), minlength=64).float()
            
            real_dist = real_dist / real_dist.sum()
            gen_dist = gen_dist / gen_dist.sum()
            
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            kl_div = F.kl_div(
                torch.log(gen_dist + eps), 
                real_dist + eps, 
                reduction='sum'
            ).item()
            metrics['kl_divergence'] = kl_div
            
            # Jensen-Shannon divergence (symmetric)
            m = (real_dist + gen_dist) / 2
            js_div = 0.5 * F.kl_div(torch.log(real_dist + eps), m + eps, reduction='sum').item()
            js_div += 0.5 * F.kl_div(torch.log(gen_dist + eps), m + eps, reduction='sum').item()
            metrics['js_divergence'] = js_div
        
        # Correlation analysis
        if generated_expr.shape[0] > 1:  # Need multiple samples
            # Gene-gene correlation preservation
            real_corr = torch.corrcoef(real_expr.T)
            gen_corr = torch.corrcoef(generated_expr.T)
            
            # Remove NaN values (can occur with constant genes)
            mask = ~(torch.isnan(real_corr) | torch.isnan(gen_corr))
            if mask.sum() > 0:
                corr_preservation = torch.corrcoef(torch.stack([
                    real_corr[mask], gen_corr[mask]
                ]))[0, 1].item()
                metrics['correlation_preservation'] = corr_preservation
        
        return metrics
    
    def evaluate_perturbation_effects(
        self,
        control_expr: torch.Tensor,
        perturbed_expr: torch.Tensor,
        target_gene_idx: int,
        expected_log2fc: float
    ) -> Dict[str, float]:
        """
        Evaluate how well the model captures perturbation effects.
        
        Args:
            control_expr: Control expression values (B, N)
            perturbed_expr: Perturbed expression values (B, N)
            target_gene_idx: Index of perturbed gene
            expected_log2fc: Expected log2 fold change
            
        Returns:
            Perturbation evaluation metrics
        """
        metrics = {}
        
        # Convert to continuous if needed
        if control_expr.dtype == torch.long:
            control_expr = self.detokenizer(control_expr)
        if perturbed_expr.dtype == torch.long:
            perturbed_expr = self.detokenizer(perturbed_expr)
        
        # Compute actual log2 fold change
        control_mean = control_expr.mean(dim=0)
        perturbed_mean = perturbed_expr.mean(dim=0)
        
        # Add pseudocount to avoid log(0)
        log2fc = torch.log2((perturbed_mean + 0.1) / (control_mean + 0.1))
        
        # Target gene effect
        target_log2fc = log2fc[target_gene_idx].item()
        metrics['target_log2fc_actual'] = target_log2fc
        metrics['target_log2fc_expected'] = expected_log2fc
        metrics['target_log2fc_error'] = abs(target_log2fc - expected_log2fc)
        
        # Direction correctness
        expected_direction = 1 if expected_log2fc > 0 else -1 if expected_log2fc < 0 else 0
        actual_direction = 1 if target_log2fc > 0.1 else -1 if target_log2fc < -0.1 else 0
        metrics['direction_correct'] = int(expected_direction == actual_direction)
        
        # Off-target effects (should be minimal)
        off_target_effects = torch.abs(log2fc)
        off_target_effects[target_gene_idx] = 0  # Exclude target gene
        metrics['off_target_mean'] = off_target_effects.mean().item()
        metrics['off_target_max'] = off_target_effects.max().item()
        
        # Specificity score (ratio of on-target to off-target effect)
        if off_target_effects.mean() > 0:
            metrics['specificity_score'] = abs(target_log2fc) / off_target_effects.mean().item()
        else:
            metrics['specificity_score'] = float('inf')
        
        return metrics
    
    def compute_biological_metrics(
        self, 
        generated_expr: torch.Tensor,
        real_expr: torch.Tensor,
        gene_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute biologically relevant metrics.
        
        Args:
            generated_expr: Generated expression (B, N)
            real_expr: Real expression (B, N)
            gene_names: Optional gene names for interpretation
            
        Returns:
            Biological evaluation metrics
        """
        metrics = {}
        
        # Convert to continuous if needed
        if generated_expr.dtype == torch.long:
            generated_expr = self.detokenizer(generated_expr)
        if real_expr.dtype == torch.long:
            real_expr = self.detokenizer(real_expr)
        
        # Expression magnitude distribution
        real_total_counts = real_expr.sum(dim=1)
        gen_total_counts = generated_expr.sum(dim=1)
        
        metrics['total_count_mean_real'] = real_total_counts.mean().item()
        metrics['total_count_mean_gen'] = gen_total_counts.mean().item()
        metrics['total_count_std_real'] = real_total_counts.std().item()
        metrics['total_count_std_gen'] = gen_total_counts.std().item()
        
        # Gene expression distribution
        real_gene_means = real_expr.mean(dim=0)
        gen_gene_means = generated_expr.mean(dim=0)
        
        # Correlation of gene means
        if len(real_gene_means) > 1:
            gene_mean_corr = torch.corrcoef(torch.stack([real_gene_means, gen_gene_means]))[0, 1]
            if not torch.isnan(gene_mean_corr):
                metrics['gene_mean_correlation'] = gene_mean_corr.item()
        
        # Highly expressed gene preservation
        top_k = min(100, len(real_gene_means))
        real_top_genes = torch.topk(real_gene_means, top_k).indices.cpu()
        gen_top_genes = torch.topk(gen_gene_means, top_k).indices.cpu()
        
        # Overlap in top genes
        overlap = len(set(real_top_genes.tolist()) & set(gen_top_genes.tolist()))
        metrics[f'top_{top_k}_gene_overlap'] = overlap / top_k
        
        return metrics