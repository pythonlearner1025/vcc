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

# Try to import flash_attn
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

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
    
    # ESM2 embeddings
    esm_matrix_path: str = "esm_all.pt"  # Path to precomputed ESM2 embeddings
    esm_proj_dim: int = 256  # Projection dimension for ESM2 embeddings
    
    # Batch conditioning
    # TODO get true number of batches in VCC
    n_batches: int = 100  # Estimate of unique batches across datasets
    batch_embed_dim: int = 64  # Batch embedding dimension
    use_batch_conditioning: bool = True
    
    # Control set conditioning
    control_set_encoder_layers: int = 2
    control_set_dim_hidden: int = 128
    
    # Diffusion parameters
    n_timesteps: int = 10
    mask_ratio: float = 0.30  # Fraction of genes to mask (BERT-style)
    schedule: str = "cosine"
    
    # Masking parameters
    pretrain_mask_ratio: float = 0.20  # Fixed mask ratio for pretraining
    finetune_mask_ratio_start: float = 0.30  # Starting mask ratio for finetuning
    finetune_mask_ratio_end: float = 0.90  # End mask ratio for finetuning (curriculum)
    finetune_mask_ratio_steps: int = 100_000  # Steps to ramp from start to end
    finetune_full_mask_prob: float = 0.10  # Probability of 100% masking during finetune
    
    # Training parameters
    pretrain_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    vcc_batch_size: int = 1
    vcc_set_size: int = 64
    
    # Training epochs (replacing step-based training)
    pretrain_epochs: int = 50
    finetune_epochs: int = 20
    pretrain_data_dir: str = "data/scRNA"
    finetune_data_path: str = "data/vcc_data/"
    hvg_info_path: str = ".txt"
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    vcc_eval_interval: int = 5000

    debug_pretrain_max_cells: Optional[int] = None 
    debug_finetune_max_cells: Optional[int] = None
    debug_eval_max_cells: int = 3200  # ~50 batches with batch_size=64
    
    @property
    def n_params(self) -> int:
        """Estimate parameter count."""
        params = self.vocab_size * self.dim  # Token embedding
        params += self.n_genes * self.dim  # Position embedding
        params += self.n_timesteps * self.dim  # Time embedding
        params += self.n_total_genes * self.gene_embed_dim  # Gene embeddings
        
        # Batch conditioning
        if self.use_batch_conditioning:
            params += self.n_batches * self.batch_embed_dim  # Batch embeddings
        
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
        
        # Use flash attention if available
        self.use_flash_attn = HAS_FLASH_ATTN
        
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
        q = self.q_proj(x).view(B, N, self.n_head, self.head_dim)  # (B, N, H, HD)
        
        # Project shared key/value
        kv_proj = self.kv_proj(kv)  # (B, M, 2*HD)
        k, v = kv_proj.chunk(2, dim=-1)  # Each is (B, M, HD)
        
        if self.use_flash_attn:
            # For flash_attn, we need to expand k, v for all heads
            # flash_attn expects (batch, seq_len, n_heads, head_dim)
            k_expanded = k.unsqueeze(2).expand(-1, -1, self.n_head, -1)  # (B, M, H, HD)
            v_expanded = v.unsqueeze(2).expand(-1, -1, self.n_head, -1)  # (B, M, H, HD)
            
            # Use flash attention
            out = flash_attn_func(q, k_expanded, v_expanded, causal=False)  # (B, N, H, HD)
            
            # Reshape output
            out = out.view(B, N, D)
        else:
            # Standard attention implementation
            q = q.transpose(1, 2)  # (B, H, N, HD)
            
            # Expand k, v for all heads
            k = k.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # (B, H, M, HD)
            v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # (B, H, M, HD)
            
            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, M)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # (B, H, N, HD)
            
            # Reshape
            out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Project output
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


class AugmentedGeneEmbedding(nn.Module):
    """Combines trainable gene ID embeddings with frozen ESM2 protein embeddings."""
    
    def __init__(self, n_genes: int, id_dim: int = 128, 
                 esm_matrix_path: str = "esm_all.pt", proj_dim: int = 256,
                 use_canonical_only: bool = True):
        super().__init__()
        # 1) Existing small, trainable ID embedding
        self.id_emb = nn.Embedding(n_genes, id_dim)
        self.use_canonical_only = use_canonical_only
        
        # 2) Load frozen ESM table
        try:
            obj = torch.load(esm_matrix_path, map_location="cpu")
            self.esm_emb = nn.Embedding.from_pretrained(obj["emb"], freeze=True)
            self.esm_dim = obj["emb"].shape[-1]
            
            # Load gene list and isoform mapping
            self.genes = obj.get("genes", [])
            self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes)}
            self.gene_to_isoform_indices = obj.get("gene_to_isoform_indices", {})
            
            # Create mapping from gene index to ESM embedding index
            self.gene_idx_to_esm_idx = torch.zeros(n_genes, dtype=torch.long)
            for gene, esm_indices in self.gene_to_isoform_indices.items():
                if gene in self.gene_to_idx:
                    gene_idx = self.gene_to_idx[gene]
                    if gene_idx < n_genes and esm_indices:
                        # Use first isoform (canonical) by default
                        self.gene_idx_to_esm_idx[gene_idx] = esm_indices[0]
            
            self.has_esm = True
            print(f"Loaded ESM2 embeddings from {esm_matrix_path} with shape {obj['emb'].shape}")
            print(f"Found {len(self.gene_to_isoform_indices)} genes with isoforms")
            n_multi_isoform = sum(1 for indices in self.gene_to_isoform_indices.values() if len(indices) > 1)
            print(f"Genes with multiple isoforms: {n_multi_isoform}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load ESM2 embeddings from {esm_matrix_path}: {e}")
            print("Using only trainable ID embeddings.")
            self.has_esm = False
        
        if self.has_esm:
            # 3) Projection + gating
            self.esm_proj = nn.Linear(self.esm_dim, proj_dim)
            self.gate = nn.Parameter(torch.zeros(1))
            self.mix = nn.Sequential(
                nn.Linear(id_dim + proj_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, id_dim)  # back to id_dim so existing code needn't change
            )
        
    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            idx: Gene indices (B,) or (B, K) 
            
        Returns:
            Gene embeddings (B, id_dim) or (B, K, id_dim)
        """
        id_vec = self.id_emb(idx)  # (B, id_dim) or (B, K, id_dim)
        
        if not self.has_esm:
            return id_vec
        
        # Handle indices that might be out of range
        device = id_vec.device
        shape = idx.shape
        idx_flat = idx.view(-1)
        
        # Map gene indices to ESM embedding indices
        # Move mapping to device if needed
        if self.gene_idx_to_esm_idx.device != device:
            self.gene_idx_to_esm_idx = self.gene_idx_to_esm_idx.to(device)
        
        # Create mask for valid gene indices
        valid_gene_mask = idx_flat < len(self.gene_idx_to_esm_idx)
        
        # Initialize seq_vec with zeros
        seq_vec_flat = torch.zeros(len(idx_flat), self.esm_proj.out_features, device=device)
        
        # Get ESM embeddings for valid genes
        if valid_gene_mask.any():
            valid_gene_idx = idx_flat[valid_gene_mask]
            # Map gene indices to ESM indices
            esm_indices = self.gene_idx_to_esm_idx[valid_gene_idx]
            
            # Check which mapped indices are valid in ESM embedding
            valid_esm_mask = (esm_indices > 0) & (esm_indices < self.esm_emb.num_embeddings)
            
            if valid_esm_mask.any():
                # Get embeddings only for valid ESM indices
                valid_esm_indices = esm_indices[valid_esm_mask]
                valid_esm = self.esm_emb(valid_esm_indices)
                
                # Project ESM embeddings
                projected = self.esm_proj(valid_esm)
                
                # Place projected embeddings in the right positions
                valid_positions = valid_gene_mask.nonzero().squeeze(-1)[valid_esm_mask]
                seq_vec_flat[valid_positions] = projected
        
        # Reshape back to original shape
        seq_vec = seq_vec_flat.view(*shape, -1)
        
        # Fuse ID and ESM embeddings
        fused = self.mix(torch.cat([id_vec, self.gate.tanh() * seq_vec], -1))
        
        return fused


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
            d_out=config.dim,
            n_layers=config.control_set_encoder_layers
        )
        
        # Conditioning embeddings
        self.gene_embed = AugmentedGeneEmbedding(
            n_genes=config.n_total_genes,
            id_dim=config.gene_embed_dim,
            esm_matrix_path=config.esm_matrix_path,
            proj_dim=config.esm_proj_dim
        )
        
        # Batch embedding
        if config.use_batch_conditioning:
            self.batch_embed = nn.Embedding(config.n_batches, config.batch_embed_dim)
        
        # Combine conditioning
        batch_dim = config.batch_embed_dim if config.use_batch_conditioning else 0
        total_cond_dim = config.gene_embed_dim + batch_dim
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
        batch_idx: Optional[torch.LongTensor] = None,  # (B,) batch indices
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention conditioning.
        
        Args:
            tokens: Input tokens of shape (batch_size, n_genes)
            timesteps: Diffusion timesteps of shape (batch_size,)
            control_set: Tokenized control cells (batch_size, set_size, n_genes)
            target_gene_idx: Target gene indices
            batch_idx: Batch indices for technical variation conditioning
            
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
        if target_gene_idx is not None:
            # Gene embedding - clamp indices to valid range to avoid CUDA assertion errors
            target_gene_idx_clamped = torch.clamp(target_gene_idx, 0, self.config.n_total_genes - 1)
            gene_emb = self.gene_embed(target_gene_idx_clamped)  # (B, gene_embed_dim)
            
            # Combine embeddings
            cond_components = [gene_emb]
            
            # Add batch embedding if provided
            if self.config.use_batch_conditioning and batch_idx is not None:
                batch_idx_clamped = torch.clamp(batch_idx, 0, self.config.n_batches - 1)
                batch_emb = self.batch_embed(batch_idx_clamped)  # (B, batch_embed_dim)
                cond_components.append(batch_emb)
            
            cond_emb = torch.cat(cond_components, dim=-1)
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
            # Fine-tuning: check for random 100% masking injection
            if torch.rand(1).item() < self.config.finetune_full_mask_prob:
                return 1.0  # 100% masking
            
            # Otherwise, use curriculum learning: ramp from start to end ratio
            progress = min(1.0, step / self.config.finetune_mask_ratio_steps)
            base_ratio = (self.config.finetune_mask_ratio_start + 
                         (self.config.finetune_mask_ratio_end - self.config.finetune_mask_ratio_start) * progress)
            
            # Sample from beta distribution for variance
            # Beta(2, 5) gives a right-skewed distribution favoring lower values
            u = torch.distributions.Beta(2, 5).sample().item()
            return base_ratio * u
        else:
            # Pretraining: use fixed mask ratio with beta sampling for variance
            u = torch.distributions.Beta(2, 5).sample().item()
            return self.config.pretrain_mask_ratio * u
    
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
        batch_idx: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute training loss with partial masking and cross-attention conditioning.
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
            batch_idx=batch_idx
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
        batch_idx: Optional[torch.LongTensor] = None,
        mask_token: int = 63,
        temperature: float = 1.0,
        device: str = 'cuda',
    ) -> torch.LongTensor:
        """
        Generate samples with cross-attention conditioning and optional partial context.
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
                batch_idx=batch_idx
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

