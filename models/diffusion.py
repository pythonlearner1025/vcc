#!/usr/bin/env python3
"""
Conditional Discrete Diffusion Transformer Model Components.

This module contains all model-related logic for the conditional diffusion transformer,
including the model architecture, diffusion process, and associated utilities.
"""

import math
import json
import torch

# Enable fast TensorFloat-32 matmul on Ampere+
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.nn.attention import sdpa_kernel, SDPBackend

# Try to import flash_attn for native MQA support
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_qkvpacked_func
    HAS_FLASH_ATTN = True
    print("Using flash_attn for MQA support")
except ImportError:
    HAS_FLASH_ATTN = False
    print("flash_attn not available, falling back to PyTorch SDPA")

import os

# Drop-in replacements enabling FP8 kernels when TE is available
Linear = torch.nn.Linear
LayerNorm = torch.nn.LayerNorm

@dataclass
class ConditionalModelConfig:
    """Configuration for conditional diffusion model."""
    # Model architecture
    train_notes: str = ""
    dim: int = 256
    n_head: int = 8
    n_layer: int = 8
    ffn_mult: int = 8
    # Data vocabulary size (number of valid expression bins)
    vocab_size: int = 128  # 64 expression bins (data tokens only; mask is separate)
    # Optional explicit mask token id; if None it defaults to vocab_size (i.e., the first id after the data vocab)
    mask_token_id: Optional[int] = None
    token_max_value: float = 9.2
    n_genes: int = 2000  # Number of HVGs
    n_latents: int = 256

    # Training target type
    target_is_delta: bool = True  # Predict Δ rather than raw counts during fine-tune
    
    # Conditioning
    n_total_genes: int = 36601  # Total genes in vocabulary for embedding
    gene_embed_dim: int = 128
    
    # ESM2 embeddings
    esm_matrix_path: Optional[str] = None # Path to precomputed ESM2 embeddings (None to skip)
    esm_proj_dim: int = 256  # Projection dimension for ESM2 embeddings
    
    # Batch conditioning
    # TODO get true number of batches in VCC
    n_technical_batches: int = 48  # Estimate of unique batches across datasets
    batch_embed_dim: int = 64  # Batch embedding dimension
    use_batch_conditioning: bool = True
    # Latent context compression: number of latent summary tokens (0 = disabled)
    context_compressed_len: int = 0
    
    # Control set conditioning
    control_set_encoder_layers: int = 2
    control_set_dim_hidden: int = 256
    
    # Diffusion parameters
    n_timesteps: int = 10
    mask_ratio: float = 0.30  # Fraction of genes to mask (BERT-style)
    schedule: str = "cosine"
    token_distribution_json: Optional[str] = None  # Path to token distribution JSON for frequency-aware masking
    token_weighting_annealing_steps: Optional[int] = 10_000  # Steps to anneal from uniform to frequency-based masking (0 or None = no annealing)
    
    # Masking parameters
    pretrain_mask_ratio: float = 0.20  # Fixed mask ratio for pretraining
    finetune_mask_ratio_start: float = 0.30  # Starting mask ratio for finetuning
    finetune_mask_ratio_end: float = 1.0  # End mask ratio for finetuning (curriculum)
    finetune_mask_ratio_steps: int = 100_000  # Steps to ramp from start to end
    finetune_full_mask_prob: float = 0.10  # Probability of 100% masking during finetune
    
    # Training parameters
    pretrain_batch_size: int = 32
    adam_lr: float = 1e-4
    muon_lr: float = 1e-4
    warmup_steps: int = 500 # lr exp inc
    num_steps: int = 1500 # lr constant
    cooldown_steps: int = 500 # lr exp dec

    finetune_muon_lr: float = 1e-2
    finetune_adam_lr: float = 1e-4
    # Enable FP8 execution via Transformer-Engine (if available)
    use_fp8: bool = True  # Now working with custom-built wheel
    # Enable auxiliary DE heads/returns
    use_aux: bool = True
    
    # Gradient checkpointing - disable for large-batch FP8 runs to see true kernel speed
    grad_ckpt: bool = True  # Enable activation checkpointing (trades compute for memory)

    vcc_batch_size: int = 1
    vcc_set_size: int = 64
    
    # Training epochs (replacing step-based training)
    pretrain_epochs: int = 50
    finetune_epochs: int = 20
    pretrain_data_dir: str = "data/scRNA"
    finetune_data_path: str = "data/vcc_data/"
    hvg_info_path: str = ".txt"
    n_xattn_loops: int = 6
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    max_eval_genes: int = 25

    debug_pretrain_max_cells: Optional[int] = None 
    debug_finetune_max_cells: Optional[int] = None
    debug_eval_max_cells: int = 3200  # ~50 batches with batch_size=64
    full_eval: bool = False  # Run full cell-eval suite each epoch
    blacklist_path: str = "data/blacklist.txt"
    
    @property
    def n_params(self) -> int:
        """Estimate parameter count."""
        # Token embedding includes an extra row for the [MASK] token
        params = (self.vocab_size + 1) * self.dim
        params += self.n_timesteps * self.dim  # Time embedding
        params += self.n_total_genes * self.gene_embed_dim  # Gene embeddings
        
        # Batch conditioning
        params += (self.n_technical_batches if self.use_batch_conditioning else 0) * self.batch_embed_dim
        # Transformer layers
        per_layer = (
            4 * self.dim * self.dim +  # QKV + proj
            2 * self.dim * self.dim * self.ffn_mult  # FFN
        )
        params += self.n_layer * per_layer
        
        # Output head predicts only data tokens (excludes [MASK])
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

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with grouped key/value projections."""
    
    def __init__(self, dim: int, n_head: int, n_group: int = 2):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.n_group = n_group  # Store n_group as instance variable
        self.head_dim = dim // n_head
        
        # Ensure n_head is divisible by n_group
        assert n_head % n_group == 0, f"n_head ({n_head}) must be divisible by n_group ({n_group})"
        self.heads_per_group = n_head // n_group
        
        # Query projection for all heads
        self.q_proj = Linear(dim, dim)
        
        # Grouped key/value projections (n_group groups)
        self.kv_proj = Linear(dim, 2 * self.n_group * self.head_dim)
        
        # Output projection
        self.out_proj = Linear(dim, dim)

        
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

        M = kv.shape[1]  # kv sequence length (6288)
        
        # Project queries for all heads
        # (128, 6288, 12, 57)
        q = self.q_proj(x).view(B, N, self.n_head, self.head_dim)  # (B, N, H, HD)
        
        # Project grouped key/value
        # (128, 6288, G, 57)
        kv_proj = self.kv_proj(kv).view(B, M, self.n_group, 2*self.head_dim)  # (B, M, G, 2*HD)
        k, v = kv_proj.chunk(2, dim=-1)  # Each is (B, M, G, HD)

        if HAS_FLASH_ATTN:
            # Flash attention natively supports GQA when k,v have fewer heads than q
            # Ensure k and v are contiguous after chunk operation
            k = k.contiguous()  # (B, M, G, HD)
            v = v.contiguous()  # (B, M, G, HD)
            
            out = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                softmax_scale=None,  # Will default to 1/sqrt(head_dim)
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False
            )  # (B, N, H, HD)
            
            # Reshape back to (B, N, D)
            out = out.contiguous().view(B, N, D)
        else:
            # ------------------------------------------------------------------
            #  Fallback: PyTorch SDPA with K/V group expansion
            # ------------------------------------------------------------------
            # Transpose for SDPA format: (B, H, N, HD)
            q = q.transpose(1, 2)  # (B, H, N, HD)
            
            # Expand k, v from groups to all heads
            # Each group serves heads_per_group query heads
            k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)  # (B, M, heads_per_group, G, HD)
            k = k.reshape(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, HD)
            
            v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)  # (B, M, heads_per_group, G, HD)
            v = v.reshape(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, HD)
            
            # Use PyTorch's SDPA with optional flash attention backend
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=0.0, 
                    is_causal=False
                )  # (B, H, N, HD)
            
            # Reshape back to (B, N, D)
            out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Project output
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer block with MQA and cross-attention support."""
    
    def __init__(self, dim: int, n_head: int, ffn_mult: int = 4):
        super().__init__()
        # Self-attention
        self.ln1 = LayerNorm(dim)
        self.self_attn = GroupedQueryAttention(dim, n_head)
        
        # Cross-attention (optional)
        self.ln2 = LayerNorm(dim)
        self.cross_attn = GroupedQueryAttention(dim, n_head)
        
        self.ln3 = LayerNorm(dim)
        try:
            from xformers.ops import FusedDenseGeluDense
            self.mlp = FusedDenseGeluDense(dim, dim * ffn_mult, dim)
            print("Built with FusedDenseGelu")
        except ImportError:
            self.mlp = nn.Sequential(
                Linear(dim, dim * ffn_mult),
                nn.GELU(),
                Linear(dim * ffn_mult, dim))
        
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
            #print(f"block context.shape: {context.shape}")
            x = x + self.cross_attn(self.ln2(x), kv=context)
        
        # FFN
        x = x + self.mlp(self.ln3(x))
        return x

class AugmentedGeneEmbedding(nn.Module):
    """Minimal version: trainable ID embedding + frozen ESM2 table."""

    def __init__(
        self,
        n_genes: int,
        id_dim: int = 128,
        esm_matrix_path: Optional[str] = None,
        proj_dim: int = 256,
    ):
        super().__init__()

        # Trainable gene‑ID embedding
        self.id_emb = nn.Embedding(n_genes, id_dim, device="cpu")

        # If no ESM table is given we just fall back to the ID embedding.
        if esm_matrix_path is None:
            self.has_esm = False
            print("NO ESM")
            return
        try:

            print("using AugmentedGeneEmbedding")
            obj = torch.load(esm_matrix_path, map_location="cpu")
            seq_dim = obj["emb"].shape[-1]

            # Build table: one row per gene, isoforms mean‑pooled.
            table = torch.zeros(n_genes, seq_dim)
            genes = obj["genes"]
            iso_map = obj["gene_to_isoform_indices"]

            for g, iso_idxs in iso_map.items():
                if g not in genes:
                    continue
                gid = genes.index(g)
                if gid >= n_genes:
                    continue
                emb = obj["emb"][iso_idxs].mean(0) if iso_idxs else obj["emb"][gid]
                table[gid] = emb

            # Frozen ESM lookup aligned with gene indices
            self.esm_emb = nn.Embedding.from_pretrained(table, freeze=True)
            self.esm_proj = nn.Linear(seq_dim, proj_dim, bias=False)

            # Simple gated fusion back to id_dim so downstream shapes stay unchanged
            self.gate = nn.Parameter(torch.ones(1) * 2)
            self.mix = nn.Linear(id_dim + proj_dim, id_dim)
            self.has_esm = True
        except Exception as e:
            self.has_esm = False
            print(e)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """(B,) or (B,K) → (B,id_dim) or (B,K,id_dim)."""
        id_vec = self.id_emb(idx)

        if not self.has_esm:
            return id_vec

        # 4‑line sequence branch
        seq_vec = self.esm_proj(self.esm_emb(idx).float())
        fused = self.mix(torch.cat([id_vec, torch.tanh(self.gate) * seq_vec], dim=-1))
        return fused

class ConditionalDiffusionTransformer(nn.Module):
    """Conditional discrete diffusion transformer that optionally injects
    per‑gene ESM2 features into the token stream whenever the underlying
    `AugmentedGeneEmbedding` has a loaded ESM table.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Enable FP8 execution if requested and Transformer-Engine is available
        self.fp8_enabled = getattr(config, "use_fp8", False) 

        # Include one extra row for the dedicated [MASK] token at id == config.vocab_size
        self.token_emb = nn.Embedding(config.vocab_size + 1, config.dim)

        # Gene‑level embeddings (trainable IDs + optional ESM2 vectors)
        self.gene_embed = AugmentedGeneEmbedding(
            n_genes=config.n_total_genes,
            id_dim=config.gene_embed_dim,
            esm_matrix_path=config.esm_matrix_path,
            proj_dim=config.esm_proj_dim,
        )

        self.gene_proj = Linear(config.gene_embed_dim, config.dim, bias=False)
        self.fuse_proj = Linear(config.dim * 2, config.dim)

        # If ESM2 is present we project it to `dim` and cache HVG indices.
        self.use_esm_in_sequence = getattr(self.gene_embed, "has_esm", False)
        # Pre-compute and cache the projected HVG features to avoid recomputation
        with torch.no_grad():
            gene_ids = torch.arange(config.n_genes, dtype=torch.long)
            gene_feat = self.gene_embed(gene_ids)  # (N, gene_embed_dim)
            gene_feat_proj = self.gene_proj(gene_feat)  # (N, D)

        self.register_buffer("_cached_gene_features", gene_feat_proj, persistent=False)

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(config.dim),
            Linear(config.dim, config.dim),
            nn.GELU(),
            Linear(config.dim, config.dim),
        )

        if config.use_batch_conditioning:
            self.batch_embed = nn.Embedding(config.n_technical_batches, config.batch_embed_dim)
        batch_dim = config.batch_embed_dim if config.use_batch_conditioning else 0
        self.cond_proj = nn.Sequential(
            Linear(config.gene_embed_dim + batch_dim, config.dim),
            nn.GELU(),
            Linear(config.dim, config.dim),
        )

        self.control_id_emb = nn.Embedding(1, config.dim)
        # ------------------------------------------------------------------
        #  Latent transformer (Perceiver-style) 
        # ------------------------------------------------------------------
        self.n_latents = getattr(config, "n_latents", 512)
        self.lat = nn.Parameter(torch.randn(1, self.n_latents, config.dim) * 0.02)

        gqa_groups = max(1, config.n_head // 2)  # Higher grouping for cross-attention

                # First cross-attention (unique weights)
        self.cross_attn_first = GroupedQueryAttention(config.dim, config.n_head, n_group=gqa_groups)

        # Shared cross-attention for all subsequent iterations
        self.cross_attn_shared = GroupedQueryAttention(config.dim, config.n_head, n_group=gqa_groups)

        # Shared latent Transformer block
        self.latent_block_shared = TransformerBlock(config.dim, config.n_head, config.ffn_mult)

        # Final decode cross-attention (latents → gene tokens)
        self.cross_attn_decode = GroupedQueryAttention(config.dim, config.n_head, n_group=gqa_groups)

        self.target_flag = nn.Embedding(2, config.dim)

        # LayerNorms for clean pre/post-attention
        self.lat_q_ln = LayerNorm(config.dim)
        self.kv_ln = LayerNorm(config.dim)
        self.gene_q_ln = LayerNorm(config.dim)
        self.lat_kv_ln = LayerNorm(config.dim)

        self.blocks = nn.ModuleList()

        self.ln_f = LayerNorm(config.dim)
        # Predict data tokens plus the [MASK] class (K + 1)
        self.head = nn.Linear(config.dim, config.vocab_size + 1)

        # ---- DE-aware heads ----
        self.de_sig_head = nn.Linear(config.dim, 1)   # BCEWithLogits on significance
        self.de_dir_head = nn.Linear(config.dim, 2)   # CE on up/down
        self.de_rank_head = nn.Linear(config.dim, 1)  # scalar score for RankNet/hinge
        # Default controlled by config.use_aux
        self.return_aux = bool(getattr(config, "use_aux", False))

        self.apply(self._init_weights)

    # ----------------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Skip pretrained/frozen ESM table and any frozen embeddings
            if hasattr(self, "gene_embed") and getattr(self.gene_embed, "has_esm", False):
                if m is getattr(self.gene_embed, "esm_emb", None):
                    return
            if getattr(m, "weight", None) is not None and (m.weight.requires_grad is False):
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @staticmethod
    def _normalize_to_BS(
        idx: Optional[torch.Tensor],
        B: int,
        S: int,
        name: str,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Normalize index-like inputs to shape (B, S).

        Accepts scalars, 1-D, or 2-D tensors:
        - 0-D: broadcast to (B,S)
        - 1-D: if length==B → (B,S); if length==S → (B,S); if length==1 → (B,S); else error
        - 2-D: allow (B,S), (B,1), (1,S), (1,1) with expansion; else error
        Returns None if idx is None.
        """
        if idx is None:
            return None
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, device=device)
        if idx.ndim == 0:
            idx = idx.view(1, 1).expand(B, S)
        elif idx.ndim == 1:
            length = idx.shape[0]
            if length == B:
                idx = idx.view(B, 1).expand(B, S)
            elif length == S:
                idx = idx.view(1, S).expand(B, S)
            elif length == 1:
                idx = idx.view(1, 1).expand(B, S)
            else:
                raise ValueError(f"{name} has shape ({length},), expected length equal to batch size {B} or set size {S}.")
        elif idx.ndim == 2:
            b, s = idx.shape
            if (b, s) == (B, S):
                pass
            elif (b, s) == (B, 1):
                idx = idx.expand(B, S)
            elif (b, s) == (1, S):
                idx = idx.expand(B, S)
            elif (b, s) == (1, 1):
                idx = idx.expand(B, S)
            else:
                raise ValueError(f"{name} has shape ({b},{s}), expected (B,S)=({B},{S}), (B,1), (1,S) or (1,1).")
        else:
            raise ValueError(f"{name} has {idx.ndim} dims; expected 0, 1 or 2.")
        return idx.to(dtype=torch.long, device=device)

    # ----------------------------------------------------------------------
    #  Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.LongTensor,  # (B, S, N)
        timesteps: torch.LongTensor,  # (B,)
        ### Conditioning ###
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        control_context: Optional[torch.Tensor] = None,
        #------------------#
        return_aux: bool = False,
    ):
        B, S, N = tokens.shape

        # --- Token + optional ESM2 features ---
        token_emb = self.token_emb(tokens) # tokens will be 64 above
        context_emb = self.token_emb(control_context) if control_context is not None else None # tokens full 0-128 

        gene_feat = self._cached_gene_features.to(tokens.device).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        token_emb = self.fuse_proj(torch.cat([token_emb, gene_feat], dim=-1))
        if context_emb is not None:
            context_emb = self.fuse_proj(torch.cat([context_emb, gene_feat], dim=-1))

        x = token_emb

        # --- Time + conditioning embeddings ---
        t_emb = self.time_emb(timesteps.float())[:, None, None, :]

        # Normalize target gene indices and build conditioning input [gene_embed, batch_embed]
        tgt = self._normalize_to_BS(target_gene_idx, B, S, "target_gene_idx", tokens.device)

        # Batch conditioning vector (broadcasted across N)
        if self.config.use_batch_conditioning and batch_idx is not None:
            bat_idx = self._normalize_to_BS(batch_idx, B, S, "batch_idx", tokens.device)
            bat_idx = torch.clamp(bat_idx, 0, self.config.n_technical_batches - 1)
            batch_cond = self.batch_embed(bat_idx)  # (B, S, batch_embed_dim)
        else:
            batch_cond = torch.zeros(B, S, getattr(self.config, "batch_embed_dim", 0), device=tokens.device, dtype=x.dtype)

        # Gene conditioning
        if tgt is not None:
            gene_cond = self.gene_embed(tgt)  # (B, S, gene_embed_dim)
        else:
            gene_cond = torch.zeros(B, S, self.config.gene_embed_dim, device=tokens.device, dtype=x.dtype)

        # Project conditioning if batch conditioning is enabled in config
        if getattr(self.config, "use_batch_conditioning", False):
            cond_input = torch.cat([gene_cond, batch_cond], dim=-1)  # (B, S, gene_embed_dim + batch_embed_dim)
            cond_vec = self.cond_proj(cond_input)[:, :, None, :]  # (B, S, 1, D)
            x = x + cond_vec + t_emb
        else:
            x = x + t_emb

        # target flag always applied
        idx = torch.arange(N, device=tokens.device)[None, None, :]  # (1, 1, N)
        if tgt is not None:
            flag = (idx == tgt[:, :, None]).long()  # (B, S, N)
        else:
            flag = torch.zeros(B, S, N, dtype=torch.long, device=tokens.device)
        x = x + self.target_flag(flag)

        # --- Flatten for attention ---
        x_seq = x.view(B, S * N, -1)  # [B, S*N, D]
        if context_emb is not None:
            #print(f'context_emb.shape: {context_emb.shape}')
            context_emb = context_emb + self.control_id_emb.weight + t_emb  # (B, S, N, D)
            context_seq = context_emb.view(B, S * N, -1)  # [B, S*N, D]
            kv_x = torch.cat([x_seq, context_seq], dim=1)
        else:
            kv_x = x_seq

        # --- Initialize latents ---
        lat = self.lat.expand(B, -1, -1)  # (B, K, D)

        # --- Iterative cross-attn + latent self-attn ---
        use_ckpt = self.config.grad_ckpt and self.training and torch.is_grad_enabled()
        kv = self.kv_ln(kv_x)
        for i in range(self.config.n_xattn_loops):  # e.g., 8
            if i == 0:
                lat = lat + self.cross_attn_first(self.lat_q_ln(lat), kv=kv)
            else:
                lat = lat + self.cross_attn_shared(self.lat_q_ln(lat), kv=kv)

            if use_ckpt:
                lat = torch.utils.checkpoint.checkpoint(
                    lambda _lat: self.latent_block_shared(_lat),
                    lat,
                    use_reentrant=True
                )
            else:
                lat = self.latent_block_shared(lat)

        # --- Final decode cross-attn ---
        x_seq = x_seq + self.cross_attn_decode(self.gene_q_ln(x_seq), kv=self.lat_kv_ln(lat))
        x = x_seq.view(B, S, N, -1)

        # --- Output ---
        h = self.ln_f(x)
        vocab_logits = self.head(h)

        if not (return_aux or self.return_aux):
            return vocab_logits

        return {
            "logits": vocab_logits,
            "hidden": h,
            "de_sig_logit": self.de_sig_head(h).squeeze(-1),
            "de_dir_logits": self.de_dir_head(h),
            "de_rank_score": self.de_rank_head(h).squeeze(-1),
        }


class PartialMaskingDiffusion:
    """Discrete diffusion with partial masking strategy."""
    
    def __init__(self, config: ConditionalModelConfig):
        self.config = config
        self.n_timesteps = config.n_timesteps
        # Data vocabulary size (excludes [MASK])
        self.data_vocab_size = config.vocab_size
        self.mask_ratio = config.mask_ratio # TODO change this? 
        # Dedicated [MASK] token id placed right after data vocab by default
        self.mask_token_id: int = int(config.mask_token_id if config.mask_token_id is not None else self.data_vocab_size)
        print(f'mask_token_id: {self.mask_token_id} (data_vocab_size={self.data_vocab_size})')
        
        # Cosine schedule
        s = 0.008
        steps = torch.arange(self.n_timesteps + 1, dtype=torch.float32)
        alphas = torch.cos((steps / self.n_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas = alphas / alphas[0]
        base_probs = 1 - alphas[:-1]
        # Finetune-time target: end at finetune_mask_ratio_end
        eps = 1e-8
        self.p_mask_probs = base_probs * (self.config.finetune_mask_ratio_end / (base_probs[-1] + eps))
        # Pretrain-time target: end at pretrain_mask_ratio
        self.p_mask_probs_pretrain = base_probs * (self.config.pretrain_mask_ratio / (base_probs[-1] + eps))
        
        # Load token weights for frequency-aware masking
        self.token_weights = None
        if config.token_distribution_json:
            self.token_weights = self._compute_token_mask_weights(config.token_distribution_json)
    
    def _compute_token_mask_weights(self, token_distribution_json: str, smoothing: float = 1.0) -> torch.Tensor:
        """Compute token mask weights based on inverse frequency."""
        with open(token_distribution_json, 'r') as f:
            token_counts = json.load(f)
        
        counts = np.zeros(self.data_vocab_size)
        total_count = 0
        for k, v in token_counts.items():
            k = int(k)
            if k < self.data_vocab_size:
                counts[k] = v
                total_count += v
        
        # Calculate frequencies
        frequencies = counts / (total_count + 1e-10)
        
        # For tokens not in the distribution, use median frequency
        seen_mask = counts > 0
        if seen_mask.any():
            median_freq = np.median(frequencies[seen_mask])
        else:
            median_freq = 1.0 / self.data_vocab_size
        
        # Set frequency for unseen tokens
        frequencies[~seen_mask] = median_freq
        
        # Compute weights from frequencies with smoothing
        # Higher weight for rarer tokens. Use a gentle power law to avoid extreme values.
        # Example: weight ∝ (1 / (f + eps))^alpha with alpha in [0,1]
        eps = smoothing / self.data_vocab_size
        alpha = 0.5
        weights = (1.0 / (frequencies + eps)) ** alpha
        
        # Normalize so average weight is 1.0 (preserves overall mask rate)
        weights = np.minimum(weights / weights.mean(), 4.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    
    def q_sample(
        self, 
        x_start: torch.LongTensor, 
        t: torch.LongTensor,
        mask_ratio: float | torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: partially mask tokens based on timestep.
        
        Returns:
            x_noisy: Partially masked tokens
            mask: Boolean mask indicating which positions were masked
        """
        B, S, N = x_start.shape
        device = x_start.device
        # Resolve mask token id
        mask_token = self.mask_token_id

        # Use provided mask ratio; support per-sample tensors and floats
        if torch.is_tensor(mask_ratio):
            mask_prob = mask_ratio.to(device).view(B)
        else:
            mask_prob = torch.full((B,), float(mask_ratio), device=device)
        
        if self.token_weights is None:
            # Uniform masking per-position
            rand = torch.rand(B, S, N, device=device)
            mask = rand < mask_prob.view(B, 1, 1)
        else:
            # Token-aware masking with annealing
            if self.config.token_weighting_annealing_steps is None or self.config.token_weighting_annealing_steps == 0:
                # No annealing - use full frequency-based masking
                annealing_factor = 1.0
            else:
                annealing_factor = min(1.0, step / self.config.token_weighting_annealing_steps)
            
            # Get frequency-based weights for tokens
            freq_weights = self.token_weights.to(device)[x_start]  # shape: [B, S, N]
            
            # Blend uniform weights (1.0) with frequency weights based on annealing
            weights = (1.0 - annealing_factor) + annealing_factor * freq_weights 
            
            # Scale mask probability by token weight
            scaled_probs = weights * mask_prob.view(B, 1, 1)
            rand = torch.rand(B, S, N, device=device)
            mask = rand < scaled_probs.clamp(0, 1)
        
        # Apply mask
        x_noisy = torch.where(mask, mask_token, x_start)
        return x_noisy, mask

    @staticmethod
    @torch.no_grad()
    def build_de_targets(delta, signif_mode="percentile", signif_arg=0.10, mlm_mask=None):
        B, N = delta.shape
        abs_delta = delta.abs()
        if mlm_mask is None:
            valid_mask = torch.isfinite(delta)
        else:
            valid_mask = torch.isfinite(delta) & mlm_mask
        de_sig = torch.zeros_like(abs_delta, dtype=torch.bool)
        if signif_mode == "percentile":
            for b in range(B):
                vb = valid_mask[b]
                if vb.sum() == 0: continue
                k = max(1, int(vb.sum().item() * signif_arg))
                vals = abs_delta[b][vb]
                thr = vals.topk(k).values[-1]
                de_sig[b] = (abs_delta[b] >= thr) & vb
        else:
            de_sig = (abs_delta >= signif_arg) & valid_mask
        de_dir     = (delta > 0).long()
        rank_score = abs_delta
        return de_sig, de_dir, rank_score, valid_mask

    def _bce_with_logits_masked(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: Optional[float] = None):
        # logits/target/mask: (B, N)
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
            loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fn(logits, target.float())
        denom = mask.float().sum().clamp_min(1.0)
        loss = (loss * mask.float()).sum() / denom
        return loss

    def _ce_masked(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # logits: (B,N,2) | target: (B,N) | mask: (B,N)
        loss = F.cross_entropy(logits.view(-1, 2), target.contiguous().view(-1), reduction="none")
        loss = loss.view_as(mask)
        denom = mask.float().sum().clamp_min(1.0)
        loss = (loss * mask.float()).sum() / denom
        return loss

    def _ranknet_loss(self, scores, targets, mask, num_pairs=1024, margin=0.0, top_percent=0.1):
        # scores/targets/mask: (B,S,N)
        s = scores.mean(dim=1)   # (B,N)
        t = targets.mean(dim=1)  # (B,N)
        m = mask.any(dim=1)      # (B,N)
        B, N = s.shape
        eps = 0.05 * self.config.token_max_value
        total = 0.0; denom = 0
        for b in range(B):
            valid = m[b]
            tb, sb = t[b], s[b]
            top_mask = valid & (tb >= eps)
            bot_mask = valid & (tb <  eps)
            if top_mask.sum()==0 or bot_mask.sum()==0: continue
            k_top = max(1, int(top_mask.sum().item()*top_percent))
            k_bot = max(1, int(bot_mask.sum().item()*top_percent))
            top_idx = torch.where(top_mask)[0][ tb[top_mask].topk(k_top).indices ]
            bot_idx = torch.where(bot_mask)[0][ (-tb[bot_mask]).topk(k_bot).indices ]
            n = min(num_pairs, top_idx.numel(), bot_idx.numel())
            if n == 0: continue
            ti = top_idx[torch.randint(0, top_idx.numel(), (n,), device=sb.device)]
            tj = bot_idx[torch.randint(0, bot_idx.numel(), (n,), device=sb.device)]
            total += (-F.logsigmoid(sb[ti] - sb[tj] - margin)).mean()
            denom += 1
        return (total/denom) if denom>0 else s.new_zeros(())

    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.LongTensor,
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        step: int = 0,
        max_steps: int = 1000,
        # ---- new knobs ----
        lambda_de: float = 0.5,
        lambda_dir: float = 0.2,
        lambda_rank: float = 0.2,
        de_pos_weight: Optional[float] = 2.0,
        signif_mode: str = "percentile",
        signif_arg: float = 0.1,
        # ---- optional direct DE inputs ----
        delta_means: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss with partial masking and cross-attention conditioning.
        """
        
        B, S, N = x_start.shape
        device = x_start.device
        
        # Sample random timesteps from [1, T-1] to avoid zero-mask edge case at t=0
        # (t=0 yields mask ratio ~0 which can create empty masks and NaNs)
        t = torch.randint(1, self.n_timesteps, (B,), device=device)
        
        # Determine mask ratio from timestep schedule (t-dependent training)
        is_conditioned = control_set is not None
        if is_conditioned:
            mask_ratio_vec = self.p_mask_probs.to(device)[t]
        else:
            mask_ratio_vec = self.p_mask_probs_pretrain.to(device)[t]

        # Optionally inject some fully masked samples during finetuning to match sampling-time start
        if is_conditioned and getattr(self.config, "finetune_full_mask_prob", 0.0) > 0.0:
            full_mask_mask = (torch.rand(B, device=device) < float(self.config.finetune_full_mask_prob))
            mask_ratio_vec = torch.where(full_mask_mask, torch.ones_like(mask_ratio_vec), mask_ratio_vec)
        
        # Partial masking using per-sample ratios
        x_noisy, mlm_mask = self.q_sample(x_start, t, mask_ratio_vec, step=step)
        '''
        | Timestep | p\_mask\_probs | masked_tokens (total 6288)  |
        | -------- | -------------- | - |
        | 0        | 0.000000       | 0 |
        | 1        | 0.011981       | 75 |
        | 2        | 0.042598       | 268 |
        | 3        | 0.090694       | 570 |
        | 4        | 0.154449       | 971 |
        | 5        | 0.231451       | 1455 |
        | 6        | 0.318788       | 2004 |
        | 7        | 0.413158       | 2598 |
        | 8        | 0.510989       | 3213 |
        | 9        | 0.608582       | 3826 |
        | 10       | 0.702247       | 4415 |
        | 11       | 0.788438       | 4957 |
        | 12       | 0.863898       | 5432 |
        | 13       | 0.925771       | 5821 |
        | 14       | 0.971718       | 6110 |
        | 15       | 1.000000       | 6288 |
        '''
        
        # Predict original tokens
        control_ctx = control_set if isinstance(control_set, torch.Tensor) else None
        out = model(
            x_noisy, t,
            target_gene_idx=target_gene_idx,
            batch_idx=batch_idx,
            control_context=control_ctx,
            return_aux=True,
        )
        logits = out["logits"] if isinstance(out, dict) else out
        # Restrict to data vocabulary during training loss
        logits_data = logits[..., :self.data_vocab_size]
        
        # Compute loss only on masked positions; guard against rare empty-mask batches
        if mlm_mask.any():
            main_loss = F.cross_entropy(
                logits_data[mlm_mask].view(-1, self.data_vocab_size),
                x_start[mlm_mask].view(-1),
                reduction='mean'
            )
        else:
            main_loss = logits.new_zeros(())

        # (4) aux DE losses (only if we have controls)
        aux_loss = logits.new_zeros(())
        aux_sig = logits.new_zeros(())
        aux_dir = logits.new_zeros(())
        aux_rank = logits.new_zeros(())

        # TODO try with no ramp. No change in MLM loss with and without ramp
        # makes me think it doesn't actually hurt
        def ramp(step, frac=0.2):
            return min(1.0, step / max(1, int(max_steps*frac)))
        scale = ramp(step)
        if control_set is not None and isinstance(out, dict) and (delta_means is not None):
            # Reduce MLM mask over set dimension to per-gene mask for DE target building
            mlm_mask_gene = mlm_mask.any(dim=1)  # (B, N)
            de_sig, de_dir, rank_score, valid = self.build_de_targets(
                delta_means,
                signif_mode=signif_mode,
                signif_arg=signif_arg,
                mlm_mask=mlm_mask_gene,
            )

            B2, S2, N2, _ = logits.shape
            # Expand per-gene targets across set dimension
            de_sig      = de_sig.unsqueeze(1).expand(-1, S2, -1)       # (B, S, N)
            de_dir_mask = de_sig                                      # (B, S, N)
            de_dir      = de_dir.unsqueeze(1).expand(-1, S2, -1)       # (B, S, N)
            rank_score  = rank_score.unsqueeze(1).expand(-1, S2, -1)   # (B, S, N)
            valid       = valid.unsqueeze(1).expand(-1, S2, -1)        # (B, S, N)

            # Restrict aux to masked genes at each set position
            valid = valid & mlm_mask
            de_dir_mask  = valid & de_dir_mask

            # (4a) significance BCE (all valid positions)
            aux_sig = self._bce_with_logits_masked(
                out["de_sig_logit"], de_sig, valid, pos_weight=de_pos_weight
            ) * lambda_de * scale
            aux_loss = aux_loss + aux_sig

            # (4b) direction CE (only where significant)
            if de_dir_mask.any():
                aux_dir = self._ce_masked(
                    out["de_dir_logits"], de_dir, de_dir_mask
                ) * lambda_dir * scale
                aux_loss = aux_loss + aux_dir

            # (4c) ranking on |Δ| using scalar score head
            aux_rank = self._ranknet_loss(
                out["de_rank_score"], rank_score, valid, num_pairs=512
            ) * lambda_rank * scale
            aux_loss = aux_loss + aux_rank
        total_loss = main_loss + aux_loss

        # Stash detailed loss breakdown for logging
        try:
            self._last_loss_stats = {
                "loss_main": float(main_loss.detach().item()),
                "loss_aux": float(aux_loss.detach().item()),
                "loss_aux_de": float(aux_sig.detach().item()),
                "loss_aux_dir": float(aux_dir.detach().item()),
                "loss_aux_rank": float(aux_rank.detach().item()),
                "loss_total": float(total_loss.detach().item()),
            }
        except Exception:
            pass

        return total_loss
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int],
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        device: str = 'cuda',
        return_aux: bool = False
    ) -> torch.LongTensor:
        """
        Generate samples with cross-attention conditioning and optional partial context.
        """
        B, N = shape
        
        # Resolve mask token and start with all masks, single-set dimension (S=1)
        mask_token = self.mask_token_id
        x = torch.full((B, 1, N), int(mask_token), device=device, dtype=torch.long)
        
        # Prepare conditioning with S=1 shapes
        control_context = None
        if isinstance(control_set, torch.Tensor):
            # Accept (B,N) or (N,) for B=1
            if control_set.ndim == 1:
                # Disambiguate: if B==1 and length==N, treat as (N,) -> (1,N)
                control_set = control_set.view(1, -1)
            control_context = control_set.unsqueeze(1)  # (B,1,N)
        # Normalize 1-D indices to (B,1) or (1,1) depending on B
        if isinstance(target_gene_idx, torch.Tensor) and target_gene_idx.ndim == 1:
            if target_gene_idx.shape[0] == B:
                target_gene_idx = target_gene_idx.view(B, 1)
            else:
                target_gene_idx = target_gene_idx.view(1, 1).expand(B, 1)
        if isinstance(batch_idx, torch.Tensor) and batch_idx.ndim == 1:
            if batch_idx.shape[0] == B:
                batch_idx = batch_idx.view(B, 1)
            else:
                batch_idx = batch_idx.view(1, 1).expand(B, 1)
        # Iterative denoising
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get model predictions with pre-encoded conditioning
            logits = model(
                x, t_batch,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx,
                control_context=control_context,
                return_aux=False
            )  # (B,1,N,V)
            
            logits = logits / temperature
            # Restrict to data vocabulary for ancestral sampling
            logits = logits[..., :self.data_vocab_size]

            # Determine which positions to unmask at this step
            current_mask_prob = self.p_mask_probs[t]
            if t > 0:
                next_mask_prob = self.p_mask_probs[t-1]
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
            
            # Sample new tokens for positions to unmask (data tokens only)
            if unmask.any():
                probs = F.softmax(logits, dim=-1)  # (B,1,N,V_data)
                new_tokens = torch.multinomial(probs.view(-1, self.data_vocab_size), 1).view(B, 1, N)
                x[unmask] = new_tokens[unmask]
        
        return x.squeeze(1)

class AbsorbingMaskMD4Continuous(nn.Module):
    """
    Continuous-time per-token absorbing-mask diffusion (MD4-style)
    (Without per-token weighting, which requires REINFORCE)
    """

    def __init__(self, config: ConditionalModelConfig):
        super().__init__()
        self.config = config
        self.K = int(config.vocab_size)
        self.mask_token_id = int(config.mask_token_id if config.mask_token_id is not None else self.K)
        self.eps = 1e-4
        self.t1 = float(getattr(config, "t1", 1e-3))
        self.antithetic = bool(getattr(config, "antithetic_time_sampling", True))

        # Per-token polynomial exponent w_v (fixed schedule by default; non-trainable)
        power_init = torch.ones(self.K) * float(getattr(config, "power_init", 1.0))
        self.register_buffer("power_const", power_init)

    # ---- Schedule and derivatives ----
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t)[v] for continuous t ∈ [t1, 1.0]."""
        t_exp = t.unsqueeze(-1)              # [B,1]
        w = F.softplus(self.power_const).to(t.device)           # [K]
        return 1.0 - (1.0 - self.eps) * t_exp**w

    def dgamma_times_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """|dγ(t) × α(t)| magnitude for polynomial schedule = w_v / t (positive)."""
        t_exp = t.unsqueeze(-1)
        w = F.softplus(self.power_const).to(t.device)
        return w / t_exp

    # ---- Forward process ----
    def q_sample(self, x0: torch.LongTensor, t: torch.Tensor) -> torch.LongTensor:
        """Mask each token with probability 1 - α_t[v]. Supports arbitrary data shapes (B, ...)."""
        B = x0.shape[0]
        alpha_t = self.alpha(t)  # [B,K]
        flat_x0 = x0.view(B, -1).clamp_max(self.K - 1)
        keep_prob = alpha_t.gather(1, flat_x0)
        keep_prob = keep_prob.view_as(x0)
        survive = torch.rand_like(keep_prob, dtype=keep_prob.dtype) < keep_prob
        return torch.where(survive, x0, torch.full_like(x0, self.mask_token_id))
    # ---- Training loss ----
    def compute_loss(
        self,
        model: nn.Module,
        x0: torch.LongTensor,
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        **_: dict,
    ) -> torch.Tensor:
        """
        Continuous-time MD4 objective with REINFORCE variance reduction.
        """
        B = x0.shape[0]
        device = x0.device

        # Antithetic t sampling
        if self.antithetic:
            t0 = torch.rand((), device=device)
            t = (t0 + torch.arange(0.0, 1.0, step=1.0 / max(1, B), device=device)) % 1.0
        else:
            t = torch.rand(B, device=device)
        # Scale to [t1, 1.0]
        t = (1 - self.t1) * t + self.t1

        # Single z_t sample (no REINFORCE)
        zt = self.q_sample(x0, t)
        out = model(
            zt,
            t,
            target_gene_idx=target_gene_idx,
            batch_idx=batch_idx,
            control_context=control_set,
            return_aux=False,
        )
        logits = out["logits"]
        loss, raw_loss = self._masked_ce_loss(zt, x0, logits, t)
        return loss, raw_loss

    def _masked_ce_loss(self, x_t, x0, logits, t):
        """
        Masked cross-entropy weighted by dgamma_times_alpha(t),
        as in MD4's continuous-time ELBO integrand.
        """
        # Ensure we have logits over data + mask classes
        log_probs = F.log_softmax(logits[..., : self.K], dim=-1)
        one_hot_x0 = F.one_hot(x0.clamp_max(self.K-1), num_classes=self.K).float()
        neg_ce = -(one_hot_x0 * log_probs).sum(dim=-1)  # [B,S,N]
        mask = (x_t == self.mask_token_id).float()
        weight = self.dgamma_times_alpha(t).mean(dim=-1).view(x_t.shape[0], 1, 1)
        loss = (mask * neg_ce * weight).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0).mean()
        unweighted_loss = (mask * neg_ce).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).mean()
        return loss, unweighted_loss

    # ---- Inference ----
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
    ):
        """
        Continuous-time ancestral sampler from t=1.0 to t1 using MD4's update rule.
        shape: (B, *data_shape) — must match model output's spatial shape.
        """
        B = shape[0]
        device = next(model.parameters()).device
        x_t = torch.full(shape, self.mask_token_id, device=device, dtype=torch.long)

        # Define discrete step grid from 1.0 down to t1
        times = torch.linspace(1.0, self.t1, self.config.n_timesteps, device=device)

        for i in range(len(times) - 1):
            t_val = times[i]
            s_val = times[i + 1]

            t_batch = torch.full((B,), t_val, device=device)
            s_batch = torch.full((B,), s_val, device=device)

            alpha_t = self.alpha(t_batch)  # [B,K]
            alpha_s = self.alpha(s_batch)  # [B,K]

            logits = model(
                x_t,
                t_batch,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx,
                control_context=control_set,
                return_aux=False,
            )
            mean_preds = F.softmax(logits[..., :self.K], dim=-1)  # [B,...,K+1]

            # Compute unmask probability for each vocab entry: (α_s - α_t) / (1 - α_t)
            # For fixed schedule, α is same for all v in 0..K-1
            unmask_prob = (alpha_s[:, 0] - alpha_t[:, 0]) / (1.0 - alpha_t[:, 0]).clamp_min(1e-8)
            # Broadcast to data shape
            unmask_prob = unmask_prob.view(B, *([1] * (x_t.ndim - 1)))

            # Probabilities for vocab tokens (exclude mask logit)
            probs_vocab = unmask_prob.unsqueeze(-1) * mean_preds[..., :self.K]
            # Probability for staying masked
            probs_mask = (1.0 - unmask_prob).unsqueeze(-1).expand_as(mean_preds[..., :1])
            # Concatenate vocab and mask probabilities
            probs = torch.cat([probs_vocab, probs_mask], dim=-1)

            # Sample from categorical
            x_new = torch.multinomial(probs.view(-1, self.K + 1), 1).view_as(x_t)

            # Only replace masked positions
            is_mask = x_t == self.mask_token_id
            x_t = torch.where(is_mask, x_new, x_t)

        return x_t

def create_muon_optimizer(model: nn.Module, config: ConditionalModelConfig):
    """Create Muon optimizer with weight decay.

    Falls back to a non-distributed variant when ``torch.distributed`` is not
    initialised (e.g. during local profiling).
    """
    # ---------------------------------------------------------
    # Build parameter groups
    # ---------------------------------------------------------
    hidden_weights = [
        p for n, p in model.named_parameters()
        if p.ndim == 2 and
        not any(x in n for x in ["token_emb", "id_emb", "batch_embed"])
    ]

    hidden_gains_biases = [
        p for n, p in model.named_parameters()
        if (p.ndim == 1 or p.ndim == 0) and
        not any(x in n for x in ["token_emb", "id_emb", "batch_embed"])
    ]

    nonhidden_params = [
        p for n, p in model.named_parameters()
        if any(x in n for x in ["token_emb", "id_emb", "batch_embed", "ln", "norm"])
    ]

    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=config.muon_lr, weight_decay=0.01),
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
             lr=config.adam_lr, betas=(0.9, 0.95), weight_decay=0.01),
    ]

    # ---------------------------------------------------------
    # Select the appropriate optimizer implementation
    # ---------------------------------------------------------
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        OptimClass = MuonWithAuxAdam
    else:
        OptimClass = SingleDeviceMuonWithAuxAdam

    optimizer = OptimClass(param_groups)
    return optimizer

def get_lr(optimizer, it: int, cfg: ConditionalModelConfig):
    assert it <= cfg.num_steps
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_steps:
        return (it+1) / cfg.warmup_steps
    # 2) constant lr for a while
    elif it < cfg.num_steps - cfg.cooldown_steps:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (cfg.num_steps - it) / cfg.cooldown_steps
        return decay_ratio