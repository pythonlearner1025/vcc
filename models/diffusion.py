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

# Replace all LayerNorm with RMSNorm throughout this module.
# Prefer PyTorch's built-in RMSNorm when available; otherwise, use a local fallback.
try:
    from torch.nn import RMSNorm as _TorchRMSNorm  # PyTorch >= 2.5
    _HAS_TORCH_RMSNORM = True
except Exception:
    _HAS_TORCH_RMSNORM = False

if _HAS_TORCH_RMSNORM:
    RMSNorm = _TorchRMSNorm
else:
    class RMSNorm(nn.Module):
        def __init__(
            self,
            normalized_shape: int | tuple[int, ...],
            eps: float = 1e-8,
            elementwise_affine: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = float(eps)
            self.elementwise_affine = bool(elementwise_affine)
            factory_kwargs = {"device": device, "dtype": dtype}
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter("weight", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(-len(self.normalized_shape), 0))
            rms_inv = x.pow(2).mean(dim=dims, keepdim=True).add(self.eps).rsqrt()
            y = x * rms_inv
            if self.elementwise_affine:
                y = y * self.weight
            return y

# Alias so existing LayerNorm instantiations become RMSNorm without further changes
LayerNorm = RMSNorm

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
    vocab_size: int = 64  # 64 expression bins (data tokens only; mask is separate)
    # Optional explicit mask token id; if None it defaults to vocab_size (i.e., the first id after the data vocab)
    mask_token_id: Optional[int] = None
    token_max_value: float = 9.2
    n_genes: int = 2000  # Number of HVGs
    n_latents: int = 256

    # Learned set compressor (per set) → memory tokens (M ≪ N)
    mem_slots: int = 8           # M, choose so S·M ≤ 256 (with S set size)
    compressor_iters: int = 2    # routing iterations
    mem_kdim: Optional[int] = None  # projection dim for Q/K in decode (defaults to dim)
    use_memory_rope: bool = True  # Apply RoPE to memory tokens for structured attention

    # Training target type
    target_is_delta: bool = True  # Predict Δ rather than raw counts during fine-tune
    
    # Conditioning
    n_total_genes: int = 36601  # Total genes in vocabulary for embedding
    gene_embed_dim: int = 128
    
    # ESM2 embeddings
    esm_matrix_path: Optional[str] = "esm_all.pt" # Path to precomputed ESM2 embeddings (None to skip)
    esm_proj_dim: int = 256  # Projection dimension for ESM2 embeddings
    
    # Batch conditioning
    # TODO get true number of batches in VCC
    n_technical_batches: int = 48  # Estimate of unique batches across datasets
    batch_embed_dim: int = 64  # Batch embedding dimension
    use_batch_conditioning: bool = False
    use_memory_rope: bool = True
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
        seq_vec = self.esm_proj(self.esm_emb(idx))
        fused = self.mix(torch.cat([id_vec, torch.tanh(self.gate) * seq_vec], dim=-1))
        return fused
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
        q = self.q_proj(x).view(B, N, self.n_head, self.head_dim)  # (B, N, H, HD)
        
        # Project grouped key/value
        # (128, 6288, G, 57)
        kv_proj = self.kv_proj(kv).view(B, M, self.n_group, 2*self.head_dim)  # (B, M, G, 2*HD)
        k, v = kv_proj.chunk(2, dim=-1)  # Each is (B, M, G, HD)

        # QK-norm (RMS along head_dim) before dot-product for stability
        eps = 1e-6
        def rmsnorm_lastdim(t: torch.Tensor) -> torch.Tensor:
            # Normalize in fp32, cast back to original dtype to ensure FA sees bf16/fp16
            rms_inv = (t.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt().to(t.dtype)
            return t * rms_inv

        q_norm = rmsnorm_lastdim(q)
        k_norm = rmsnorm_lastdim(k)
        v_norm = rmsnorm_lastdim(v)

        if HAS_FLASH_ATTN and x.is_cuda:
            # Flash attention natively supports GQA when k,v have fewer heads than q
            # Ensure k and v are contiguous after chunk operation
            k = k_norm.contiguous()  # (B, M, G, HD)
            v = v_norm.contiguous()  # (B, M, G, HD)
            
            # Flash attn expects (B, N, H, HD) for q and (B, M, G, HD) for grouped k/v
            out = flash_attn_func(
                q_norm, k, v,
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
            qn = q_norm.transpose(1, 2)  # (B, H, N, HD)
            
            # Expand k, v from groups to all heads
            # Each group serves heads_per_group query heads
            k = k_norm.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)  # (B, M, heads_per_group, G, HD)
            k = k.reshape(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, HD)
            
            v = v_norm.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)  # (B, M, heads_per_group, G, HD)
            v = v.reshape(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, HD)
            
            # Use PyTorch's SDPA; pick backend based on device
            backend = SDPBackend.FLASH_ATTENTION if qn.is_cuda else SDPBackend.MATH
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(
                    qn, k, v, 
                    dropout_p=0.0, 
                    is_causal=False
                )  # (B, H, N, HD)
            
            # Reshape back to (B, N, D)
            out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Project output
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer block with MQA and cross-attention support, with residual scaling."""
    
    def __init__(self, dim: int, n_head: int, ffn_mult: int = 4, n_group: int = 4, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.ln1 = LayerNorm(dim)
        self.cross_attn = GroupedQueryAttention(dim, n_head, n_group=n_group)
        
        self.ln2 = LayerNorm(dim)
        try:
            from xformers.ops import FusedDenseGeluDense
            self.mlp = FusedDenseGeluDense(dim, dim * ffn_mult, dim)
        except ImportError:
            self.mlp = nn.Sequential(
                Linear(dim, dim * ffn_mult),
                nn.GELU(),
                Linear(dim * ffn_mult, dim)
            )

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if kv is not None:
            x = x + self.residual_scale * self.cross_attn(self.ln1(x), kv=kv)
        else:
            x = x + self.residual_scale * self.cross_attn(self.ln1(x))
        
        x = x + self.residual_scale * self.mlp(self.ln2(x))
        return x

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding applied along the gene axis (N).

    Applies 2D rotations to interleaved feature pairs. Expects even hidden dim.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires even hidden dimension")
        self.dim = dim
        self.base = float(base)
        # Fix: Correct frequency calculation for RoPE
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _sin_cos(self, positions: torch.Tensor, dtype: torch.dtype, device: torch.device):
        freqs = torch.outer(positions, self.inv_freq)  # (N, D/2)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos[None, None, ...], sin[None, None, ...]  # (1,1,N,D/2)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B,S,N,D), positions: (N,)
        cos, sin = self._sin_cos(positions, x.dtype, x.device)
        return self._apply_rope(x, cos, sin)

class LearnedSetCompressorFA(nn.Module):
    """
    Compress (B,S,N,D) genes -> (B,S,M,D) memory using slot attention with FlashAttention.
    Permutation-invariant (no positions required).
    """

    def __init__(self, dim: int, M: int = 8, heads: int = 8, iters: int = 2):
        super().__init__()
        assert dim % heads == 0
        self.M, self.H, self.Hd = M, heads, dim // heads
        self.iters = int(iters)
        self.slots = nn.Parameter(torch.randn(1, 1, M, dim) * 0.02)
        self.q_proj = Linear(dim, dim, bias=False)
        self.k_proj = Linear(dim, dim, bias=False)
        self.v_proj = Linear(dim, dim, bias=False)
        self.o_proj = Linear(dim, dim, bias=False)
        # Removed norm_gene to preserve RoPE information and prevent gradient instability
        self.norm_gene = LayerNorm(dim)  
        self.norm_slot = LayerNorm(dim)

    def _reshape_h(self, x: torch.Tensor) -> torch.Tensor:
        B_, L, D = x.shape
        return x.view(B_, L, self.H, self.Hd)

    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        """
        x_gene: (B,S,N,D)
        returns mem: (B,S,M,D)
        """
        B, S, N, D = x_gene.shape
        BS = B * S

        # Project gene features directly (no norm to preserve RoPE)
        xg = x_gene.reshape(BS, N, D)
        k = self._reshape_h(self.k_proj(self.norm_gene(xg)))  # (BS,N,H,Hd)
        v = self._reshape_h(self.v_proj(self.norm_gene(xg)))  # (BS,N,H,Hd)

        # init slots per set
        slots = self.slots.expand(B, S, -1, -1).reshape(BS, self.M, D)
        for _ in range(self.iters):
            qs = self._reshape_h(self.q_proj(self.norm_slot(slots)))  # (BS,M,H,Hd)

            if HAS_FLASH_ATTN:
                out = flash_attn_func(
                    qs, k, v,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False,
                )  # (BS,M,H,Hd)
            else:
                qh = qs.transpose(1, 2)  # (BS,H,M,Hd)
                kh = k.transpose(1, 2)   # (BS,H,N,Hd)
                vh = v.transpose(1, 2)   # (BS,H,N,Hd)
                out = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
                out = out.transpose(1, 2)  # (BS,M,H,Hd)

            out = out.reshape(BS, self.M, D)
            slots = slots + self.o_proj(out)  # residual update

        mem = slots.view(B, S, self.M, D)
        return mem

class DecodeAttentionFA(nn.Module):
    """
    Attend from genes (N) to set memory (M) using FlashAttention.
    Maps (B,S,N,D) gene queries over (B,S,M,D) memory -> (B,S,N,D_out)
    """

    def __init__(self, dim: int, heads: int = 8, ctx_dim: Optional[int] = None):
        super().__init__()
        assert dim % heads == 0
        self.H, self.Hd = heads, dim // heads
        d_out = ctx_dim or dim
        self.q_proj = Linear(dim, dim, bias=False)
        self.k_proj = Linear(dim, dim, bias=False)
        self.v_proj = Linear(dim, dim, bias=False)
        self.o_proj = Linear(dim, d_out, bias=False)
        self.q_norm = LayerNorm(dim)
        self.k_norm = LayerNorm(dim)

    def _reshape_h(self, x: torch.Tensor) -> torch.Tensor:
        B_, L, D = x.shape
        return x.view(B_, L, self.H, self.Hd)

    def forward(self, gene_q: torch.Tensor, mem_kv: torch.Tensor) -> torch.Tensor:
        """
        gene_q: (B,S,N,D)
        mem_kv: (B,S,M,D)
        returns ctx: (B,S,N,D_out)
        """
        B, S, N, D = gene_q.shape
        _, _, M, _ = mem_kv.shape
        BS = B * S

        q = self._reshape_h(self.q_proj(self.q_norm(gene_q).reshape(BS, N, D)))  # (BS,N,H,Hd)
        k = self._reshape_h(self.k_proj(self.k_norm(mem_kv).reshape(BS, M, D)))  # (BS,M,H,Hd)
        v = self._reshape_h(self.v_proj(mem_kv.reshape(BS, M, D)))               # (BS,M,H,Hd)

        if HAS_FLASH_ATTN:
            out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)  # (BS,N,H,Hd)
        else:
            qh = q.transpose(1, 2)  # (BS,H,N,Hd)
            kh = k.transpose(1, 2)  # (BS,H,M,Hd)
            vh = v.transpose(1, 2)  # (BS,H,M,Hd)
            out = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2)  # (BS,N,H,Hd)

        out = out.reshape(BS, N, D)
        return self.o_proj(out).view(B, S, N, -1)

class ConditionalDiffusionTransformer(nn.Module):
    """Conditional discrete diffusion transformer with residual scaling in blocks."""
    
    def __init__(self, config, residual_scale: float = 1.0):
        super().__init__()
        self.config = config
        self.fp8_enabled = getattr(config, "use_fp8", False) 

        self.token_emb = nn.Embedding(config.vocab_size + 1, config.dim)
        self.gene_embed = AugmentedGeneEmbedding(
            n_genes=config.n_total_genes,
            id_dim=config.gene_embed_dim,
            esm_matrix_path=config.esm_matrix_path,
            proj_dim=config.esm_proj_dim,
        )
        self.gene_proj = Linear(config.gene_embed_dim, config.dim, bias=False)
        self.fuse_proj = Linear(config.dim * 2, config.dim)

        self.use_esm_in_sequence = getattr(self.gene_embed, "has_esm", False)
        with torch.no_grad():
            gene_ids = torch.arange(config.n_genes, dtype=torch.long)
            gene_feat = self.gene_embed(gene_ids)
            gene_feat_proj = self.gene_proj(gene_feat)
        self.register_buffer("_cached_gene_features", gene_feat_proj, persistent=False)

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
        # Learned set compressor (FlashAttention) to produce memory tokens per set
        self.mem_slots = int(getattr(config, "n_latents", 8))
        self.x_compressor = LearnedSetCompressorFA(
            dim=config.dim,
            M=self.mem_slots,
            heads=config.n_head,
            iters=int(getattr(config, "compressor_iters", 2)),
        )
        self.c_compressor = LearnedSetCompressorFA(
            dim=config.dim,
            M=self.mem_slots,
            heads=config.n_head,
            iters=int(getattr(config, "compressor_iters", 2)),
        )

        gqa_groups = max(1, config.n_head // 2)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.dim, config.n_head, config.ffn_mult, n_group=gqa_groups, residual_scale=residual_scale)
            for _ in range(config.n_layer)
        ])
        # Decode attention from genes to memory using FlashAttention
        self.decode_attn = DecodeAttentionFA(config.dim, heads=config.n_head, ctx_dim=config.dim)

        self.target_flag = nn.Embedding(2, config.dim)
        self.segment_emb = nn.Embedding(2, config.dim)

        self.lat_q_ln = LayerNorm(config.dim)
        self.kv_ln = LayerNorm(config.dim)

        # Rotary positional embedding along gene dimension
        self.rope = RotaryPositionalEmbedding(config.dim)
        
        # Memory RoPE: Apply positional encoding to compressed memory tokens
        # Pros: Ordered specialization, structured attention, better routing
        # Cons: Loss of permutation invariance, might need tuning for stability
        self.use_memory_rope = getattr(config, 'use_memory_rope', True)
        if self.use_memory_rope:
            # Use same base as gene RoPE for stability (10000.0)
            # Higher bases can cause gradient instability during early training
            self.memory_rope = RotaryPositionalEmbedding(config.dim, base=10000.0)

        self.ln_f = LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size + 1)
        self.de_sig_head = nn.Linear(config.dim, 1)
        self.de_dir_head = nn.Linear(config.dim, 2)
        self.de_rank_head = nn.Linear(config.dim, 1)
        self.return_aux = bool(getattr(config, "use_aux", False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            if hasattr(self, "gene_embed") and getattr(self.gene_embed, "has_esm", False):
                if m is getattr(self.gene_embed, "esm_emb", None):
                    return
            if getattr(m, "weight", None) is not None and not m.weight.requires_grad:
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @staticmethod
    def _normalize_to_BS(idx: Optional[torch.Tensor], B: int, S: int, name: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
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
                raise ValueError(f"{name} shape mismatch")
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
                raise ValueError(f"{name} shape mismatch")
        else:
            raise ValueError(f"{name} ndim invalid")
        return idx.to(dtype=torch.long, device=device)

    def forward(self, tokens, timesteps, target_gene_idx=None, batch_idx=None, control_context=None, return_aux=False):
        B, S, N = tokens.shape
        token_emb = self.token_emb(tokens)
        context_emb = self.token_emb(control_context) if control_context is not None else None

        gene_feat = self._cached_gene_features.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        token_emb = self.fuse_proj(torch.cat([token_emb, gene_feat], dim=-1))
        if context_emb is not None:
            context_emb = self.fuse_proj(torch.cat([context_emb, gene_feat], dim=-1))

        x = token_emb
        t_emb = self.time_emb(timesteps.float())[:, None, None, :]

        if target_gene_idx is not None:
            tgt = self._normalize_to_BS(target_gene_idx, B, S, "target_gene_idx", tokens.device)
            gene_cond = self.gene_embed(tgt)
        else:
            gene_cond = torch.zeros(B, S, self.config.gene_embed_dim, device=tokens.device, dtype=x.dtype)

        if self.config.use_batch_conditioning and batch_idx is not None:
            bat_idx = self._normalize_to_BS(batch_idx, B, S, "batch_idx", tokens.device)
            bat_idx = torch.clamp(bat_idx, 0, self.config.n_technical_batches - 1)
            batch_cond = self.batch_embed(bat_idx)
        else:
            batch_cond = torch.zeros(B, S, getattr(self.config, "batch_embed_dim", 0), device=tokens.device, dtype=x.dtype)

        if getattr(self.config, "use_batch_conditioning", False):
            cond_input = torch.cat([gene_cond, batch_cond], dim=-1)
            cond_vec = self.cond_proj(cond_input)[:, :, None, :]
            x = x + cond_vec + t_emb
            context_emb = context_emb + cond_vec + t_emb
        else:
            x = x + t_emb
            if context_emb is not None:
                context_emb = context_emb + t_emb

        idx = torch.arange(N, device=tokens.device)[None, None, :]
        if target_gene_idx is not None:
            flag = (idx == tgt[:, :, None]).long()
        else:
            flag = torch.zeros(B, S, N, dtype=torch.long, device=tokens.device)
        x = x + self.target_flag(flag)

        # Apply RoPE to x and context along gene axis (N)
        positions = torch.arange(N, device=tokens.device)
        x = self.rope(x, positions)
        if context_emb is not None:
            context_emb = self.rope(context_emb, positions)
            #x = torch.cat([x, context_emb], dim=1)  # Concatenate along sequence dimension
            cmem = self.c_compressor(context_emb)                        # (B,S,M,D) or (B,2S,M,D) if context_emb present
            cmem_seq = cmem.view(B, S * self.mem_slots, -1)  # (B,S·M,D)

        # Compress per-set genes to memory tokens
        # Note: Using RoPE-encoded features for compression to maintain positional awareness
        xmem = self.x_compressor(x)                        # (B,S,M,D)
        if context_emb is not None:
            # Apply memory RoPE along slot dimension (M) before flattening
            if getattr(self, 'use_memory_rope', False):
                slot_positions = torch.arange(self.mem_slots, device=tokens.device)
                xmem = self.memory_rope(xmem, slot_positions)
                cmem = self.memory_rope(cmem, slot_positions)
            # Flatten to sequences for attention over slots across sets
            xmem_seq = xmem.view(B, S * self.mem_slots, -1)  # (B,S·M,D)
            cmem_seq = cmem.view(B, S * self.mem_slots, -1)  # (B,S·M,D)
            # Segment/type embeddings to disambiguate sources
            xmem_seq = xmem_seq + self.segment_emb.weight[0]
            cmem_seq = cmem_seq + self.segment_emb.weight[1]
            kv_for_blocks = self.kv_ln(torch.cat([xmem_seq, cmem_seq], dim=1))  # include self + control as K/V
        else:
            if getattr(self, 'use_memory_rope', False):
                slot_positions = torch.arange(self.mem_slots, device=tokens.device)
                xmem = self.memory_rope(xmem, slot_positions)
            xmem_seq = xmem.view(B, S * self.mem_slots, -1)
            xmem_seq = xmem_seq + self.segment_emb.weight[0]
            # Feed explicit normalized K/V to stabilise attention scales even without context
            kv_for_blocks = self.kv_ln(xmem_seq)

        lat = xmem_seq
        use_ckpt = self.config.grad_ckpt and self.training and torch.is_grad_enabled()
        for _, block in enumerate(self.blocks):
            if use_ckpt:
                lat = torch.utils.checkpoint.checkpoint(lambda _lat,_kv_for_blocks : block(_lat, _kv_for_blocks), lat, kv_for_blocks, use_reentrant=True)
            else:
                lat = block(lat, kv=kv_for_blocks)

        mem_kv = lat.view(B, S, self.mem_slots, -1)  # (B,S,M,D)
        # Decode from genes to memory using FlashAttention-based module
        # Use the RoPE-encoded features for query in decode attention
        ctx = self.decode_attn(gene_q=x, mem_kv=mem_kv)  # (B,S,N,D)
        h = self.ln_f(ctx)
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


class AbsorbingMaskMD4Continuous(nn.Module):
    """
    Continuous-time per-token absorbing-mask diffusion (MD4-style)
    with auxiliary DE losses ported from PartialMaskingDiffusion.
    """

    def __init__(self, config: ConditionalModelConfig):
        super().__init__()
        self.config = config
        self.K = int(config.vocab_size)
        self.mask_token_id = self.K
        self.eps = 1e-4
        self.t1 = 1e-2  # float(getattr(config, "t1", 1e-2))
        self.antithetic = bool(getattr(config, "antithetic_time_sampling", True))
        # Per-token polynomial exponent w_v (fixed schedule by default; non-trainable)
        self.power_const = torch.ones(self.K) * float(getattr(config, "power_init", 1.0))

    # ---- Schedule and derivatives ----
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        # Scalar α(t). If you want per-class exponents w_v, aggregate to scalar here.
        w = F.softplus(self.power_const).mean()  # or just a single learnable scalar
        return 1.0 - (1.0 - self.eps) * t.clamp_min(1e-8) ** w

    def dgamma_times_alpha(self, t: torch.Tensor) -> torch.Tensor:
        t_exp = t.unsqueeze(-1)
        w = F.softplus(self.power_const).to(t.device)
        return w / t_exp

    # ---- Forward process ----
    def q_sample(self, x0: torch.LongTensor, t: torch.Tensor) -> torch.LongTensor:
        B = x0.shape[0]
        alpha_t = self.alpha(t).view(B, *([1] * (x0.ndim - 1)))  # broadcast to x0 shape
        keep_prob = alpha_t.expand_as(x0)  # same keep prob for all tokens
        survive = torch.rand_like(keep_prob, dtype=keep_prob.dtype) < keep_prob
        return torch.where(survive, x0, torch.full_like(x0, self.mask_token_id)) # replaced self.mask_token_id

    # ---- Helpers (ported) ----
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
                if vb.sum() == 0:
                    continue
                k = max(1, int(vb.sum().item() * signif_arg))
                vals = abs_delta[b][vb]
                thr = vals.topk(k).values[-1]
                de_sig[b] = (abs_delta[b] >= thr) & vb
        else:
            de_sig = (abs_delta >= signif_arg) & valid_mask
        de_dir = (delta > 0).long()
        rank_score = abs_delta
        return de_sig, de_dir, rank_score, valid_mask

    def _bce_with_logits_masked(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: Optional[float] = None):
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
        loss = F.cross_entropy(logits.view(-1, 2), target.contiguous().view(-1), reduction="none")
        loss = loss.view_as(mask)
        denom = mask.float().sum().clamp_min(1.0)
        loss = (loss * mask.float()).sum() / denom
        return loss

    def _ranknet_loss(self, scores, targets, mask, num_pairs=1024, margin=0.0, top_percent=0.1):
        s = scores.mean(dim=1)  # (B,N)
        t = targets.mean(dim=1)  # (B,N)
        m = mask.any(dim=1)  # (B,N)
        B, N = s.shape
        eps = 0.05 * self.config.token_max_value
        total = 0.0
        denom = 0
        for b in range(B):
            valid = m[b]
            tb, sb = t[b], s[b]
            top_mask = valid & (tb >= eps)
            bot_mask = valid & (tb < eps)
            if top_mask.sum() == 0 or bot_mask.sum() == 0:
                continue
            k_top = max(1, int(top_mask.sum().item() * top_percent))
            k_bot = max(1, int(bot_mask.sum().item() * top_percent))
            top_idx = torch.where(top_mask)[0][tb[top_mask].topk(k_top).indices]
            bot_idx = torch.where(bot_mask)[0][(-tb[bot_mask]).topk(k_bot).indices]
            n = min(num_pairs, top_idx.numel(), bot_idx.numel())
            if n == 0:
                continue
            ti = top_idx[torch.randint(0, top_idx.numel(), (n,), device=sb.device)]
            tj = bot_idx[torch.randint(0, bot_idx.numel(), (n,), device=sb.device)]
            total += (-F.logsigmoid(sb[ti] - sb[tj] - margin)).mean()
            denom += 1
        return (total / denom) if denom > 0 else s.new_zeros(())

    def _masked_ce_loss(self, x_t, x0, logits, t):
        # Ensure we have logits over data classes only (exclude mask category)
        log_probs = F.log_softmax(logits[..., : self.K], dim=-1)
        one_hot_x0 = F.one_hot(x0.clamp_max(self.K-1), num_classes=self.K).float()
        neg_ce = -(one_hot_x0 * log_probs).sum(dim=-1)  # [B,S,N]
        mask = (x_t == self.mask_token_id).float() # replaced mask_token_id
        weight = self.dgamma_times_alpha(t).mean(dim=-1).view(x_t.shape[0], 1, 1)
        loss = (mask * neg_ce * weight).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        unweighted_loss = (mask * neg_ce).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        return loss.mean(), unweighted_loss.mean()

    def _masked_ce_soft_loss(self, x_t, logits, t, soft_targets):
        """
        x_t:          (B,S,N) masked input tokens at step t
        logits:       (B,S,N,K+1) model output logits (data bins + mask bin)
        t:            (B,) time steps
        soft_targets: (B,N,K) per gene soft target dist (sum=1 over K data bins)
        """
        # 1. Take log-softmax over data bins only (exclude mask bin)
        log_probs = F.log_softmax(logits[..., : self.K], dim=-1)  # (B,S,N,K)
        
        # Get probabilities for KL divergence computation
        probs = F.softmax(logits[..., : self.K], dim=-1)  # (B,S,N,K)

        # 2. Broadcast soft_targets across S
        soft_targets_b = soft_targets.unsqueeze(1).expand_as(log_probs)  # (B,S,N,K)

        # 3. Negative cross-entropy with soft targets
        neg_ce = -(soft_targets_b * log_probs).sum(dim=-1)  # (B,S,N)

        # 4. Mask out positions that aren't masked in x_t
        mask = (x_t == self.mask_token_id).float()  # (B,S,N)
        
        # Compute KL divergence: KL(soft_targets || predicted_probs)
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        epsilon = 1e-10  # Small value to avoid log(0)
        soft_targets_log = torch.log(soft_targets_b + epsilon)
        kl_div = (soft_targets_b * (soft_targets_log - log_probs)).sum(dim=-1)  # (B,S,N)
        
        # Average KL divergence over masked positions
        masked_kl = (mask * kl_div).sum() / mask.sum().clamp_min(1.0)
        
        # Define effective_bins function
        def effective_bins(soft_targets):  # (B,N,K)
            p = soft_targets.clamp_min(1e-12)
            H = -(p * p.log()).sum(dim=-1)      # (B,N)
            eff = torch.exp(H)                  # exp(entropy)
            return eff.median().item(), eff.mean().item()
        
        # Print KL divergence statistics
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter % 100 == 0:  # Print every 10 iterations
            # Compute effective bins for soft targets (sharpened)
            eff_median, eff_mean = effective_bins(soft_targets)
            
            # Compute effective bins for predicted probabilities (only for masked positions)
            # Average probs across S dimension for masked positions  
            mask_expanded = mask.unsqueeze(-1).expand_as(probs)  # (B,S,N,K)
            masked_probs = probs * mask_expanded  # Zero out non-masked positions
            # Average over S dimension for masked positions
            avg_probs = masked_probs.mean(dim=1)  # (B,N,K)
            pred_eff_median, pred_eff_mean = effective_bins(avg_probs)
            
            print(f"[Step {self._log_counter}] KL divergence (masked tokens): {masked_kl.item():.4f}")
            print(f"  - Mean KL per masked position: {masked_kl.item():.4f}")
            print(f"  - Max KL per position: {(mask * kl_div).max().item():.4f}")
            print(f"  - Min KL per position (non-zero): {kl_div[mask > 0].min().item() if (mask > 0).any() else 0:.4f}")
            print(f"  - Num masked tokens: {mask.sum().item():.0f}")
            print(f"  - Effective bins (sharpened targets γ=3.0) - Median: {eff_median:.2f}, Mean: {eff_mean:.2f}")
            print(f"  - Effective bins (predictions) - Median: {pred_eff_median:.2f}, Mean: {pred_eff_mean:.2f}")

        # 5. Weight schedule (same as before)
        weight = self.dgamma_times_alpha(t).mean(dim=-1).view(x_t.shape[0], 1, 1)

        # 6. Weighted average loss over masked positions
        loss = (mask * neg_ce * weight).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

        # 7. Unweighted version for logging
        unweighted_loss = (mask * neg_ce).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

        return loss.mean(), unweighted_loss.mean()

    # ---- Training loss ----
    def compute_loss(
        self,
        model: nn.Module,
        x0: torch.LongTensor,
        control_set: Optional[torch.LongTensor] = None,
        target_gene_idx: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        step: int = 0,
        max_steps: int = 1000,
        lambda_de: float = 0.5,
        lambda_dir: float = 0.2,
        lambda_rank: float = 0.2,
        de_pos_weight: Optional[float] = 2.0,
        signif_mode: str = "percentile",
        signif_arg: float = 0.1,
        delta_means: Optional[torch.FloatTensor] = None,
        soft_targets: Optional[torch.FloatTensor] = None,
        return_aux: bool = True,
    ) -> torch.Tensor:
        """Continuous-time MD4 objective with optional auxiliary DE losses."""
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

        # Single z_t sample
        zt = self.q_sample(x0, t)
        out = model(
            zt,
            t,
            target_gene_idx=target_gene_idx,
            batch_idx=batch_idx,
            control_context=control_set,
            return_aux=return_aux,
        )
        logits = out["logits"] if isinstance(out, dict) else out

        #main_loss, raw_main_loss = self._masked_ce_loss(zt, x0, logits, t)
        main_loss, raw_main_loss = self._masked_ce_soft_loss(zt, logits, t, soft_targets)

        aux_loss = logits.new_zeros(())
        aux_sig = logits.new_zeros(())
        aux_dir = logits.new_zeros(())
        aux_rank = logits.new_zeros(())

        def ramp(cur_step, frac=0.2):
            return min(1.0, cur_step / max(1, int(max_steps * frac)))

        scale = ramp(step)
        if return_aux and (control_set is not None) and isinstance(out, dict) and (delta_means is not None):
            # Build DE targets using per-gene mask derived from zt
            mlm_mask = (zt == self.mask_token_id)  # replaced mask_token_id (B,S,N) boolean
            mlm_mask_gene = mlm_mask.any(dim=1)    # (B,N)
            de_sig, de_dir, rank_score, valid = self.build_de_targets(
                delta_means,
                signif_mode=signif_mode,
                signif_arg=signif_arg,
                mlm_mask=mlm_mask_gene,
            )

            # Expand to (B,S,N)
            B2, S2, N2, _ = logits.shape
            de_sig = de_sig.unsqueeze(1).expand(-1, S2, -1)
            de_dir_mask = de_sig
            de_dir = de_dir.unsqueeze(1).expand(-1, S2, -1)
            rank_score = rank_score.unsqueeze(1).expand(-1, S2, -1)
            valid = valid.unsqueeze(1).expand(-1, S2, -1)

            # Restrict aux to masked genes at each set position
            valid = valid & mlm_mask
            de_dir_mask = valid & de_dir_mask

            # Compute individual aux losses
            aux_sig = self._bce_with_logits_masked(out["de_sig_logit"], de_sig, valid, pos_weight=de_pos_weight) * lambda_de * scale
            aux_loss = aux_loss + aux_sig

            if de_dir_mask.any():
                aux_dir = self._ce_masked(out["de_dir_logits"], de_dir, de_dir_mask) * lambda_dir * scale
                aux_loss = aux_loss + aux_dir

            aux_rank = self._ranknet_loss(out["de_rank_score"], rank_score, valid, num_pairs=512) * lambda_rank * scale
            aux_loss = aux_loss + aux_rank

        total_loss = main_loss + aux_loss

        # Stash detailed loss breakdown for logging
        try:
            self._last_loss_stats = {
                "loss_main": float(main_loss.detach().item()),
                "loss_main_unweighted": float(raw_main_loss.detach().item()),
                "loss_aux": float(aux_loss.detach().item()),
                "loss_aux_de": float(aux_sig.detach().item()),
                "loss_aux_dir": float(aux_dir.detach().item()),
                "loss_aux_rank": float(aux_rank.detach().item()),
                "loss_total": float(total_loss.detach().item()),
            }
        except Exception:
            pass

        return total_loss, raw_main_loss



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
        B = shape[0]
        device = next(model.parameters()).device
        x_t = torch.full(shape, self.mask_token_id, device=device, dtype=torch.long)

        for i in range(self.config.n_timesteps - 1):
            # fraction from high→low
            t_frac = (self.config.n_timesteps - i) / self.config.n_timesteps
            s_frac = t_frac - 1.0 / self.config.n_timesteps

            t_val = math.cos(math.pi / 2.0 * (1.0 - t_frac))
            s_val = math.cos(math.pi / 2.0 * (1.0 - s_frac))

            t_batch = torch.full((B,), t_val, device=device)
            s_batch = torch.full((B,), s_val, device=device)

            # α(t), α(s) are computed from these
            alpha_t = self.alpha(t_batch)
            alpha_s = self.alpha(s_batch)

            logits = model(
                x_t,
                t_batch,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx,
                control_context=control_set,
                return_aux=False,
            )
            if isinstance(logits, dict):
                logits = logits.get("logits", logits)
            probs_data = F.softmax(logits[..., : self.K], dim=-1)  # (B,S,N,K)

            # MD4: unmask_prob = (α_s - α_t) / (1 - α_t)
            unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t).clamp_min(1e-8)  # (B,)
            # reshape for broadcasting over (S,N)
            unmask_prob = unmask_prob.view(B, *([1] * (x_t.ndim - 1)))

            is_mask = (x_t == self.K)
            if is_mask.any():
                # sample unmask flags only on masked positions
                rand = torch.rand_like(is_mask, dtype=probs_data.dtype)
                unmask = (rand < unmask_prob).to(is_mask.dtype) & is_mask

                if unmask.any():
                    # sample tokens for the to-be-unmasked positions
                    x_new = torch.multinomial(
                        probs_data.view(-1, self.K), 1
                    ).view_as(x_t)
                    x_t = torch.where(unmask.bool(), x_new, x_t)

        # FINAL DECODE STEP (t=0): fill any remaining masks
        remaining = (x_t == self.mask_token_id)
        if remaining.any():
            t0 = torch.zeros(B, device=device)
            logits0 = model(
                x_t,
                t0,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx,
                control_context=control_set,
                return_aux=False,
            )
            if isinstance(logits0, dict):
                logits0 = logits0.get("logits", logits0)
            probs0 = F.softmax(logits0[..., : self.K], dim=-1)
            x_fill = torch.multinomial(probs0.view(-1, self.K), 1).view_as(x_t)
            x_t = torch.where(remaining, x_fill, x_t)

        return x_t

def create_muon_optimizer(model: nn.Module, config: ConditionalModelConfig, use_muon=True):
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

    print(f"use_muon: {use_muon}")

    param_groups = [
        dict(params=hidden_weights, use_muon=use_muon,
             lr=config.muon_lr if use_muon else config.adam_lr, 
             weight_decay=0.01,
             **({"betas": (0.9, 0.95)} if not use_muon else {})),
        # no decay on norms/embeds
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
             lr=config.adam_lr, betas=(0.9, 0.95), weight_decay=0.00),
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