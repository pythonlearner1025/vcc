#!/usr/bin/env python3
"""
Variational Auto-Encoder with frozen ESM gene embeddings and vector-quantised latent
tokens (VQ-VAE style).

The module contains three public classes:

1. LatentTokenizer – splits the latent vector into fixed-size patches and converts
   them to discrete codebook indices (token ids).  A matching `detokenise` method
   re-constructs the latent representation from tokens.
2. VAE – end-to-end model implementing the architecture sketched in the engineering
   note.  Exported helpers: `encode`, `reparameterise`, `tokenise`, `detokenise`,
   `decode`, `kl_loss`, and `recon_loss`.
3. load_esm_embeddings – convenience for loading a gene-ordered frozen ESM matrix
   from the same pickled format already used by `AugmentedGeneEmbedding` (see
   `models/diffusion.py`).

Only the gate MLP, encoder, latent tokenizer & decoder weights are *trainable* –
all ESM tables remain frozen.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility – ESM table loader (shared with diffusion AugmentedGeneEmbedding)
# ---------------------------------------------------------------------------

def load_esm_embeddings(path: str | Path, n_genes: int) -> torch.Tensor:
    """Load the mean-pooled per-gene ESM embedding matrix used across the repo.

    Expects the checkpoint file produced by the preprocessing script which stores
    a dictionary with keys:
        • "emb" – Tensor [n_isoforms, D]
        • "genes" – list[str]  isoform-level gene ids (same order as emb rows)
        • "gene_to_isoform_indices" – dict[str, list[int]]

    The helper mean-pools isoforms to one vector per gene and returns a tensor
    of shape *(n_genes, D)* sorted by gene id (index == gene id).
    """
    path = Path(path)
    obj = torch.load(path, map_location="cpu")

    seq_dim = obj["emb"].shape[-1]
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
    return table  # (G, D)


# ---------------------------------------------------------------------------
# 1) LatentTokenizer – vector-quantised latent patches
# ---------------------------------------------------------------------------

class LatentTokenizer(nn.Module):
    """VQ-VAE style nearest-neighbour codebook.

    Given a latent vector *z ∈ ℝ^{B×L}* this module splits it into patches of
    *patch_dim* (default 8) → (B, P, patch_dim) and for each patch selects the
    closest entry in a learnable codebook (``nn.Embedding``).  The returned
    *token ids* share the same `[MASK]` id used by the diffusion model – i.e.
    *mask_id = codebook_size – 1*.
    """

    def __init__(self, codebook_size: int = 256, patch_dim: int = 8):
        super().__init__()
        assert (
            codebook_size >= 2
        ), "Need at least 2 entries because the last one is reserved for [MASK]"
        self.patch_dim = patch_dim
        self.codebook_size = codebook_size
        # last entry == mask token (never updated)
        self.codebook = nn.Embedding(codebook_size, patch_dim)
        nn.init.uniform_(self.codebook.weight[:-1], -1.0, 1.0)
        with torch.no_grad():
            self.codebook.weight[-1].zero_()  # [MASK] vector

    # ---- helper -----------------------------------------------------------------
    def _split_patches(self, z: torch.Tensor) -> torch.Tensor:
        B, L = z.shape
        assert (
            L % self.patch_dim == 0
        ), f"latent_dim ({L}) must be divisible by patch_dim ({self.patch_dim})"
        return z.view(B, L // self.patch_dim, self.patch_dim)  # (B, P, patch_dim)

    # ---- public API -------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.LongTensor:
        """Quantise *z* and return token ids → (B, P)."""
        patches = self._split_patches(z)  # (B, P, D)
        B, P, D = patches.shape

        # distances ‖x − e‖² = ‖x‖² + ‖e‖² − 2·x·e
        x_sq = patches.pow(2).sum(-1, keepdim=True)  # (B, P, 1)
        code_sq = self.codebook.weight.pow(2).sum(-1)  # (K,)
        # (B, P, K)
        dist = x_sq + code_sq.unsqueeze(0).unsqueeze(0) - 2 * torch.matmul(
            patches, self.codebook.weight.t()
        )
        tokens = torch.argmin(dist, dim=-1)  # (B, P)
        return tokens

    def detokenise(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert token ids back to patches → (B, P, patch_dim)."""
        return self.codebook(tokens)

    # convenience wrappers --------------------------------------------------------
    def tokenise(self, z: torch.Tensor) -> torch.LongTensor:  # alias
        return self.forward(z)

    def decode_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:  # alias
        return self.detokenise(tokens)


# ---------------------------------------------------------------------------
# 2) VAE – ESM-aware encoder-decoder with VQ latent
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """VAE with frozen per-gene ESM embeddings and vector-quantised latent space."""

    def __init__(
        self,
        n_genes: int,
        esm_emb_path: Optional[str] = None,
        esm_dim: int = 512,
        latent_dim: int = 64,
        codebook_size: int = 256,
        patch_dim: int = 8,
        enc_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # ------------------------------------------------------------------
        # Frozen ESM table (G, E)
        # ------------------------------------------------------------------
        if esm_emb_path:
            esm_table = load_esm_embeddings(esm_emb_path, n_genes)
        else:
            esm_table = torch.zeros(n_genes, esm_dim)
        self.register_buffer("esm_emb", esm_table, persistent=False)  # not trainable
        self.esm_dim = esm_table.shape[1]

        # ------------------------------------------------------------------
        # Per-gene gate  w_g = σ( MLP(e_g) )    → (G,)
        # ------------------------------------------------------------------
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.esm_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Encoder  h = (x ⊙ w) → μ, logσ² ∈ ℝ^{B×latent_dim}
        # ------------------------------------------------------------------
        self.enc_net = nn.Sequential(
            nn.Linear(n_genes, enc_hidden_dim),
            nn.GELU(),
            nn.Linear(enc_hidden_dim, enc_hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(enc_hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(enc_hidden_dim, latent_dim)

        # ------------------------------------------------------------------
        # Latent tokenizer (vector quantisation)
        # ------------------------------------------------------------------
        assert (
            latent_dim % patch_dim == 0
        ), "latent_dim must be divisible by patch_dim for patching"
        self.tokenizer = LatentTokenizer(codebook_size, patch_dim)

        # ------------------------------------------------------------------
        # Decoder
        #   z → proj_z (B, E)  ;  gene_weight = Linear(e_g)  (G, E)
        #   recon = proj_z @ gene_weight^T + b_g
        # ------------------------------------------------------------------
        self.z_to_esm = nn.Linear(latent_dim, self.esm_dim)
        self.readout_proj = nn.Linear(self.esm_dim, self.esm_dim, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))

    # ------------------------------------------------------------------
    # Core VAE steps
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return μ, logσ²."""
        # Per-gene gating
        with torch.enable_grad():  # gate_mlp parameters ARE trainable
            w = self.gate_mlp(self.esm_emb).squeeze(-1)  # (G,)
        h = x * w  # (B, G)
        h = self.enc_net(h)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var

    @staticmethod
    def _reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:  # alias
        return self._reparameterise(mu, log_var)

    # ------------------------------------------------------------------
    # Latent tokenisation helpers
    # ------------------------------------------------------------------
    def tokenise(self, z: torch.Tensor) -> torch.LongTensor:
        return self.tokenizer.tokenise(z)

    def detokenise(self, tokens: torch.LongTensor) -> torch.Tensor:
        patches = self.tokenizer.detokenise(tokens)  # (B, P, D)
        return patches.reshape(tokens.size(0), -1)  # (B, latent_dim)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z_proj = self.z_to_esm(z)  # (B, E)

        # gene_weight: (G, E)
        with torch.no_grad():  # esm_emb frozen, readout_proj is trainable
            gene_weight = self.readout_proj(self.esm_emb)  # (G, E)
        recon = torch.matmul(z_proj, gene_weight.t()) + self.gene_bias  # (B, G)
        return recon

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.exp(log_var) + mu.pow(2) - 1.0 - log_var)

    @staticmethod
    def recon_loss(x: torch.Tensor, recon: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return F.mse_loss(recon, x, reduction=reduction)

    # ------------------------------------------------------------------
    # full forward pass (encode → sample → decode)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
