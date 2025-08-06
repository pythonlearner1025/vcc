from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset

class TokenizedScRNADataset(Dataset):
    """Wrapper around ScRNADatasetWithHVGs that applies tokenization."""
    
    def __init__(self, scrna_dataset: ScRNADatasetWithHVGs, tokenizer):
        self.dataset = scrna_dataset
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]  # Already returns tensor with HVG genes
        # Apply tokenizer to convert continuous expression to discrete tokens
        tokens = self.tokenizer(x)
        return tokens

def create_logbin_tokenizer(vocab_size: int = 64, max_value: float = 9.2):
    class SimpleTokenizer:
        def __init__(self, vocab_size, max_value):
            self.vocab_size = vocab_size - 1  # Reserve last token for [MASK]
            self.max_value = max_value
            self.mask_token = vocab_size - 1
            
            # Define bins for expression values - use log-scale for better distribution
            # Bin 0: exactly 0 (very common in scRNA-seq)
            # Bins 1-(vocab_size-2): log-scale from 0.1 to max_value
            self.bins = torch.zeros(self.vocab_size)
            self.bins[1:] = torch.logspace(
                np.log10(0.1), 
                np.log10(max_value), 
                self.vocab_size - 1 # -2 because its inclusive
            )

            self.normalize_checked = False 
            
        def __call__(self, x):
            """Tokenize expression values into discrete bins."""
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            
            # Efficient tensor-aware validation
            if not self.normalize_checked: 
                if not torch.all((x >= 0) & (x < self.max_value)):
                    invalid_mask = (x < 0) | (x >= self.max_value)
                    invalid_values = x[invalid_mask]
                    raise ValueError(f"Input values {invalid_values[:5].tolist()}... are out of the expected range [0, {self.max_value}). "
                                f"Found {invalid_mask.sum().item()} invalid values out of {x.numel()} total.")
                else: self.normalize_checked = True
            
            # Ensure bins are on the same device as input
            if x.device != self.bins.device:
                self.bins = self.bins.to(x.device)
            
            # Handle zero values explicitly
            zero_mask = (x == 0)
            
            # Clip values to max range
            x_clipped = torch.clamp(x, 0, self.max_value)
            
            # Bucketize into bins
            tokens = torch.bucketize(x_clipped, self.bins)
            
            # Ensure zero values map to token 0
            tokens[zero_mask] = 0
            
            return tokens.clamp(0, self.vocab_size - 1)
        
        def detokenize(self, tokens):
            """Convert tokens back to approximate expression values."""
            # Use bin centers for reconstruction
            bin_centers = torch.zeros(self.vocab_size)
            bin_centers[0] = 0.0  # Zero bin
            
            for i in range(1, self.vocab_size - 1):
                bin_centers[i] = (self.bins[i] + self.bins[i+1]) / 2
            bin_centers[-1] = self.bins[-1]  # Last bin uses upper bound
            
            return bin_centers[tokens]
    
    tokenizer = SimpleTokenizer(vocab_size, max_value)
    return tokenizer, tokenizer.detokenize

class DeltaTokenizer:
    """Symmetric tokenizer for perturbation Δ values.

    The vocabulary is split evenly between down-regulation (negative Δ) and up-regulation (positive Δ).
    The final token id (``vocab_size-1``) is reserved for the `[MASK]` token used by the diffusion model.
    """

    def __init__(self, vocab_size: int = 256, max_abs: float = 9.2, min_abs: float = 1e-3):
        """Args
        -----
        vocab_size:
            Total vocabulary size **including** the mask token. Must be even so that the neutral
            token sits exactly in the centre.
        max_abs:
            Largest absolute Δ (in log1p-space) to represent explicitly (≈log1p(10 000)=9.21).
            Any value with |Δ| > ``max_abs`` will be clipped to the most extreme non-mask token.
        min_abs:
            Smallest non-zero |Δ| distinguishable by the tokenizer. Values |Δ| < ``min_abs`` are
            assigned to the neutral bin (token ``half``).
        """
        # ------------------------------------------------------------------
        # Sanity checks & bookkeeping
        # ------------------------------------------------------------------
        assert vocab_size % 2 == 0, "Prefer even vocab so that the mid-token encodes Δ≈0."

        # Reserve last id for [MASK]
        self.mask_token: int = vocab_size - 1
        self._value_vocab_size: int = vocab_size - 1  # usable ids 0‥vocab_size-2

        self._half: int = self._value_vocab_size // 2  # token index representing Δ≈0

        # ------------------------------------------------------------------
        # Build symmetric bin edges around 0
        # ------------------------------------------------------------------
        # Positive bin edges (> 0), distributed logarithmically in Δ-space
        pos_edges = torch.logspace(
            np.log10(min_abs),
            np.log10(max_abs),
            steps=self._half,
            base=10.0,
        )
        neg_edges = -pos_edges.flip(0)

        # 0 sits exactly in the middle so that token `half` encodes Δ≈0
        self.bins = torch.cat([neg_edges, torch.tensor([0.0]), pos_edges])  # len == _value_vocab_size

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Vectorised tokenisation.

        Any value outside the representable range will be saturated to the most extreme
        non-mask token, never the `[MASK]` token.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # Keep `bins` on the same device as input for speed
        if x.device != self.bins.device:
            self.bins = self.bins.to(x.device)

        tokens = torch.bucketize(x, self.bins)
        return tokens.clamp(0, self._value_vocab_size - 1)  # ‑- reserve final id for [MASK]

    # ------------------------------------------------------------------
    # Helper for inverse transform (approximate)
    # ------------------------------------------------------------------
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map discrete tokens back to the centre of their bins (approximate Δ)."""
        if tokens.device != self.bins.device:
            bins = self.bins.to(tokens.device)
        else:
            bins = self.bins

        # Bin centres – last token shares the last bin edge (saturating)
        centres = torch.zeros(self._value_vocab_size, device=bins.device)
        # Default: mid-bin centres
        centres[:-1] = (bins[:-1] + bins[1:]) / 2
        centres[-1] = bins[-1]
        # Explicitly enforce neutral token to 0
        centres[self._half] = 0.0

        tokens_clamped = tokens.clamp(0, self._value_vocab_size - 1)
        return centres[tokens_clamped]

def create_delta_tokenizer(vocab_size: int = 256, max_abs: float = 9.2, min_abs: float = 1e-3):
    """Factory matching the signature of ``create_logbin_tokenizer`` used elsewhere."""
    tok = DeltaTokenizer(vocab_size=vocab_size, max_abs=max_abs, min_abs=min_abs)
    return tok, tok.detokenize