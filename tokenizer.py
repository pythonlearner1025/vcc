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

class SimpleTokenizer:
    def __init__(self, vocab_size, max_value):
        # Use full data vocabulary [0 .. vocab_size-1] for value tokens.
        # The diffusion model reserves the mask token at id == vocab_size.
        self.vocab_size = vocab_size
        self.max_value = max_value
        self.mask_token = vocab_size  # not used here; mask handled by diffusion
        
        # Define bins for expression values - use log-scale for better distribution
        # Bin 0: exactly 0 (very common in scRNA-seq)
        # Bins 1-(vocab_size-2): log-scale from 0.1 to max_value
        self.bins = torch.zeros(self.vocab_size)
        self.bins[1:] = torch.logspace(
            np.log10(0.1), 
            np.log10(max_value), 
            self.vocab_size - 1
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
        
        # Guard against occasional mask ids or out-of-range values by clamping
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens)
        tokens_clamped = tokens.clamp(0, self.vocab_size - 1)
        return bin_centers[tokens_clamped]

def create_logbin_tokenizer(vocab_size: int = 64, max_value: float = 9.2):
    tokenizer = SimpleTokenizer(vocab_size, max_value)
    return tokenizer, tokenizer.detokenize

@torch.no_grad()
def fit_delta_bins_quantiles(delta_all: torch.Tensor, K: int, eps: float = 1e-6):
    """
    Fit quantile-based bins for delta tokenization.
    
    delta_all: (T, N) or (T,) all Δ samples across training (flatten ok)
    K: total vocab including mask -> usable bins = K-1
    Returns: bins tensor of length (K-1) for torch.bucketize (symmetrized around 0)
    """
    assert K % 2 == 0, "prefer even K so middle token is neutral"
    V = K - 1                      # usable value bins (exclude mask)
    half = V // 2                  # tokens per side, neutral sits at index `half`

    x = delta_all.abs().reshape(-1)
    x = x[torch.isfinite(x)]
    x = x[x > 0]                   # ignore exact 0 for magnitude quantiles
    
    # Quantile cutpoints for magnitudes (half bins on + side)
    # Use monotone increasing quantiles from small to large magnitudes
    q = torch.linspace(0, 1, steps=half+1, dtype=x.dtype, device=x.device)
    q[0] = eps; q[-1] = 1 - eps    # avoid degenerate edges
    pos_edges = torch.quantile(x, q)[1:]  # length = half

    # Symmetric edges vector of length V (neg side, 0, pos side)
    neg_edges = -pos_edges.flip(0)
    bins = torch.cat([neg_edges, torch.tensor([0.0], device=x.device, dtype=x.dtype), pos_edges])  # len V
    return bins  # pass to tokenizer as `self.bins`

class DeltaTokenizer:
    """
    Tokenizer for Δ with K data bins (no mask inside).
    |Δ| < min_abs -> neutral mid token.
    Supports both logspace and quantile-based binning.
    """
    def __init__(self, value_vocab_size: int = 64, max_abs: float = 9.2, min_abs: float = 1e-3, 
                 delta_all: torch.Tensor = None, use_quantiles: bool = False):
        assert value_vocab_size >= 3 and value_vocab_size % 2 == 1, \
            "Prefer odd K so there is an exact neutral bin."
        self.K = int(value_vocab_size)
        self.half = self.K // 2
        self.min_abs = float(min_abs)

        if use_quantiles and delta_all is not None:
            # Use quantile-based bins fitted to actual data distribution
            # Note: fit_delta_bins_quantiles expects K to include mask token
            # So we pass K+1 since our K is the number of value bins
            self.bins = fit_delta_bins_quantiles(delta_all, K=self.K+1, eps=1e-6)
        else:
            # Use original logspace binning
            pos_edges = torch.logspace(np.log10(min_abs), np.log10(max_abs), steps=self.half, base=10.0)
            neg_edges = -pos_edges.flip(0)
            self.bins = torch.cat([neg_edges, torch.tensor([0.0]), pos_edges])  # len = K

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.device != self.bins.device:
            self.bins = self.bins.to(x.device)
        tokens = torch.bucketize(x, self.bins).clamp(0, self.K - 1)
        neutral = (x.abs() < self.min_abs)
        tokens[neutral] = self.half
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        bins = self.bins if tokens.device == self.bins.device else self.bins.to(tokens.device)
        centres = torch.empty(self.K, device=bins.device)
        centres[:-1] = (bins[:-1] + bins[1:]) / 2
        centres[-1] = bins[-1]
        centres[self.half] = 0.0
        return centres[tokens.clamp(0, self.K - 1)]

def create_delta_tokenizer(vocab_size: int = 64, max_value: float = 9.2, min_abs: float = 1e-3,
                          delta_all: torch.Tensor = None, use_quantiles: bool = False):
    """
    Create a delta tokenizer with either logspace or quantile-based binning.
    
    Args:
        vocab_size: Number of vocabulary bins (excluding mask)
        max_value: Maximum absolute value for logspace binning (ignored if use_quantiles=True)
        min_abs: Minimum absolute value threshold for neutral bin
        delta_all: Training delta values for fitting quantile bins (required if use_quantiles=True)
        use_quantiles: Whether to use quantile-based binning instead of logspace
    
    Returns:
        Tuple of (tokenizer, detokenize_fn)
    """
    tok = DeltaTokenizer(value_vocab_size=vocab_size, max_abs=max_value, min_abs=min_abs,
                        delta_all=delta_all, use_quantiles=use_quantiles)
    return tok, tok.detokenize