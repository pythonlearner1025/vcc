from __future__ import annotations

import torch
import numpy as np

class DeltaTokenizer:
    """Symmetric tokenizer for perturbation Δ values.

    The vocabulary is split evenly between down-regulation (negative Δ) and up-regulation (positive Δ).
    The final token id (``vocab_size-1``) is reserved for the `[MASK]` token used by the diffusion model.
    """

    def __init__(self, vocab_size: int = 256, max_abs: float = 5.0):
        """Args
        -----
        vocab_size:
            Total vocabulary size **including** the mask token.  Must be even so that the neutral
            token sits exactly in the centre.
        max_abs:
            Largest absolute Δ (in log10 space) to represent explicitly.  Larger magnitudes will be
            clipped to the most extreme non-mask token.
        """
        assert vocab_size % 2 == 0, "Prefer even vocab so that the mid-token encodes Δ≈0."

        # Reserve last id for [MASK]
        self.mask_token: int = vocab_size - 1
        self._value_vocab_size: int = vocab_size - 1  # usable ids 0‥vocab_size-2

        half: int = self._value_vocab_size // 2

        # Positive/negative bin edges in log-space (exclusive of 0)
        pos_edges = torch.logspace(-max_abs, max_abs, steps=half, base=10.0)
        neg_edges = -pos_edges.flip(0)

        # Concatenate into one monotonic tensor with an explicit 0 in the middle
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
        centres[:-1] = (bins[:-1] + bins[1:]) / 2
        centres[-1] = bins[-1]

        tokens_clamped = tokens.clamp(0, self._value_vocab_size - 1)
        return centres[tokens_clamped]


def create_delta_tokenizer(vocab_size: int = 256, max_abs: float = 5.0):
    """Factory matching the signature of ``create_simple_tokenizer`` used elsewhere."""
    tok = DeltaTokenizer(vocab_size=vocab_size, max_abs=max_abs)
    return tok, tok.detokenize