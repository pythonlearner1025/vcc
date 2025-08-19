"""engine.agc
Simple Adaptive Gradient Clipping (AGC).

Implements the algorithm described in
"Normalized Training of Deep Nets" (Brock et al., 2021).

The core idea is to limit the ratio between gradient norm and parameter
norm. For each parameter tensor \(\theta\) with gradient \(g\) we enforce

    ||g||_2 \leq clip_factor * (||\theta||_2 + eps)

When the constraint is violated the gradient is rescaled in-place.

Example
-------
>>> from engine.agc import adaptive_clip_grad
>>> adaptive_clip_grad(model.parameters(), clip_factor=0.01)
"""
from __future__ import annotations

from typing import Iterable

import torch


def adaptive_clip_grad(
    parameters: Iterable[torch.nn.Parameter],
    *,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
) -> tuple[float, float]:
    """Applies Adaptive Gradient Clipping (AGC) in-place.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        Model parameters to process. Typically ``model.parameters()``.
    clip_factor : float, default 0.01
        Maximum allowed ratio between \|g\| and \|\theta\|. Brock et al. use
        0.01 which works well in practice.
    eps : float, default 1e-3
        Small constant for numerical stability and to protect parameters
        with near-zero norms (e.g. biases).
    
    Returns
    -------
    tuple[float, float]
        A tuple of (total_grad_norm_before, total_grad_norm_after) clipping.
    """
    before = 0
    after = 0
    for p in parameters:
        if p.grad is None:
            continue
        # Detach to avoid tracking in autograd graph; keep dtype/device.
        param_norm = torch.norm(p.detach())
        grad_norm = torch.norm(p.grad.detach())
        before += grad_norm.item()

        # Compute max allowed grad norm for this parameter.
        max_norm = (param_norm + eps) * clip_factor
        if grad_norm > max_norm:
            # Rescale gradient in-place.
            clip_coef = max_norm / (grad_norm + eps)
            p.grad.mul_(clip_coef)
            grad_norm = grad_norm.item() * clip_coef
        after += grad_norm.item()
    
    return before, after 