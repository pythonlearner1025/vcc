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
    max_norm: float | None = None,
) -> tuple[float, float]:
    before = 0.0
    after = 0.0
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return before, after

    if max_norm is not None:
        # Global clipping path
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in params]))
        before = total_norm.item()
        if total_norm > max_norm:
            scale = max_norm / (total_norm + eps)
            for p in params:
                p.grad.mul_(scale)
        after = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in params])).item()
        return before, after

    # AGC path
    for p in params:
        param_norm = torch.norm(p.detach())
        grad_norm = torch.norm(p.grad.detach())
        before += grad_norm.item()

        local_max = (param_norm + eps) * clip_factor
        if grad_norm > local_max:
            scale = local_max / (grad_norm + eps)
            p.grad.mul_(scale)
            grad_norm = grad_norm.item() * scale
        after += grad_norm.item()

    return before, after
