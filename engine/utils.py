import importlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

# Optional Transformer Engine (FP8)
USE_TE = int(os.getenv("TE", 0))
HAS_TE = False
try:
    if USE_TE:
        import transformer_engine.pytorch as te  # type: ignore
        from transformer_engine.pytorch import fp8_autocast  # type: ignore
        HAS_TE = True
    else:
        raise ImportError
except Exception:
    def fp8_autocast(enabled: bool = True):  # type: ignore
        return nullcontext()
    HAS_TE = False


def import_from_string(path: str) -> Any:
    """Import an object from a dotted path string.

    Supports both "module.submodule:attr" and "module.submodule.attr" forms.
    """
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        parts = path.split(".")
        module_name, attr_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_autocast_ctx(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16):
    """Return the appropriate autocast context manager.

    - If Transformer Engine is available and the model config requests FP8, use fp8_autocast.
    - Otherwise use PyTorch AMP with the requested dtype (default BF16).
    """
    use_fp8 = getattr(getattr(model, "config", object()), "use_fp8", False)
    if HAS_TE and use_fp8:
        return fp8_autocast(enabled=True)
    return torch.amp.autocast("cuda", dtype=dtype)


@dataclass
class Batch:
    tokens: torch.Tensor
    control: Optional[torch.Tensor] = None
    target_gene_idx: Optional[torch.Tensor] = None
    batch_idx: Optional[torch.Tensor] = None


def adapt_batch(batch: Any, device: torch.device) -> Batch:
    """Standardize various batch structures to a common Batch dataclass.

    Accepted inputs:
    - torch.Tensor → tokens
    - dict with keys 'tokens' and optional 'control', 'target_gene_idx', 'batch_idx'
    - tuple/list where first element is tokens (discouraged; provided for leniency)
    """
    if isinstance(batch, torch.Tensor):
        tokens = batch.to(device, non_blocking=True)
        return Batch(tokens=tokens)
    if isinstance(batch, dict):
        tokens = batch["tokens"].to(device, non_blocking=True)
        control = batch.get("control")
        if control is not None:
            control = control.to(device, non_blocking=True)
        target_gene_idx = batch.get("target_gene_idx")
        if target_gene_idx is not None:
            target_gene_idx = target_gene_idx.to(device, non_blocking=True)
        batch_idx = batch.get("batch_idx")
        if batch_idx is not None:
            batch_idx = batch_idx.to(device, non_blocking=True)
        return Batch(tokens=tokens, control=control, target_gene_idx=target_gene_idx, batch_idx=batch_idx)
    if isinstance(batch, (tuple, list)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
        tokens = batch[0].to(device, non_blocking=True)
        return Batch(tokens=tokens)
    raise TypeError("Unsupported batch type; provide a tensor or dict with 'tokens'.")


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    config: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "state": state,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)