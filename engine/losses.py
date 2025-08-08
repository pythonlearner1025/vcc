from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from engine.utils import Batch


@dataclass
class DiffusionLoss:
    # Dotted path to diffusion class; defaults to models.diffusion:PartialMaskingDiffusion
    diffusion_class: str = "models.diffusion:PartialMaskingDiffusion"
    diffusion_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from engine.utils import import_from_string
        DiffClass = import_from_string(self.diffusion_class)
        # The diffusion constructor typically expects a config stored on the model
        self._create = DiffClass

    def compute_loss(self, model: torch.nn.Module, batch: Batch, step: int) -> torch.Tensor:
        # Build diffusion object lazily and cache on the model for reuse
        if not hasattr(model, "_generic_trainer_diffusion"):
            diffusion = self._create(getattr(model, "config", None), **self.diffusion_args)
            setattr(model, "_generic_trainer_diffusion", diffusion)
        diffusion = getattr(model, "_generic_trainer_diffusion")
        return diffusion.compute_loss(
            model,
            batch.tokens,
            control_set=batch.control,
            target_gene_idx=batch.target_gene_idx,
            batch_idx=batch.batch_idx,
            step=step,
        )