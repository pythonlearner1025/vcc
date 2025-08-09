from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from engine.utils import Batch


@dataclass
class DiffusionLoss:
    # Dotted path to diffusion class; defaults to models.diffusion:PartialMaskingDiffusion
    diffusion_class: str = "models.diffusion:PartialMaskingDiffusion"
    diffusion_args: Dict[str, Any] = field(default_factory=dict)
    # Aux loss knobs (can be overridden via YAML loss.loss_args)
    lambda_de: float = 0.5
    lambda_dir: float = 0.2
    lambda_rank: float = 0.2
    de_pos_weight: float = 2.0
    signif_mode: str = "percentile"
    signif_arg: float = 0.1

    def __post_init__(self):
        from engine.utils import import_from_string
        DiffClass = import_from_string(self.diffusion_class)
        # The diffusion constructor typically expects a config stored on the model
        self._create = DiffClass

    def compute_loss(self, model: torch.nn.Module, batch: Batch, phase_step: int, total_phase_steps: int) -> torch.Tensor:
        # Build diffusion object lazily and cache on the model for reuse
        if not hasattr(model, "_generic_trainer_diffusion"):
            diffusion = self._create(getattr(model, "config", None), **self.diffusion_args)
            setattr(model, "_generic_trainer_diffusion", diffusion)
        diffusion = getattr(model, "_generic_trainer_diffusion")
        loss = diffusion.compute_loss(
            model,
            batch.tokens,
            control_set=batch.control,
            target_gene_idx=batch.target_gene_idx,
            batch_idx=batch.batch_idx,
            delta_means=batch.delta_means,
            step=phase_step,
            max_steps=total_phase_steps,
            lambda_de=self.lambda_de,
            lambda_dir=self.lambda_dir,
            lambda_rank=self.lambda_rank,
            de_pos_weight=self.de_pos_weight,
            signif_mode=self.signif_mode,
            signif_arg=self.signif_arg,
        )
        # Expose detailed stats to trainer via attribute on model to avoid changing interfaces
        try:
            stats = getattr(diffusion, "_last_loss_stats", None)
            if stats is not None:
                setattr(model, "_last_loss_stats", stats)
        except Exception:
            pass
        return loss