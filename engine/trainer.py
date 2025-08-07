from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from engine.utils import Batch, adapt_batch, get_autocast_ctx, save_checkpoint


@dataclass
class DatasetSpec:
    name: str
    dataloader_factory: str
    dataloader_args: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 1
    log_every_steps: int = 100
    eval_every_epochs: int = 1
    save_every_steps: int = 1000
    max_steps_per_epoch: Optional[int] = None
    lr: Optional[float] = None
    hooks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    model_class: str
    model_args: Dict[str, Any] = field(default_factory=dict)
    # Optional: construct a config object and pass as single positional arg
    config_class: Optional[str] = None
    config_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerSpec:
    optimizer_class: str
    optimizer_args: Dict[str, Any] = field(default_factory=dict)
    scheduler_class: Optional[str] = None
    scheduler_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossSpec:
    loss_class: str
    loss_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    run_name: str
    output_dir: str = "checkpoints"
    seed: int = 42
    device: str = "cuda"
    amp_dtype: str = "bf16"
    use_wandb: bool = True
    wandb_project: str = "training"
    model: ModelSpec = field(default_factory=ModelSpec)  # type: ignore
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)  # type: ignore
    loss: LossSpec = field(default_factory=LossSpec)  # type: ignore
    datasets: List[DatasetSpec] = field(default_factory=list)
    resume_from: Optional[str] = None


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._maybe_set_seed(cfg.seed)
        self._build_dirs()
        self._init_wandb()
        self.model = self._build_model().to(self.device)
        self.optimizer, self.scheduler = self._build_optimizer()
        self.loss_adapter = self._build_loss_adapter()
        self.global_step = 0
        self.global_epoch = 0

    def _maybe_set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_dirs(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.cfg.output_dir) / f"{self.cfg.run_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _init_wandb(self) -> None:
        self.wandb = None
        if self.cfg.use_wandb:
            import wandb
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name, config=self._export_config())
            self.wandb = wandb

    def _build_model(self) -> torch.nn.Module:
        from engine.utils import import_from_string
        ModelClass = import_from_string(self.cfg.model.model_class)
        if self.cfg.model.config_class:
            ConfigClass = import_from_string(self.cfg.model.config_class)
            config_obj = ConfigClass(**self.cfg.model.config_args)
            model = ModelClass(config_obj)
        else:
            model = ModelClass(**self.cfg.model.model_args)
        return model

    def _build_optimizer(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        from engine.utils import import_from_string
        OptimClass = import_from_string(self.cfg.optimizer.optimizer_class)
        optimizer = OptimClass(self.model.parameters(), **self.cfg.optimizer.optimizer_args)
        scheduler = None
        if self.cfg.optimizer.scheduler_class:
            SchedClass = import_from_string(self.cfg.optimizer.scheduler_class)
            scheduler = SchedClass(optimizer, **self.cfg.optimizer.scheduler_args)
        return optimizer, scheduler

    def _build_loss_adapter(self) -> Callable[[torch.nn.Module, Batch, int], torch.Tensor]:
        from engine.utils import import_from_string
        LossClass = import_from_string(self.cfg.loss.loss_class)
        loss_obj = LossClass(**self.cfg.loss.loss_args)
        def _adapter(model: torch.nn.Module, batch: Batch, step: int) -> torch.Tensor:
            return loss_obj.compute_loss(model, batch, step)
        return _adapter

    def _export_config(self) -> Dict[str, Any]:
        def as_dict(dc):
            if hasattr(dc, "__dataclass_fields__"):
                out = {}
                for k in dc.__dataclass_fields__.keys():  # type: ignore
                    out[k] = as_dict(getattr(dc, k))
                return out
            if isinstance(dc, list):
                return [as_dict(x) for x in dc]
            return dc
        return as_dict(self.cfg)

    def resume_if_needed(self) -> None:
        if not self.cfg.resume_from:
            return
        ckpt = torch.load(self.cfg.resume_from, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])  # type: ignore
        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
            except Exception as e:
                print(f"[warn] could not restore optimizer: {e}")
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # type: ignore
            except Exception as e:
                print(f"[warn] could not restore scheduler: {e}")
        state = ckpt.get("state", {})
        self.global_step = int(state.get("global_step", 0))
        self.global_epoch = int(state.get("global_epoch", 0))
        print(f"Resumed from {self.cfg.resume_from}: step={self.global_step}, epoch={self.global_epoch}")

    def train(self) -> None:
        self.resume_if_needed()
        for ds_idx, ds_spec in enumerate(self.cfg.datasets):
            print(f"\n==> Phase {ds_idx+1}/{len(self.cfg.datasets)}: {ds_spec.name} ({ds_spec.epochs} epochs)")
            dataset, dataloader = self._build_dataloader(ds_spec)
            self._train_phase(ds_spec, dataloader)

    def _build_dataloader(self, ds_spec: DatasetSpec) -> Tuple[Any, DataLoader]:
        from engine.utils import import_from_string
        factory = import_from_string(ds_spec.dataloader_factory)
        result = factory(**ds_spec.dataloader_args)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], DataLoader):
            return result
        if isinstance(result, DataLoader):
            return None, result
        raise TypeError("Dataloader factory must return DataLoader or (dataset, DataLoader)")

    def _maybe_update_lr(self, lr: Optional[float]) -> None:
        if lr is None:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _log(self, payload: Dict[str, Any]) -> None:
        if self.wandb is not None:
            self.wandb.log(payload)

    def _save(self, step_or_epoch: str) -> None:
        ckpt_path = self.run_dir / f"checkpoint_{step_or_epoch}.pt"
        save_checkpoint(
            str(ckpt_path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self._export_config(),
            state={"global_step": self.global_step, "global_epoch": self.global_epoch},
        )
        print(f"Saved checkpoint: {ckpt_path}")

    def _eval(self, ds_spec: DatasetSpec) -> None:
        pass

    def _train_phase(self, ds_spec: DatasetSpec, dataloader: DataLoader) -> None:
        device = self.device
        self._maybe_update_lr(ds_spec.lr)
        total_steps_per_epoch = len(dataloader)
        max_steps = ds_spec.max_steps_per_epoch or total_steps_per_epoch
        autocast_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16

        for local_epoch in range(ds_spec.epochs):
            epoch_losses: List[float] = []
            self.model.train()
            start_t = time.time()
            for step_idx, batch in enumerate(dataloader):
                if step_idx >= max_steps:
                    break
                batch_std: Batch = adapt_batch(batch, device)
                with get_autocast_ctx(self.model, autocast_dtype):
                    loss = self.loss_adapter(self.model, batch_std, self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                epoch_losses.append(loss.item())
                if self.global_step % max(1, ds_spec.log_every_steps) == 0:
                    avg = float(np.mean(epoch_losses[-100:] if len(epoch_losses) > 100 else epoch_losses))
                    self._log({
                        "train/loss": loss.item(),
                        "train/loss_avg": avg,
                        "train/epoch": self.global_epoch,
                        "train/global_step": self.global_step,
                    })
                if self.global_step % max(1, ds_spec.save_every_steps) == 0:
                    self._save(f"ep{self.global_epoch}_step{self.global_step}")
            self.global_epoch += 1
            epoch_time = time.time() - start_t
            avg_epoch_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)
            print(f"Epoch {self.global_epoch} finished in {epoch_time:.1f}s | loss {avg_epoch_loss:.4f}")
            if (local_epoch + 1) % max(1, ds_spec.eval_every_epochs) == 0:
                self._eval(ds_spec)
            self._save(f"ep{self.global_epoch}")

    def _forward_loss(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError