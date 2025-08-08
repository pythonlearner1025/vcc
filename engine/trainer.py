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
    # Optional: limit number of validation steps per eval
    validation_steps: Optional[int] = None
    # Optional: per-dataset learning rates and warmup
    muon_lr: Optional[float] = None
    adam_lr: Optional[float] = None
    warmup_steps: Optional[int] = None
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
        # Build all dataloaders first to compute global batch mapping
        self._phases: List[Tuple[str, Any, DataLoader]] = []
        for ds_spec in self.cfg.datasets:
            dataset, dataloader = self._build_dataloader(ds_spec)
            self._phases.append((ds_spec.name, dataset, dataloader))
        # Compute global mapping and set n_technical_batches on model config if present
        from engine.batches import build_global_batch_mapping, attach_global_batch_mapping, pad_batches_for_fp8
        self.global_batch_mapping = build_global_batch_mapping(self._phases)
        # Update model config arg if exists
        if self.cfg.model.config_args is not None:
            n_batches = len(self.global_batch_mapping)
            # pad to 16 for FP8 safety; harmless otherwise
            self.cfg.model.config_args["n_technical_batches"] = pad_batches_for_fp8(n_batches)
        # Now build model/optim/loss
        self.model = self._build_model().to(self.device)
        self.optimizer, self.scheduler = self._build_optimizer()
        self.loss_adapter = self._build_loss_adapter()
        # Attach mapping to all dataloaders
        self._phases = [
            (name, ds, attach_global_batch_mapping(name, ds, dl, self.global_batch_mapping))
            for name, ds, dl in self._phases
        ]
        self.global_step = 0
        self.global_epoch = 0

    def _reinit_token_and_head(self) -> None:
        """Reinitialise token embedding and output head weights.

        Used when switching from log-binned tokens (pretrain) to delta tokens (finetune)
        so that token semantics do not leak across phases.
        """
        import torch.nn as nn
        import torch.nn.init as init

        tok = getattr(self.model, "token_emb", None)
        head = getattr(self.model, "head", None)
        if isinstance(tok, nn.Embedding):
            init.normal_(tok.weight, mean=0.0, std=0.02)
        if isinstance(head, nn.Linear):
            init.normal_(head.weight, mean=0.0, std=0.02)
            if head.bias is not None:
                init.zeros_(head.bias)
        print("[trainer] Reinitialised token embedding and output head weights")
        self._log({"model/reinit_token_head": 1, "train/global_step": self.global_step})

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
        OptimObj = import_from_string(self.cfg.optimizer.optimizer_class)
        optimizer = None
        # Case 1: standard torch optimizer class
        if isinstance(OptimObj, type) and issubclass(OptimObj, torch.optim.Optimizer):
            optimizer = OptimObj(self.model.parameters(), **self.cfg.optimizer.optimizer_args)
        else:
            # Case 2: factory function (e.g., models.diffusion:create_muon_optimizer)
            if callable(OptimObj):
                model_cfg = getattr(self.model, "config", None)
                try:
                    optimizer = OptimObj(self.model, model_cfg, **self.cfg.optimizer.optimizer_args)
                except TypeError:
                    optimizer = OptimObj(self.model, model_cfg)
            else:
                raise TypeError("optimizer_class must be a torch.optim.Optimizer subclass or a callable factory")
        # Default: no scheduler here; we will apply per-dataset warmup manually.
        return optimizer, None

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
            # Use prebuilt dataloader with attached mapping
            _, dataloader = next((d for d in self._phases if d[0] == ds_spec.name), (None, None, None))[1:]
            self._train_phase(ds_spec, dataloader)
            # After scrna_pretrain completes, reset token embedding/head for delta-token finetuning
            if ds_spec.name.lower() == "scrna_pretrain":
                self._reinit_token_and_head()

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
        # Only implement validation for VCC phases for now
        phase_name = ds_spec.name.lower()
        if "vcc" not in phase_name:
            return

        # Retrieve the train dataloader for this phase (it may carry val handles)
        dataset, train_loader = next((d for d in self._phases if d[0] == ds_spec.name), (None, None, None))[1:]
        val_loader = getattr(train_loader, "val_dataloader", None)
        val_dataset = getattr(train_loader, "val_dataset", None)
        if val_loader is None or val_dataset is None:
            print("[eval] No validation dataloader available for phase", ds_spec.name)
            return

        # Ensure the same global batch mapping is applied to validation
        from engine.batches import attach_global_batch_mapping
        val_loader = attach_global_batch_mapping(ds_spec.name + "_val", val_dataset, val_loader, self.global_batch_mapping)

        # Lazily construct diffusion object to compute loss directly
        from engine.utils import get_autocast_ctx
        autocast_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16

        # Validation steps: default to one full pass over val loader, but allow override
        validation_steps = getattr(ds_spec, "validation_steps", None)
        steps_done = 0
        val_losses: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                if validation_steps is not None and steps_done >= int(validation_steps):
                    break
                batch_std: Batch = adapt_batch(batch, self.device)
                with get_autocast_ctx(self.model, autocast_dtype):
                    loss = self.loss_adapter(self.model, batch_std, self.global_step)
                val_losses.append(loss.item())
                steps_done += 1

        avg_val = float(np.mean(val_losses) if val_losses else 0.0)
        print(f"[eval] {ds_spec.name}: val_loss={avg_val:.4f} over {steps_done} steps")
        self._log({
            "val/loss": avg_val,
            "val/steps": steps_done,
            "val/epoch": self.global_epoch,
            "phase": ds_spec.name,
        })

        # Optional full evaluation using evaluate._run_validation_merged
        # Triggered by model.config.full_eval if available
        full_eval = bool(getattr(getattr(self.model, "config", object()), "full_eval", False))
        if full_eval:
            try:
                from evaluate import _prepare_gene_mappings, _run_validation_merged
                # Prepare gene mappings based on the dataset's AnnData
                # Attempt to locate paths from model.config; fallback to dataset attributes
                cfg_obj = getattr(self.model, "config", None)
                # Prefer dataloader args for accuracy
                finetune_path = (
                    ds_spec.dataloader_args.get("adata_path")
                    if isinstance(getattr(ds_spec, "dataloader_args", None), dict) else None
                ) or getattr(cfg_obj, "finetune_data_path", None) or getattr(val_dataset, "adata_path", None)
                hvg_path = (
                    ds_spec.dataloader_args.get("hvg_info_path")
                    if isinstance(getattr(ds_spec, "dataloader_args", None), dict) else None
                ) or getattr(cfg_obj, "hvg_info_path", None)
                if finetune_path is None or hvg_path is None:
                    raise RuntimeError("full_eval requires config.finetune_data_path and config.hvg_info_path")
                gene_names, hvg_to_full, hvg_gene_ids = _prepare_gene_mappings(finetune_path, hvg_path)

                # Build simple args namespace for evaluate routine
                class _Args:
                    val_genes_number = 10
                    val_set_size = getattr(cfg_obj, "vcc_set_size", getattr(cfg_obj, "pretrain_batch_size", 16))
                    blacklist_path = getattr(cfg_obj, "blacklist_path", "assets/blacklist.txt")

                # Create output dir inside run_dir
                out_dir = self.run_dir / f"eval_ep{self.global_epoch}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Tokeniser/Detokeniser for delta tokens
                from tokenizer import create_delta_tokenizer
                tokenizer, detok = create_delta_tokenizer(getattr(cfg_obj, "vocab_size", 128), max_abs=getattr(cfg_obj, "token_max_value", 10.82), min_abs=1e-3)

                # Ensure val_dataset has mapping
                setattr(val_dataset, "batch_to_idx", self.global_batch_mapping)

                # Ensure diffusion instance
                diffusion_obj = getattr(self.model, "_generic_trainer_diffusion", None)
                if diffusion_obj is None:
                    try:
                        from models.diffusion import PartialMaskingDiffusion as _PMD
                        diffusion_obj = _PMD(cfg_obj)
                    except Exception:
                        diffusion_obj = None

                _run_validation_merged(
                    cfg_obj,
                    self.model,
                    diffusion_obj,
                    tokenizer,
                    detok,
                    gene_names,
                    hvg_to_full,
                    hvg_gene_ids,
                    _Args,
                    self.device,
                    out_dir,
                    batch_to_idx=self.global_batch_mapping,
                    val_ds=val_dataset,
                    val_dl=val_loader,
                    max_genes=getattr(cfg_obj, "max_eval_genes", 10),
                )
            except Exception as e:
                print(f"[eval] full_eval failed: {e}")

    def _train_phase(self, ds_spec: DatasetSpec, dataloader: DataLoader) -> None:
        device = self.device
        # Apply per-dataset learning rates if provided
        if ds_spec.muon_lr is not None or ds_spec.adam_lr is not None:
            for param_group in self.optimizer.param_groups:
                # param groups are tagged by custom keys from create_muon_optimizer
                if param_group.get("use_muon", False):
                    if ds_spec.muon_lr is not None:
                        param_group["lr"] = ds_spec.muon_lr
                else:
                    if ds_spec.adam_lr is not None:
                        param_group["lr"] = ds_spec.adam_lr
        else:
            self._maybe_update_lr(ds_spec.lr)
        total_steps_per_epoch = len(dataloader)
        max_steps = ds_spec.max_steps_per_epoch or total_steps_per_epoch
        autocast_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16

        for local_epoch in range(ds_spec.epochs):
            epoch_losses: List[float] = []
            self.model.train()
            start_t = time.time()
            total_steps = len(dataloader)
            for step_idx, batch in enumerate(dataloader):
                if step_idx >= max_steps:
                    break
                batch_std: Batch = adapt_batch(batch, device)
                with get_autocast_ctx(self.model, autocast_dtype):
                    loss = self.loss_adapter(self.model, batch_std, self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                # Per-dataset warmup only: scale base LR multipliers during warmup
                if ds_spec.warmup_steps is not None and ds_spec.warmup_steps > 0:
                    # Compute scale in [0,1]
                    scale = min(1.0, (self.global_step + 1) / float(ds_spec.warmup_steps))
                    for pg in self.optimizer.param_groups:
                        base_lr = pg.get("base_lr", pg["lr"]) if pg.get("_base_lr_cached", False) else pg["lr"]
                        # cache original if not cached
                        if not pg.get("_base_lr_cached", False):
                            pg["base_lr"] = base_lr
                            pg["_base_lr_cached"] = True
                        desired = pg["base_lr"] * scale
                        pg["lr"] = desired
                self.global_step += 1
                epoch_losses.append(loss.item())
                if self.global_step % max(1, ds_spec.log_every_steps) == 0:
                    avg = float(np.mean(epoch_losses[-100:] if len(epoch_losses) > 100 else epoch_losses))
                    stats = {
                        "train/loss": loss.item(),
                        "train/loss_avg": avg,
                        "train/epoch": self.global_epoch,
                        "train/global_step": self.global_step,
                    }
                    print(f"[Epoch {local_epoch}][{self.global_step}/{total_steps}] | loss {stats['train/loss']:.3f} avg_loss {stats['train/loss_avg']:.3f}")
                    self._log(stats)
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