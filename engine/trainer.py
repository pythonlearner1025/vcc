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
from engine.agc import adaptive_clip_grad


@dataclass
class DatasetSpec:
    name: str
    dataloader_factory: str
    dataloader_args: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 1
    log_every_steps: int = 100
    eval_every_epochs: int = 1
    save_every_steps: int = 1000
    max_steps: Optional[int] = None
    lr: Optional[float] = None
    # Optional: limit number of validation steps per eval
    validation_steps: Optional[int] = None
    # Optional: per-dataset learning rates and warmup
    muon_lr: Optional[float] = None
    adam_lr: Optional[float] = None
    warmup_steps: Optional[int] = None
    # Optional: per-dataset token weighting annealing steps
    token_weighting_annealing_steps: Optional[int] = None
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
    eval_mode: bool = False
    test_prep_mode: bool = False
    test_genes_path: str = "."
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
        print("n_technical_batches: ", self.cfg.model.config_args["n_technical_batches"])
        # Now build model/optim/loss
        self.model = self._build_model().to(self.device, dtype=torch.bfloat16)
        print(f"\nModel parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
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
        def _adapter(model: torch.nn.Module, batch: Batch, phase_step: int, total_phase_steps) -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return loss_obj.compute_loss(model, batch, phase_step, total_phase_steps)
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
        # Tolerant load: use matching tensors from checkpoint; zero-init missing; keep current for mismatched shapes
        self._safe_load_model_state(ckpt.get("model_state_dict", {}))
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

    def _safe_load_model_state(self, src_state: Dict[str, torch.Tensor]) -> None:
        """Load model state dict tolerantly.

        - If a parameter exists in the checkpoint with the same shape, load it.
        - If missing in the checkpoint, zero-initialize the model parameter.
        - If present but with mismatched shape, keep the current model parameter.
        """
        if not isinstance(src_state, dict) or len(src_state) == 0:
            print("[warn] checkpoint has no model_state_dict; skipping model restore")
            return

        dst_state = self.model.state_dict()
        new_state: Dict[str, torch.Tensor] = {}
        missing_keys: List[str] = []
        mismatched_keys: List[str] = []

        for key, dst_tensor in dst_state.items():
            if key in src_state:
                src_tensor = src_state[key]
                if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
                    # Move to same device/dtype as destination tensor
                    new_state[key] = src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype)
                else:
                    mismatched_keys.append(key)
                    new_state[key] = dst_tensor
            else:
                missing_keys.append(key)
                new_state[key] = torch.zeros_like(dst_tensor)

        # Load strictly since we've filled all required keys
        self.model.load_state_dict(new_state, strict=True)

        if missing_keys:
            print(f"[resume] zero-initialized {len(missing_keys)} missing params (e.g., {missing_keys[:6]})")
        if mismatched_keys:
            print(f"[resume] kept current params for {len(mismatched_keys)} shape-mismatched keys (e.g., {mismatched_keys[:6]})")

    def train(self) -> None:
        self.resume_if_needed()
        for ds_idx, ds_spec in enumerate(self.cfg.datasets):
            print(f"\n==> Phase {ds_idx+1}/{len(self.cfg.datasets)}: {ds_spec.name} ({ds_spec.epochs} epochs)")
            # Use prebuilt dataloader with attached mapping
            _, dataloader = next((d for d in self._phases if d[0] == ds_spec.name), (None, None, None))[1:]
            self._train_phase(ds_spec, dataloader)
            # After scrna_pretrain completes, reset token embedding/head for delta-token finetuning
            if ds_spec.name.lower() == "scrna_pretrain" and ds_spec.epochs > 0:
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

        print(f"len val_dataset:{len(val_dataset)}, val_dataloader: {len(val_loader)}")
        total_val_steps = len(val_loader)

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
                    losses = self.loss_adapter(self.model, batch_std, steps_done, total_val_steps)
                loss, raw_loss = losses
                print(loss.item(), raw_loss.item())
                val_losses.append(raw_loss.item())
                steps_done += 1
                print(f"steps_done: {steps_done}")

        avg_val = float(np.mean(val_losses) if val_losses else 0.0)
        print(f"[eval] {ds_spec.name}: val_loss={avg_val:.4f} over {steps_done} steps")
        self._log({
            "val/loss": avg_val,
            "val/steps": steps_done,
            "val/epoch": self.global_epoch,
            "phase": ds_spec.name,
        })

        print(f"len val_dataset:{len(val_dataset)}, val_dataloader: {len(val_loader)}")
        # Optional full evaluation using evaluate._run_validation_merged
        # Triggered by model.config.full_eval if available
        full_eval = bool(getattr(getattr(self.model, "config", object()), "full_eval", False))
        if full_eval:
            try:
                from evaluate import _prepare_gene_mappings, _run_validation_merged, _run_test_generation
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
                    set_size = ds_spec.dataloader_args.get("set_size", getattr(cfg_obj, "pretrain_batch_size", 32))
                    vocab_size = ds_spec.dataloader_args.get("vocab_size", getattr(cfg_obj, "vocab_size", 64))
                    batch_size = ds_spec.dataloader_args.get("batch_size", getattr(cfg_obj, "batch_size", 4))
                    test_genes_path = self.cfg.test_genes_path
                    blacklist_path = getattr(cfg_obj, "blacklist_path", "assets/blacklist.txt")
                
                # cfg datasets
                # 

                # Create output dir inside run_dir
                out_dir = self.run_dir / f"eval_ep{self.global_epoch}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Tokeniser/Detokeniser for delta tokens
                from tokenizer import create_delta_tokenizer
                tokenizer, detok = create_delta_tokenizer(getattr(cfg_obj, "vocab_size", 64), max_value=getattr(cfg_obj, "token_max_value", 10.82))

                # Ensure val_dataset has mapping
                setattr(val_dataset, "batch_to_idx", self.global_batch_mapping)

                # Ensure diffusion instance
                diffusion_obj = getattr(self.model, "_generic_trainer_diffusion", None)
                if diffusion_obj is None:
                    try:
                        from models.diffusion import AbsorbingMaskMD4Continuous as _MD4
                        diffusion_obj = _MD4(cfg_obj)
                    except Exception:
                        diffusion_obj = None
                
                # Run full evaluation without tracking gradients to avoid autograd graph buildup
                with torch.no_grad():
                    if self.cfg.test_prep_mode:
                        print("test prep mode")
                        _run_test_generation(
                            cfg_obj,
                            self.model,
                            diffusion_obj,
                            detok,
                            gene_names,
                            hvg_to_full,
                            hvg_gene_ids,
                            _Args,
                            self.device,
                            out_dir,
                            1
                        )
                    else:
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
                import traceback
                print(f"[eval] full_eval failed: {e}")
                print(f"[eval] Traceback:\n{traceback.format_exc()}")

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

        # Reset per-phase base learning rate anchors for warmup scaling so they do not
        # leak across phases. We always capture the base from the LR just applied above.
        if ds_spec.warmup_steps is not None and ds_spec.warmup_steps > 0:
            for pg in self.optimizer.param_groups:
                pg["base_lr"] = pg["lr"]
                pg.pop("_base_lr_cached", None)  # clear any previous phase cache flag

        # If the diffusion object is cached on the model from prior phases, update its
        # config for per-dataset token weighting annealing if requested.
        if ds_spec.token_weighting_annealing_steps is not None and hasattr(self.model, "_generic_trainer_diffusion"):
            try:
                self.model._generic_trainer_diffusion.config.token_weighting_annealing_steps = ds_spec.token_weighting_annealing_steps
            except Exception:
                pass
        total_steps_per_epoch = len(dataloader)
        total_steps_per_phase = total_steps_per_epoch*ds_spec.epochs
        max_steps = ds_spec.max_steps or total_steps_per_phase
        autocast_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16

        # Maintain a local per-phase step counter to drive per-dataset warmup
        phase_step = 0

        for local_epoch in range(ds_spec.epochs):
            if self.cfg.eval_mode: 
                self._eval(ds_spec)
                break
            epoch_losses: List[float] = []
            # Initialize gradient norm tracking variables
            last_grad_norm_before = 0.0
            last_grad_norm_after = 0.0
            self.model.train()
            start_t = time.time()
            for step_idx, batch in enumerate(dataloader):
                if phase_step >= max_steps:
                    break
                batch_std: Batch = adapt_batch(batch, device)
                with get_autocast_ctx(self.model, autocast_dtype):
                    # Pass phase-local step and total phase steps to the loss
                    losses = self.loss_adapter(self.model, batch_std, phase_step, total_steps_per_phase)
                self.optimizer.zero_grad(set_to_none=True)
                loss, raw_loss = losses
                loss.backward()
                # Adaptive Gradient Clipping (AGC)
                grad_norm_before, grad_norm_after = adaptive_clip_grad(self.model.parameters(), clip_factor=0.01)
                # Store gradient norms for logging
                last_grad_norm_before = grad_norm_before
                last_grad_norm_after = grad_norm_after
                self.optimizer.step()
                # Per-dataset schedule: linear warmup then immediate linear cooldown (no flat section),
                # using phase-local steps and total phase steps.
                warmup_steps = int(ds_spec.warmup_steps or 0)
                cooldown_steps = max(1, total_steps_per_phase - max(0, warmup_steps))
                if warmup_steps > 0 and (phase_step + 1) <= warmup_steps:
                    scale = (phase_step + 1) / float(warmup_steps)
                else:
                    steps_into_cooldown = max(0, (phase_step + 1) - max(0, warmup_steps))
                    decay_ratio = max(0.0, (cooldown_steps - steps_into_cooldown) / float(cooldown_steps))
                    scale = decay_ratio
                for pg in self.optimizer.param_groups:
                    base_lr = pg.get("base_lr", pg["lr"])  # base captured at phase start
                    pg["lr"] = base_lr * scale
                phase_step += 1
                self.global_step += 1
                epoch_losses.append(raw_loss.item())
                if phase_step % max(1, ds_spec.log_every_steps) == 0:
                    avg = float(np.mean(epoch_losses[-100:] if len(epoch_losses) > 100 else epoch_losses))
                    # Gather learning rates (support muon+adam split or generic optimizer)
                    muon_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups if pg.get("use_muon", False)]
                    adam_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups if not pg.get("use_muon", False)]
                    has_split = any(pg.get("use_muon", False) for pg in self.optimizer.param_groups)
                    lr_muon = float(np.mean(muon_lrs)) if muon_lrs else None
                    lr_adam = float(np.mean(adam_lrs)) if adam_lrs else None
                    lr_overall = float(np.mean([pg.get("lr", 0.0) for pg in self.optimizer.param_groups]))
                    stats = {
                        "train/loss": raw_loss.item(),
                        "train/loss_avg": avg,
                        "train/epoch": self.global_epoch,
                        "train/global_step": self.global_step,
                        "train/lr": lr_overall,
                        "train/grad_norm_before": last_grad_norm_before,
                        "train/grad_norm_after": last_grad_norm_after,
                    }
                    if has_split:
                        if lr_muon is not None:
                            stats["train/lr_muon"] = lr_muon
                        if lr_adam is not None:
                            stats["train/lr_adam"] = lr_adam
                    # Merge auxiliary loss breakdown if available
                    extra = getattr(self.model, "_last_loss_stats", None)
                    if isinstance(extra, dict):
                        stats.update({
                            "train/loss_main": extra.get("loss_main", None),
                            "train/loss_aux": extra.get("loss_aux", None),
                            "train/loss_aux_de": extra.get("loss_aux_de", None),
                            "train/loss_aux_dir": extra.get("loss_aux_dir", None),
                            "train/loss_aux_rank": extra.get("loss_aux_rank", None),
                            "train/loss_total": extra.get("loss_total", None),
                        })
                    # Build LR string for console
                    if has_split:
                        lr_str = f"lr_muon {lr_muon:.3e} lr_adam {lr_adam:.3e}"
                    else:
                        lr_str = f"lr {lr_overall:.3e}"
                    print(f"[Epoch {local_epoch}][{phase_step}/{total_steps_per_phase}] | loss {stats['train/loss']:.3f} avg_loss {stats['train/loss_avg']:.3f} | grad_norm: {last_grad_norm_before:.3f} -> {last_grad_norm_after:.3f} | {lr_str} | lr_overall {lr_overall:.3e}")
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