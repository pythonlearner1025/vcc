#!/usr/bin/env python3
"""
Minimal evaluation aligned with the generic YAML-driven trainer.

Behavior:
- Load a checkpoint .pt and its config (prefer provided YAML; fall back to config embedded in the checkpoint)
- Filter to VCC finetune phase(s) only
- Build dataloaders, compute global technical batch mapping, construct the model, load weights
- Attach mapping and run a concise validation loop (like eval_mode)
- Optionally accept --test_prep (no-op placeholder for now to keep this script simple)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from engine.utils import import_from_string, adapt_batch, load_checkpoint
from engine.batches import build_global_batch_mapping, attach_global_batch_mapping, pad_batches_for_fp8


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
    validation_steps: Optional[int] = None
    muon_lr: Optional[float] = None
    adam_lr: Optional[float] = None
    warmup_steps: Optional[int] = None
    token_weighting_annealing_steps: Optional[int] = None
    hooks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    model_class: str
    model_args: Dict[str, Any] = field(default_factory=dict)
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
    eval_mode: bool = True
    output_dir: str = "checkpoints"
    seed: int = 42
    device: str = "cuda"
    amp_dtype: str = "bf16"
    use_wandb: bool = False
    wandb_project: str = "eval"
    model: ModelSpec = field(default_factory=lambda: ModelSpec(model_class=""))  # type: ignore
    optimizer: OptimizerSpec = field(default_factory=lambda: OptimizerSpec(optimizer_class="torch.optim.Adam"))  # type: ignore
    loss: LossSpec = field(default_factory=lambda: LossSpec(loss_class="engine.losses:DiffusionLoss"))  # type: ignore
    datasets: List[DatasetSpec] = field(default_factory=list)
    resume_from: Optional[str] = None


def _dict_to_obj(cls, d: Dict[str, Any]):
    if hasattr(cls, "__dataclass_fields__"):
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore
        kwargs = {}
        for k, v in d.items():
            if k not in fields:
                continue
            f = cls.__dataclass_fields__[k]  # type: ignore
            if isinstance(v, dict) and hasattr(f.type, "__dataclass_fields__"):
                kwargs[k] = _dict_to_obj(f.type, v)
            elif isinstance(v, list) and len(v) > 0 and hasattr(f.type, "__args__"):
                inner = f.type.__args__[0]
                kwargs[k] = [_dict_to_obj(inner, x) if isinstance(x, dict) else x for x in v]
            else:
                kwargs[k] = v
        return cls(**kwargs)
    return d


def _load_cfg_from_yaml_or_ckpt(ckpt_path: Path, yaml_path: Optional[Path]) -> ExperimentConfig:
    if yaml_path is not None and yaml_path.exists():
        import yaml
        raw = yaml.safe_load(open(yaml_path, "r"))
        return _dict_to_obj(ExperimentConfig, raw)
    payload = load_checkpoint(str(ckpt_path))
    raw_cfg = payload.get("config")
    if raw_cfg is None:
        raise RuntimeError("Checkpoint has no embedded config; provide --config YAML path")
    return _dict_to_obj(ExperimentConfig, raw_cfg)


def _filter_to_vcc_validation(cfg: ExperimentConfig) -> ExperimentConfig:
    vcc_phases: List[DatasetSpec] = [ds for ds in cfg.datasets if "vcc" in ds.name.lower()]
    if not vcc_phases:
        raise RuntimeError("No VCC phases found in config; cannot run evaluation")
    for ds in vcc_phases:
        ds.epochs = 1
    cfg.datasets = vcc_phases
    cfg.eval_mode = True
    return cfg


def _ensure_dataclasses(cfg: ExperimentConfig) -> ExperimentConfig:
    # Coerce nested mappings into dataclass instances if any slipped through
    if isinstance(cfg.model, dict):
        cfg.model = ModelSpec(**cfg.model)  # type: ignore
    if isinstance(cfg.optimizer, dict):
        cfg.optimizer = OptimizerSpec(**cfg.optimizer)  # type: ignore
    if isinstance(cfg.loss, dict):
        cfg.loss = LossSpec(**cfg.loss)  # type: ignore
    if isinstance(cfg.datasets, list):
        new_ds: List[DatasetSpec] = []
        for ds in cfg.datasets:
            if isinstance(ds, DatasetSpec):
                new_ds.append(ds)
            elif isinstance(ds, dict):
                new_ds.append(DatasetSpec(**ds))
            else:
                raise TypeError("Unsupported dataset spec type")
        cfg.datasets = new_ds
    return cfg


def _build_vcc_dataloaders(cfg: ExperimentConfig):
    phases: List[Tuple[str, Any, Any]] = []  # (name, dataset, dataloader)
    for ds_spec in cfg.datasets:
        factory = import_from_string(ds_spec.dataloader_factory)
        result = factory(**ds_spec.dataloader_args)
        if isinstance(result, tuple) and len(result) == 2:
            dataset, dataloader = result
        else:
            raise TypeError("Dataloader factory must return (dataset, dataloader)")
        phases.append((ds_spec.name, dataset, dataloader))
    return phases


def _build_model_from_cfg(cfg: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    ModelClass = import_from_string(cfg.model.model_class)
    if cfg.model.config_class:
        ConfigClass = import_from_string(cfg.model.config_class)
        config_obj = ConfigClass(**cfg.model.config_args)
        model = ModelClass(config_obj)
    else:
        model = ModelClass(**cfg.model.model_args)
    return model.to(device)


def _load_model_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    payload = load_checkpoint(str(ckpt_path))
    state = payload.get("model_state_dict")
    if state is None:
        raise RuntimeError("Checkpoint missing model_state_dict")
    model.load_state_dict(state, strict=False)


def _read_ckpt_n_batches(ckpt_path: Path) -> Optional[int]:
    try:
        payload = load_checkpoint(str(ckpt_path))
        cfg = payload.get("config", {})
        model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        config_args = model.get("config_args", {}) if isinstance(model, dict) else {}
        n_tb = config_args.get("n_technical_batches", None)
        if isinstance(n_tb, int):
            return n_tb
    except Exception:
        pass
    return None


def run_validation(cfg: ExperimentConfig, ckpt_path: Path, yaml_path: Optional[Path], max_val_steps: Optional[int] = None) -> Dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.amp_dtype.lower() == "bf16" else torch.float16

    phases = _build_vcc_dataloaders(cfg)

    mapping = build_global_batch_mapping(phases)
    # Prefer the n_technical_batches that the checkpoint was trained with to avoid size mismatch.
    ckpt_n_tb = _read_ckpt_n_batches(ckpt_path)
    if ckpt_n_tb is not None:
        cfg.model.config_args["n_technical_batches"] = ckpt_n_tb
    else:
        if cfg.model.config_args is not None:
            cfg.model.config_args["n_technical_batches"] = pad_batches_for_fp8(len(mapping))

    model = _build_model_from_cfg(cfg, device)
    _load_model_weights(model, ckpt_path)
    model.eval().requires_grad_(False)

    phases = [(name, ds, attach_global_batch_mapping(name, ds, dl, mapping)) for name, ds, dl in phases]

    LossClass = import_from_string(cfg.loss.loss_class)
    loss_obj = LossClass(**cfg.loss.loss_args)

    results: Dict[str, Any] = {}
    for phase_name, dataset, train_loader in phases:
        val_loader = getattr(train_loader, "val_dataloader", None)
        val_dataset = getattr(train_loader, "val_dataset", None)
        if val_loader is None or val_dataset is None:
            print(f"[eval] No validation dataloader available for phase {phase_name}")
            continue
        val_loader = attach_global_batch_mapping(phase_name + "_val", val_dataset, val_loader, mapping)

        losses: List[float] = []
        with torch.no_grad():
            for step_idx, batch in enumerate(val_loader):
                if max_val_steps is not None and step_idx >= max_val_steps:
                    break
                batch_std = adapt_batch(batch, device)
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    loss = loss_obj.compute_loss(model, batch_std, step=0)
                losses.append(float(loss.item()))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        results[phase_name] = {"val_loss": avg_loss, "n_steps": len(losses)}
        print(f"[eval:{phase_name}] avg_loss={avg_loss:.4f} steps={len(losses)}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint with new trainer logic")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML to override embedded config")
    ap.add_argument("--max_val_steps", type=int, default=None, help="Optional cap on validation steps")
    ap.add_argument("--test_prep", action="store_true", help="Optional: placeholder flag for test prep")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    yaml_path = Path(args.config) if args.config else None

    cfg_full = _load_cfg_from_yaml_or_ckpt(ckpt_path, yaml_path)
    cfg_full = _ensure_dataclasses(cfg_full)
    cfg_eval = _filter_to_vcc_validation(cfg_full)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evals") / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_validation(cfg_eval, ckpt_path, yaml_path, max_val_steps=args.max_val_steps)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "results": results}, f, indent=2)
    print(f"Saved results → {out_dir/'summary.json'}")

    if args.test_prep:
        print("--test_prep was set; placeholder only (no generation performed).")


if __name__ == "__main__":
    main()