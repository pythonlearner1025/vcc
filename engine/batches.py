from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader


def _try_get_unique_batches(obj: Any) -> List[str]:
    # 1) explicit attribute provided by dataset/collator wrappers
    if hasattr(obj, "unique_batches") and isinstance(getattr(obj, "unique_batches"), (list, tuple)):
        return list(getattr(obj, "unique_batches"))
    # 2) AnnData-backed datasets (e.g. VCC) – inspect obs['batch'] if available
    adata = getattr(obj, "adata", None)
    if adata is not None:
        try:
            batches = list(map(str, list(set(adata.obs["batch"].values))))
            batches.sort()
            return batches
        except Exception:
            pass
    return []


def discover_phase_batches(phase_name: str, dataset: Any, dataloader: DataLoader) -> List[str]:
    names: List[str] = []
    names.extend(_try_get_unique_batches(dataset))
    # In case the dataloader wraps another dataset (e.g., Tokenized wrapper)
    base_ds = getattr(dataloader, "dataset", None)
    if base_ds is not None and base_ds is not dataset:
        names.extend(_try_get_unique_batches(base_ds))
    # de-duplicate
    names = sorted(list(set(names)))
    if not names:
        # Treat entire dataset as a single technical batch
        return [f"{phase_name}"]
    return names


def build_global_batch_mapping(phases: Sequence[Tuple[str, Any, DataLoader]]) -> Dict[str, int]:
    all_names: List[str] = []
    for phase_name, dataset, dataloader in phases:
        local = discover_phase_batches(phase_name, dataset, dataloader)
        all_names.extend(local)
    unique = sorted(list(set(all_names)))
    return {name: idx for idx, name in enumerate(unique)}


class _CollateWithBatchIndex:
    def __init__(self, base_collate, mapping: Dict[str, int], default_name: str):
        self.base_collate = base_collate
        self.batch_to_idx = mapping
        self.default_name = default_name

    def __call__(self, batch):
        out = self.base_collate(batch) if callable(self.base_collate) else batch
        # Expect dict or tensor; insert batch_idx if missing
        if isinstance(out, dict):
            if "batch_idx" not in out:
                # If per-sample batch names are present (optional), map them
                names = out.get("batch_name", None)
                if names is not None:
                    if isinstance(names, list):
                        idx = torch.tensor([self.batch_to_idx.get(str(n), self.batch_to_idx[self.default_name]) for n in names], dtype=torch.long)
                    else:
                        # assume tensor of strings is unlikely; fallback to default
                        bsz = out["tokens"].shape[0]
                        idx = torch.full((bsz,), self.batch_to_idx[self.default_name], dtype=torch.long)
                else:
                    bsz = out["tokens"].shape[0]
                    idx = torch.full((bsz,), self.batch_to_idx[self.default_name], dtype=torch.long)
                out["batch_idx"] = idx
        elif isinstance(out, torch.Tensor):
            bsz = out.shape[0]
            # convert to dict
            out = {
                "tokens": out,
                "batch_idx": torch.full((bsz,), self.batch_to_idx[self.default_name], dtype=torch.long),
            }
        return out


def attach_global_batch_mapping(phase_name: str, dataset: Any, dataloader: DataLoader, mapping: Dict[str, int]) -> DataLoader:
    # Best-effort: populate on collate_fn and dataset if they expose fields
    default_name = discover_phase_batches(phase_name, dataset, dataloader)[0]

    # 1) Collate function attribute
    if hasattr(dataloader, "collate_fn") and dataloader.collate_fn is not None:
        cf = dataloader.collate_fn
        # inject mapping if the collate object supports it
        if hasattr(cf, "batch_to_idx"):
            try:
                cf.batch_to_idx = mapping
            except Exception:
                pass
        # Wrap collate to enforce batch_idx presence
        from torch.utils.data._utils.collate import default_collate
        base = cf if callable(cf) else default_collate
        dataloader.collate_fn = _CollateWithBatchIndex(base, mapping, default_name)
    else:
        # No collate: wrap default collate
        from torch.utils.data._utils.collate import default_collate
        dataloader.collate_fn = _CollateWithBatchIndex(default_collate, mapping, default_name)

    # 2) Dataset attributes
    for target in [dataset, getattr(dataloader, "dataset", None)]:
        if target is None:
            continue
        if hasattr(target, "batch_to_idx"):
            try:
                setattr(target, "batch_to_idx", mapping)
            except Exception:
                pass
        if hasattr(target, "set_batch_mapping") and callable(getattr(target, "set_batch_mapping")):
            try:
                target.set_batch_mapping(mapping)
            except Exception:
                pass
    return dataloader


def pad_batches_for_fp8(n_batches: int) -> int:
    # Round up to multiple of 16 for FP8 kernels if needed
    if n_batches % 16 == 0:
        return n_batches
    return ((n_batches + 15) // 16) * 16