#!/usr/bin/env python3
"""
Orion paired dataset & dataloader utilities

This module provides a drop-in replacement for `dataset.vcc_paired_dataloader`.

The dataset now returns a *set* of `set_size` perturbed cells that all share the
same `gene_target`, plus a *matched* control set of the same size coming from
non-targeting cells **preferentially from the same GEM batch**.  This exactly
mirrors the semantics expected by the existing `VCCCollator`:
`perturbed_expr` and `control_expr` are `(S, N)` float tensors where
`S == set_size` and `N == n_genes`.
"""
from __future__ import annotations

import json
import random
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="Variable names are not unique*", category=UserWarning)

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from dataset.collators import OrionCollator


__all__ = [
    "OrionPairedDataset",
    "create_orion_paired_dataloader",
    "create_orion_train_val_dataloaders",
]


class OrionPairedDataset(Dataset):  # noqa: E501
    """Dataset yielding `(set_size, n_genes)` perturbed/control expression sets."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        batches_dir: str | Path,
        *,
        set_size: int = 16,
        hvg_gene_ids: List[str] | None = None,
        control_label: str = "Non-Targeting",
        seed: int = 42,
    ) -> None:
        self.root = Path(batches_dir)
        self.set_size = set_size
        self.control_label = control_label
        self.rng = random.Random(seed)

        self._scan_batches()

        # ------------------------------------------------------------------
        # Runtime cache for opened .h5ad files (per DataLoader worker)
        # ------------------------------------------------------------------
        self._adata_cache: Dict[str, sc.AnnData] = {}
        self._gene_symbols_cache: Dict[str, np.ndarray] = {}
        self._sample_cache: Dict[str, np.ndarray] = {}

        # Apply HVG filtering if provided
        if hvg_gene_ids is not None:
            self._apply_hvg_filter(hvg_gene_ids)
            self.hvg_gene_ids = hvg_gene_ids
        else:
            self.hvg_gene_ids = self.gene_list  # defined in _scan_batches

        self.target_genes = sorted(self._pert_cells_by_gene.keys())
        # Pre-compute total number of samples (= genes) for __len__
        self._gene_for_set: List[str] = []
        for gt in self.target_genes:
            cells_per_gene = len(self._pert_cells_by_gene[gt])
            n_sets = (cells_per_gene + self.set_size - 1) // self.set_size
            self._gene_for_set.extend([gt] * n_sets)

    # ------------------------------------------------------------------
    # Metadata scanning helpers
    # ------------------------------------------------------------------

    def _scan_batches(self) -> None:
        """Build or load cached batch indexes used for efficient sampling."""
        cache_path = self.root / "orion_index_cache.pkl"

        # ------------------------------------------------------------------
        # Fast path – load from cache if available
        # ------------------------------------------------------------------
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cache = pickle.load(f)

                # Restore attributes
                self._pert_cells_by_gene = defaultdict(list, cache["pert_cells_by_gene"])
                self._ctrl_by_sample = defaultdict(list, cache["ctrl_by_sample"])
                self._ctrl_pool = cache["ctrl_pool"]
                self._ctrl_file_to_count = defaultdict(int, cache["ctrl_file_to_count"])
                self.gene_list = cache["gene_list"]
                self._keep_gene_idx = cache.get("keep_gene_idx")
                self.unique_batches = cache["unique_batches"]
                return  # loaded successfully – skip expensive scan
            except Exception as e:
                # Cache may be corrupted or incompatible with current code – rebuild
                print(f"Warning: failed to load Orion index cache: {e}. Rebuilding …")

        # ------------------------------------------------------------------
        # Slow path – scan all batches to build the index
        # ------------------------------------------------------------------
        meta_paths = sorted(self.root.glob("*.json"))
        if not meta_paths:
            raise FileNotFoundError(f"No *.json metadata files in {self.root}")

        self._pert_cells_by_gene: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
        self._ctrl_by_sample: defaultdict[str, List[Tuple[str, int]]] = defaultdict(list)

        first_gene_list: List[str] | None = None
        for meta_path in tqdm(meta_paths, desc="Scanning batches"):
            meta = json.loads(meta_path.read_text())
            h5_path = str(meta.get("file", meta_path.with_suffix(".h5ad")))
            adata = sc.read_h5ad(h5_path, backed="r")

            if first_gene_list is None:
                first_gene_list = list(adata.var_names)

            gene_targets = adata.obs["gene_target"].values
            samples = adata.obs["sample"].values
            n = adata.n_obs
            for i in range(n):
                gt = str(gene_targets[i])
                sample = str(samples[i])
                if gt == self.control_label:
                    self._ctrl_by_sample[sample].append((h5_path, i))
                else:
                    self._pert_cells_by_gene[gt].append((h5_path, i, sample))
            adata.file.close()

        if not self._pert_cells_by_gene:
            raise RuntimeError("No perturbed cells found in provided batches")
        if not self._ctrl_by_sample:
            raise RuntimeError(f"No control cells with label '{self.control_label}' found")

        # Build flat control index for fallback sampling
        self._ctrl_pool: List[Tuple[str, int]] = sum(self._ctrl_by_sample.values(), [])
        self._ctrl_file_to_count: Dict[str, int] = defaultdict(int)
        for fp, _idx in self._ctrl_pool:
            self._ctrl_file_to_count[fp] += 1

        self.gene_list = first_gene_list if first_gene_list is not None else []

        # ------------------------------------------------------------------
        # Collapse duplicate genes (keep one copy randomly)
        # ------------------------------------------------------------------
        if self.gene_list:
            symbol_to_indices: Dict[str, List[int]] = defaultdict(list)
            for idx, g in enumerate(self.gene_list):
                symbol_to_indices[g].append(idx)

            keep_idx_set = set()
            duplicates = 0
            for sym, idx_list in symbol_to_indices.items():
                if len(idx_list) == 1:
                    keep_idx_set.add(idx_list[0])
                else:
                    chosen = self.rng.choice(idx_list)
                    keep_idx_set.add(chosen)
                    duplicates += len(idx_list) - 1
            if duplicates:
                print(f"OrionPairedDataset: collapsed {duplicates} duplicate gene columns → {len(keep_idx_set)} unique genes")
            self._keep_gene_idx = sorted(keep_idx_set)
            self.gene_list = [self.gene_list[i] for i in self._keep_gene_idx]
        else:
            self._keep_gene_idx = None

        # ------------------------------------------------------------------
        # Collect unique batch / sample names for downstream conditioning
        # ------------------------------------------------------------------
        pert_samples_all = [s for cells in self._pert_cells_by_gene.values() for _fp, _idx, s in cells]
        ctrl_samples_all = list(self._ctrl_by_sample.keys())
        self.unique_batches = sorted(set(pert_samples_all) | set(ctrl_samples_all))

        # ------------------------------------------------------------------
        # Save the newly built index for future runs
        # ------------------------------------------------------------------
        try:
            cache = {
                "pert_cells_by_gene": dict(self._pert_cells_by_gene),
                "ctrl_by_sample": dict(self._ctrl_by_sample),
                "ctrl_pool": self._ctrl_pool,
                "ctrl_file_to_count": dict(self._ctrl_file_to_count),
                "gene_list": self.gene_list,
                "keep_gene_idx": self._keep_gene_idx,
                "unique_batches": self.unique_batches
            }
            with cache_path.open("wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            # Do not fail hard if caching fails – still usable
            print(f"Warning: failed to write Orion index cache: {e}")

    # ------------------------------------------------------------------
    # HVG filtering
    # ------------------------------------------------------------------

    def _apply_hvg_filter(self, hvg_gene_ids: List[str]):
        # Build map from gene -> index in full gene list
        idx_map = {g: i for i, g in enumerate(self.gene_list)}
        self._hvg_idx: List[int] = []
        missing = 0
        for g in hvg_gene_ids:
            if g in idx_map:
                self._hvg_idx.append(idx_map[g])
            else:
                missing += 1
        if missing:
            print(f"OrionPairedDataset: {missing} HVG genes not found in data – ignored")
        if not self._hvg_idx:
            raise RuntimeError("None of the provided HVG genes found in Orion data")

    # ------------------------------------------------------------------
    # Utility helper – load one expression vector
    # ------------------------------------------------------------------

    def _get_adata(self, file_path: str):
        """Return a cached `AnnData` handle (backed mode, read-only)."""
        adata = self._adata_cache.get(file_path)
        if adata is None:
            # Opening with backed='r' is fast and memory-efficient – keep handle open
            adata = sc.read_h5ad(file_path, backed="r")
            self._adata_cache[file_path] = adata
            # Cache gene symbols and sample column for faster access later on
            gene_symbols = (
                adata.var["gene_symbol"].values
                if "gene_symbol" in adata.var.columns
                else adata.var_names.values
            )
            self._gene_symbols_cache[file_path] = gene_symbols
            self._sample_cache[file_path] = adata.obs["sample"].values  # NumPy array – cheap to keep
        return adata

    def _load_row(self, file_path: str, row_idx: int):
        adata = self._get_adata(file_path)
        row = adata.X[row_idx]
        expr_full = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()

        # Apply duplicate collapse
        if self._keep_gene_idx is not None:
            expr_unique = expr_full[self._keep_gene_idx]
        else:
            expr_unique = expr_full

        # Apply HVG filter if requested
        if hasattr(self, "_hvg_idx"):
            expr = expr_unique[self._hvg_idx]
        else:
            expr = expr_unique

        # Retrieve cached metadata (avoids repeated string conversions)
        sample_arr = self._sample_cache[file_path]
        sample = str(sample_arr[row_idx])

        gene_symbols = self._gene_symbols_cache[file_path]
        if self._keep_gene_idx is not None:
            gene_symbols = gene_symbols[self._keep_gene_idx]
        if hasattr(self, "_hvg_idx"):
            gene_symbols = gene_symbols[self._hvg_idx]

        return expr, sample, gene_symbols

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):  # type: ignore[override]
        return len(self._gene_for_set)

    # ------------------------------------------------------------------
    # Resource clean-up
    # ------------------------------------------------------------------
    def __del__(self):
        # Ensure we close any open AnnData file handles to avoid file descriptor leaks
        cache = getattr(self, "_adata_cache", {})
        for adata in cache.values():
            try:
                adata.file.close()
            except Exception:
                pass

    def __getitem__(self, idx: int):  # type: ignore[override]
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        gene_target = self._gene_for_set[idx]
        pert_cells = self._pert_cells_by_gene[gene_target]

        # ------------------------------------------------------------------
        # Select perturbed cell indices for this set
        # ------------------------------------------------------------------
        if len(pert_cells) >= self.set_size:
            chosen_pert = self.rng.sample(pert_cells, self.set_size)
        else:
            # sample with replacement
            chosen_pert = [self.rng.choice(pert_cells) for _ in range(self.set_size)]

        # Load perturbed expression rows
        pert_expr_list: List[np.ndarray] = []
        pert_samples: List[str] = []
        gene_symbols_ref: Sequence[str] | None = None
        for fp, row_idx, sample in chosen_pert:
            expr, sample_name, gene_syms = self._load_row(fp, row_idx)
            pert_expr_list.append(expr)
            pert_samples.append(sample_name)
            gene_symbols_ref = gene_syms  # store last seen – same across rows in file

        pert_expr = torch.from_numpy(np.stack(pert_expr_list)).float()  # (S,N)

        # ------------------------------------------------------------------
        # Build control set – batch-matched where possible
        # ------------------------------------------------------------------
        ctrl_expr_list: List[np.ndarray] = []
        ctrl_samples: List[str] = []
        for sample_name in pert_samples:
            candidates = self._ctrl_by_sample.get(sample_name)
            if candidates:
                fp_c, row_c = self.rng.choice(candidates)
            else:
                fp_c, row_c = self.rng.choice(self._ctrl_pool)
            expr_c, sample_c, _gene_syms_c = self._load_row(fp_c, row_c)
            ctrl_expr_list.append(expr_c)
            ctrl_samples.append(sample_c)
        ctrl_expr = torch.from_numpy(np.stack(ctrl_expr_list)).float()  # (S,N)

        # Determine target-gene column index (may be absent ⇒ -1)
        matches = np.where(gene_symbols_ref == gene_target)[0] if gene_symbols_ref is not None else []
        target_gene_idx = int(matches[0]) if len(matches) > 0 else -1

        return {
            "perturbed_expr": pert_expr,   # (S,N)
            "control_expr":   ctrl_expr,   # (S,N)
            "target_gene":    gene_target,
            "target_gene_idx": target_gene_idx,
            "pert_batches":   pert_samples,   # List[str] length S
            "ctrl_batches":   ctrl_samples,   # List[str] length S
        }


# -------------------------------------------------------------------------
# Helper for building DataLoader (shared by train/val)
# -------------------------------------------------------------------------

def _build_dataloader(dataset: Dataset, collate_fn=None, *, shuffle: bool, num_workers: int, prefetch_factor: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=1,  # one set per batch – collator flattens later
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )


# -------------------------------------------------------------------------
# Public API – mirrors vcc_paired_dataloader (subset)
# -------------------------------------------------------------------------

def create_orion_paired_dataloader(
    batches_dir: str | Path,
    set_size: int = 16,
    hvg_gene_ids: List[str] | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
    random_seed: int = 42,
    tokenizer=None,  # kept for signature parity – collate handled by caller
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    control_label: str = "Non-Targeting",
):
    ds = OrionPairedDataset(batches_dir, set_size=set_size, hvg_gene_ids=hvg_gene_ids, control_label=control_label, seed=random_seed)
    dl = _build_dataloader(ds, None, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)
    return ds, dl


def create_orion_train_val_dataloaders(
    batches_dir: str | Path,
    set_size: int = 16,
    hvg_gene_ids: List[str] | None = None,
    num_workers: int = 4,
    random_seed: int = 42,
    tokenizer=None,  # kept for signature parity
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    train_split: float = 0.8,
    shuffle_train: bool = True,
    control_label: str = "Non-Targeting",
):
    full_ds = OrionPairedDataset(batches_dir, set_size=set_size, hvg_gene_ids=hvg_gene_ids, control_label=control_label, seed=random_seed)

    n_total = len(full_ds)
    n_train = int(n_total * train_split)
    rng = random.Random(random_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    collate_fn = OrionCollator(tokenizer, set_size) if tokenizer else None

    train_dl = _build_dataloader(
        train_ds,
        shuffle=shuffle_train,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    val_dl = _build_dataloader(
        val_ds,
        shuffle=False,
        num_workers=0,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return (train_ds, train_dl), (val_ds, val_dl)