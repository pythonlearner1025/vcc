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
        self._ensembl_to_symbol_map: Dict[str, str] = {}  # Map Ensembl IDs to gene symbols

        # Build Ensembl ID to gene symbol mapping if needed
        if hvg_gene_ids is not None and hvg_gene_ids and hvg_gene_ids[0].startswith('ENSG'):
            print("OrionPairedDataset: HVG list contains Ensembl IDs, building mapping to gene symbols...")
            self._build_ensembl_to_symbol_map()
            
            # Also check which Ensembl IDs are unmapped in the actual data
            self._check_unmapped_genes()
            
            # Convert HVG Ensembl IDs to gene symbols
            hvg_symbols = []
            unmapped = []
            kept_as_ensembl = []
            
            for ensembl_id in hvg_gene_ids:
                if ensembl_id in self._unmapped_ensembl_ids:
                    # This gene wasn't properly mapped in the data, keep as Ensembl ID
                    hvg_symbols.append(ensembl_id)
                    kept_as_ensembl.append(ensembl_id)
                elif ensembl_id in self._ensembl_to_symbol_map:
                    hvg_symbols.append(self._ensembl_to_symbol_map[ensembl_id])
                else:
                    unmapped.append(ensembl_id)
            
            if kept_as_ensembl:
                print(f"  Kept {len(kept_as_ensembl)} genes as Ensembl IDs (unmapped in data)")
            if unmapped:
                print(f"  Warning: {len(unmapped)} HVG Ensembl IDs could not be mapped")
            
            print(f"  Converted {len(hvg_symbols)}/{len(hvg_gene_ids)} HVG genes for matching")
            hvg_gene_ids = hvg_symbols  # Use converted list for filtering

        # Apply HVG filtering if provided
        if hvg_gene_ids is not None:
            self._apply_hvg_filter(hvg_gene_ids)
            self.hvg_gene_ids = hvg_gene_ids
        else:
            self.hvg_gene_ids = self.gene_list  # defined in _scan_batches

        all_target_genes = sorted(self._pert_cells_by_gene.keys())
        
        # Filter out target genes that are not in the HVG list
        if hvg_gene_ids is not None:
            hvg_set = set(hvg_gene_ids)
            
            # Keep only target genes that are in HVG list
            target_genes_in_hvg = [gt for gt in all_target_genes if gt in hvg_set]
            target_genes_not_in_hvg = [gt for gt in all_target_genes if gt not in hvg_set]
            
            print(f"OrionPairedDataset: Filtering target genes by HVG list...")
            print(f"  - Total target genes: {len(all_target_genes)}")
            print(f"  - Target genes IN HVG list: {len(target_genes_in_hvg)} ({100*len(target_genes_in_hvg)/len(all_target_genes) if all_target_genes else 0:.1f}%)")
            print(f"  - Target genes NOT in HVG list (excluded): {len(target_genes_not_in_hvg)} ({100*len(target_genes_not_in_hvg)/len(all_target_genes) if all_target_genes else 0:.1f}%)")
            
            if target_genes_not_in_hvg and len(target_genes_not_in_hvg) <= 20:
                print(f"  - Excluded non-HVG targets: {target_genes_not_in_hvg}")
            elif target_genes_not_in_hvg:
                print(f"  - Examples of excluded non-HVG targets: {target_genes_not_in_hvg[:10]}")
            
            # Remove non-HVG target genes from the perturbation dictionary
            for gene_to_remove in target_genes_not_in_hvg:
                del self._pert_cells_by_gene[gene_to_remove]
            
            self.target_genes = target_genes_in_hvg
            print(f"  - Final number of target genes after filtering: {len(self.target_genes)}")
        else:
            # No HVG filtering, use all target genes
            self.target_genes = all_target_genes
            print(f"  - No HVG filtering applied, using all {len(self.target_genes)} target genes")
        
        if not self.target_genes:
            raise RuntimeError("No target genes remain after HVG filtering! Check if HVG list overlaps with perturbed genes.")
        
        # Pre-compute total number of samples (= genes) for __len__
        self._gene_for_set: List[str] = []
        for gt in self.target_genes:
            cells_per_gene = len(self._pert_cells_by_gene[gt])
            n_sets = (cells_per_gene + self.set_size - 1) // self.set_size
            self._gene_for_set.extend([gt] * n_sets)

    # ------------------------------------------------------------------
    # Build Ensembl to gene symbol mapping
    # ------------------------------------------------------------------
    
    def _build_ensembl_to_symbol_map(self) -> None:
        """Build a mapping from Ensembl IDs to gene symbols using symbol2ens.pkl."""
        # Try multiple possible locations for the symbol2ens.pkl file
        possible_paths = [
            Path("/workspace/vcc/symbol2ens.pkl"),
            Path("symbol2ens.pkl"),
            self.root.parent / "symbol2ens.pkl",
            Path("assets/symbol2ens.pkl"),
        ]
        
        symbol2ens_path = None
        for path in possible_paths:
            if path.exists():
                symbol2ens_path = path
                break
        
        if symbol2ens_path is None:
            print(f"  Warning: Could not find symbol2ens.pkl file, falling back to data-based mapping")
            # Fallback to building from data files
            self._build_ensembl_to_symbol_map_from_data()
            return
        
        try:
            # Load the symbol to Ensembl mapping
            with symbol2ens_path.open("rb") as f:
                symbol2ens = pickle.load(f)
            
            # Invert the mapping to get Ensembl to symbol
            for symbol, ensembl in symbol2ens.items():
                # Only add if it's a valid Ensembl ID
                if ensembl.startswith("ENSG"):
                    self._ensembl_to_symbol_map[ensembl] = symbol
            
            print(f"  Built mapping for {len(self._ensembl_to_symbol_map)} Ensembl IDs to gene symbols from {symbol2ens_path}")
        except Exception as e:
            print(f"  Warning: Could not load symbol2ens.pkl: {e}, falling back to data-based mapping")
            self._build_ensembl_to_symbol_map_from_data()
    
    def _check_unmapped_genes(self) -> None:
        """Check which Ensembl IDs are unmapped (gene_symbol == ensembl_id) in the data."""
        self._unmapped_ensembl_ids = set()
        
        meta_paths = sorted(self.root.glob("*.json"))
        if not meta_paths:
            return
            
        # Check the first batch file
        meta = json.loads(meta_paths[0].read_text())
        h5_path = str(meta.get("file", meta_paths[0].with_suffix(".h5ad")))
        
        try:
            adata = sc.read_h5ad(h5_path, backed="r")
            
            if "gene_symbol" in adata.var.columns:
                for ensembl_id, gene_symbol in zip(adata.var_names, adata.var["gene_symbol"]):
                    # If gene_symbol is the same as ensembl_id, it's unmapped
                    if str(ensembl_id).startswith("ENSG") and str(gene_symbol) == str(ensembl_id):
                        self._unmapped_ensembl_ids.add(str(ensembl_id))
            
            adata.file.close()
            
            if self._unmapped_ensembl_ids:
                print(f"  Found {len(self._unmapped_ensembl_ids)} unmapped Ensembl IDs in data")
        except Exception as e:
            print(f"  Warning: Could not check unmapped genes: {e}")
    
    def _build_ensembl_to_symbol_map_from_data(self) -> None:
        """Fallback method to build mapping from data files if symbol2ens.pkl is not available."""
        meta_paths = sorted(self.root.glob("*.json"))
        if not meta_paths:
            return
        
        # Try to load from the first batch
        meta = json.loads(meta_paths[0].read_text())
        h5_path = str(meta.get("file", meta_paths[0].with_suffix(".h5ad")))
        
        try:
            adata = sc.read_h5ad(h5_path, backed="r")
            
            # Build the mapping
            if "gene_symbol" in adata.var.columns:
                for ensembl_id, gene_symbol in zip(adata.var_names, adata.var["gene_symbol"]):
                    # For unmapped genes (where symbol == ensembl_id), keep the Ensembl ID as the symbol
                    # This ensures we don't lose genes like ENSG00000170846 that weren't properly mapped
                    if str(gene_symbol).startswith("ENSG") and str(ensembl_id) == str(gene_symbol):
                        # Keep the Ensembl ID as the symbol for unmapped genes
                        self._ensembl_to_symbol_map[str(ensembl_id)] = str(gene_symbol)
                    elif not str(gene_symbol).startswith("ENSG"):
                        # Normal case: gene symbol is different from Ensembl ID
                        self._ensembl_to_symbol_map[str(ensembl_id)] = str(gene_symbol)
            
            adata.file.close()
            print(f"  Built mapping for {len(self._ensembl_to_symbol_map)} Ensembl IDs to gene symbols from data")
        except Exception as e:
            print(f"  Warning: Could not build Ensembl to symbol mapping from data: {e}")

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
                self.gene_symbols_list = cache.get("gene_symbols_list", self.gene_list)
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
        
        # Also get gene symbols if available
        if meta_paths:
            try:
                meta = json.loads(meta_paths[0].read_text())
                h5_path = str(meta.get("file", meta_paths[0].with_suffix(".h5ad")))
                adata = sc.read_h5ad(h5_path, backed="r")
                if "gene_symbol" in adata.var.columns:
                    self.gene_symbols_list = list(adata.var["gene_symbol"].values)
                else:
                    self.gene_symbols_list = self.gene_list
                adata.file.close()
            except:
                self.gene_symbols_list = self.gene_list
        else:
            self.gene_symbols_list = self.gene_list

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
            # Also update gene_symbols_list if it exists
            if hasattr(self, 'gene_symbols_list') and self.gene_symbols_list:
                self.gene_symbols_list = [self.gene_symbols_list[i] for i in self._keep_gene_idx]
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
                "gene_symbols_list": getattr(self, 'gene_symbols_list', self.gene_list),
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
        # Note: gene_list may contain Ensembl IDs, while hvg_gene_ids are gene symbols after conversion
        idx_map = {g: i for i, g in enumerate(self.gene_list)}
        
        # Also try with gene symbols if we have them
        if hasattr(self, 'gene_symbols_list'):
            idx_map_symbols = {g: i for i, g in enumerate(self.gene_symbols_list)}
        else:
            idx_map_symbols = idx_map
        
        self._hvg_idx: List[int] = []
        missing = 0
        for g in hvg_gene_ids:
            if g in idx_map_symbols:
                self._hvg_idx.append(idx_map_symbols[g])
            elif g in idx_map:
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

        # Determine target-gene column index
        # After filtering, all target genes should be in the HVG list
        matches = np.where(gene_symbols_ref == gene_target)[0] if gene_symbols_ref is not None else []
        
        if len(matches) == 0:
            # This should not happen after filtering non-HVG targets
            print(f"ERROR: Target gene '{gene_target}' not found in gene symbols after HVG filtering - this shouldn't happen!")
            print(f"DEBUG: gene_symbols shape: {gene_symbols_ref.shape if hasattr(gene_symbols_ref, 'shape') else len(gene_symbols_ref) if gene_symbols_ref is not None else 0}")
            print(f"DEBUG: First 10 gene symbols: {list(gene_symbols_ref[:10]) if gene_symbols_ref is not None and len(gene_symbols_ref) > 0 else []}")
            target_gene_idx = -1
        elif len(matches) > 1:
            print(f"Warning: Multiple matches ({len(matches)}) found for target gene '{gene_target}', using first match at index {matches[0]}")
            target_gene_idx = int(matches[0])
        else:
            # Successfully found the target gene
            target_gene_idx = int(matches[0])

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

def _build_dataloader(dataset: Dataset, collate_fn=None, *, shuffle: bool, num_workers: int, prefetch_factor: int, pin_memory: bool, batch_size: int = 1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
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
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    random_seed: int = 42,
    tokenizer=None,  # kept for signature parity – collate handled by caller
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    control_label: str = "Non-Targeting",
):
    ds = OrionPairedDataset(batches_dir, set_size=set_size, hvg_gene_ids=hvg_gene_ids, control_label=control_label, seed=random_seed)
    dl = _build_dataloader(ds, None, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory, batch_size=batch_size)
    return ds, dl


def create_orion_train_val_dataloaders(
    batches_dir: str | Path,
    set_size: int = 16,
    hvg_gene_ids: List[str] | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    random_seed: int = 42,
    tokenizer=None,  # kept for signature parity
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    train_split: float = 1.0,
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
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    val_dl = _build_dataloader(
        val_ds,
        shuffle=False,
        num_workers=0,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        batch_size=24, # hardcode to 24
    )

    return (train_ds, train_dl), (val_ds, val_dl)