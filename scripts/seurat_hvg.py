#!/usr/bin/env python3
"""
compute_hvg_scanpy.py
--------------------------------
Port of *scran_hvg.py* that replaces the R‑based scran HVG computation with
`scanpy.pp.highly_variable_genes` (flavour='seurat_v3').

Key additions
-------------
1. **scanpy‑native HVG selection** – identical Seurat v3 formula, no R bridge.
2. **Gene whitelist** – genes supplied by the user (e.g. validation/test panel)
   are force‑included in the final HVG list.
3. **Memory‑aware streaming** – the (potentially 100 M+ cell) pre‑training
   corpus is subsampled or processed in blocks, avoiding dense loading.

This script yields a single text file containing an *ordered* HVG list
(common ordering = gene list) to be used by both pre‑training and
fine‑tuning stages.

Examples
~~~~~~~~
$ python compute_hvg_scanpy.py \
    --scrna_dir /path/to/batched_pretrain \
    --finetune_path /path/to/finetune.h5ad \
    --whitelist_path /path/to/whitelist.txt \
    --n_hvgs 2000 \
    --max_cells_pretrain 1000000 \
    --output_dir ./hvg_out
"""

from __future__ import annotations
import argparse
from pathlib import Path
import logging
import random
from typing import List, Set
import sys
import os

# Add scripts directory to Python path to import fast_hvg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import fast_hvg
import scanpy as sc
import anndata as ad
import h5py
import glob
import scipy.sparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_whitelist(path: str | None) -> Set[str]:
    if path is None:
        return set()
    with open(path) as fh:
        genes = {ln.strip() for ln in fh if ln.strip()}
    logger.info("Whitelist size: %d genes", len(genes))
    return genes

def load_finetune(path: str, max_cells: int = None) -> ad.AnnData:
    logger.info("Loading finetune dataset: %s", path)
    
    # First, check the size of the dataset
    adata_info = sc.read_h5ad(path, backed='r')
    n_cells_total = adata_info.n_obs
    logger.info("Finetune dataset size: %d cells  %d genes", n_cells_total, adata_info.n_vars)
    
    # If we need to subsample, do it efficiently
    if max_cells is not None and n_cells_total > max_cells:
        logger.info("Subsampling finetune data from %d to %d cells", n_cells_total, max_cells)
        indices = np.random.choice(n_cells_total, size=max_cells, replace=False)
        indices = np.sort(indices)  # Sort for efficient h5 access
        
        # Load only the sampled cells
        adata = adata_info[indices, :].to_memory()
        adata_info.file.close()  # Close the backed file
    else:
        # If no subsampling or dataset is small enough, load fully
        adata_info.file.close()
        adata = sc.read_h5ad(path, backed=None)
    
    logger.info("Loaded finetune cells: %d  genes: %d", *adata.shape)
    
    # Use Ensembl IDs if available for consistent gene matching
    if 'gene_id' in adata.var.columns:
        # Store original gene symbols for later use
        adata.var['gene_symbol'] = adata.var_names
        # Use Ensembl IDs as var_names for intersection
        adata.var_names = adata.var['gene_id']
        logger.info("Using Ensembl IDs from 'gene_id' column for gene matching")
    
    # Ensure data is in CSR format for memory efficiency
    if hasattr(adata.X, 'tocsr'):
        adata.X = adata.X.tocsr()
    
    return adata

# ---------------------------------------------------------------------
# Pre‑training data streaming
# ---------------------------------------------------------------------

def _collect_batch_files(scrna_dir: str) -> List[str]:
    return sorted(glob.glob(str(Path(scrna_dir) / "batch_*.h5")))


def _sample_cells_from_batch(batch_file: str, target: int) -> tuple[np.ndarray, List[str]]:
    """Randomly sample *target* cells (rows) from an on‑disk h5 batch."""
    with h5py.File(batch_file, "r") as f:
        X = f["X"]  # shape (cells, genes) – assumed CSR‑compressed sparse saved dense
        n_cells = X.shape[0]
        idx = np.random.choice(n_cells, size=int(min(target, n_cells)), replace=False)
        # Sort indices for h5py compatibility
        idx = np.sort(idx)
        data = X[idx]
        # to CSR sparse to keep memory low if not already
        if not hasattr(data, "todense"):
            # data is dense numpy array, convert to sparse
            data = scipy.sparse.csr_matrix(data)
        else:
            # data is already sparse, ensure it's CSR format
            data = data.tocsr()
        cells = f["cells"][idx].astype(str).tolist()
    return data, cells


def load_pretrain_sample(scrna_dir: str, max_cells: int) -> ad.AnnData:
    """Return an AnnData composed of <=*max_cells* sampled uniformly over batches."""
    batch_files = _collect_batch_files(scrna_dir)
    if not batch_files:
        raise FileNotFoundError("No batch_*.h5 found in %s" % scrna_dir)
    logger.info("Found %d pretrain batches", len(batch_files))

    # First file supplies gene list
    with h5py.File(batch_files[0], "r") as f:
        genes = f["genes"][:].astype(str)
    per_batch = max_cells // len(batch_files)
    X_blocks, cell_ids = [], []
    for bf in batch_files:
        Xb, cells_b = _sample_cells_from_batch(bf, per_batch)
        X_blocks.append(Xb)
        cell_ids.extend(cells_b)
    X = scipy.sparse.vstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]
    adata = ad.AnnData(X, obs=dict(cell_id=cell_ids), var=dict(gene_id=genes))
    # Set gene_id as the var index so intersection works properly
    adata.var_names = adata.var["gene_id"]
    logger.info("Pretrain sample: %d cells  %d genes", *adata.shape)
    return adata

# ---------------------------------------------------------------------
# HVG computation (Seurat v3 flavour)
# ---------------------------------------------------------------------

def compute_hvgs(adata: ad.AnnData, n_top: int) -> List[str]:
    logger.info("Running scanpy.pp.highly_variable_genes (seurat_v3) …")
    
    # Create a copy of just the data we need to avoid modifying views
    print("copying...")
    if adata.is_view:
        adata = adata.copy()
    
    print("entering...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="seurat_v3"
    )
    hvgs = adata.var.index[adata.var["highly_variable"]].tolist()
    logger.info("Detected %d HVGs", len(hvgs))
    return hvgs

# ---------------------------------------------------------------------
# HVG merging / whitelisting
# ---------------------------------------------------------------------

def merge_hvg_lists(
    pretrain_hvgs: List[str],
    finetune_hvgs: List[str],
    whitelist: Set[str],
    n_final: int,
) -> List[str]:
    """Return *ordered* union: whitelist first, then HVGs by precedence.

    Precedence = appear in both > finetune only > pretrain only (modifiable).
    The list is trimmed / extended to satisfy *n_final* while never dropping
    any whitelisted genes.
    """
    common = [g for g in pretrain_hvgs if g in finetune_hvgs]
    only_fine = [g for g in finetune_hvgs if g not in pretrain_hvgs]
    only_pre = [g for g in pretrain_hvgs if g not in finetune_hvgs]

    ordered = list(dict.fromkeys(list(whitelist) + common + only_fine + only_pre))

    logger.info("HVG composition before trimming:")
    logger.info("  Whitelisted genes: %d", len(whitelist))
    logger.info("  Common to pretrain & finetune: %d", len(common))
    logger.info("  Finetune-only: %d", len(only_fine))
    logger.info("  Pretrain-only: %d", len(only_pre))

    if len(ordered) > n_final:
        # Retain whitelist, then top genes until budget exhausted
        extra = [g for g in ordered if g not in whitelist][: n_final - len(whitelist)]
        ordered = list(whitelist) + extra
        logger.info("Trimmed to %d genes (%d whitelist + %d other)", 
                   len(ordered), len(whitelist), len(extra))
    elif len(ordered) < n_final:
        logger.warning(
            "Final HVG list shorter (%d) than requested %d – pad with extra pretrain genes",
            len(ordered),
            n_final,
        )
    logger.info("Final HVG list size: %d", len(ordered))
    return ordered

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(args):
    import gc  # Import garbage collector
    
    whitelist = load_whitelist(args.whitelist_path)

    finetune = load_finetune(args.finetune_path, max_cells=args.max_cells_finetune)
    # Pretrain sampling – huge corpora → sample or stream
    pretrain = load_pretrain_sample(
        args.scrna_dir, max_cells=args.max_cells_pretrain
    )

    # Intersect gene sets to ensure shared ordering
    logger.info("Pretrain first 5 genes: %s", list(pretrain.var_names[:5]))
    logger.info("Finetune first 5 genes: %s", list(finetune.var_names[:5]))
    logger.info("Pretrain total genes: %d", len(pretrain.var_names))
    logger.info("Finetune total genes: %d", len(finetune.var_names))
    
    common_genes = pretrain.var_names.intersection(finetune.var_names)
    logger.info("Common genes retained: %d", len(common_genes))
    
    if len(common_genes) == 0:
        logger.error("No common genes found! Exiting.")
        return
    
    pretrain = pretrain[:, common_genes]
    finetune = finetune[:, common_genes]

    # Compute HVGs for pretrain
    hvgs_pre = compute_hvgs(pretrain, args.n_hvgs)
    
    # Clean up pretrain memory before computing finetune HVGs
    del pretrain
    gc.collect()
    
    # Compute HVGs for finetune
    hvgs_fine = compute_hvgs(finetune, args.n_hvgs)
    
    # Clean up finetune memory
    del finetune
    gc.collect()

    final_hvgs = merge_hvg_lists(
        hvgs_pre, hvgs_fine, whitelist, n_final=args.n_hvgs
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"hvg_seuratv3_{len(final_hvgs)}.txt"
    with open(out_file, "w") as fh:
        fh.write("\n".join(final_hvgs))
    logger.info("Written HVG list → %s", out_file)

# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified HVG selection (Seurat v3)")
    p.add_argument("--scrna_dir", default="data/scRNA/processed", help="Directory with batch_*.h5")
    p.add_argument("--finetune_path", default="data/vcc_data/adata_Training.h5ad", help="Finetune .h5ad file")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--whitelist_path", default="data/vcc_perturbed_genes.txt",help="Gene list to force‑include")
    p.add_argument("--n_hvgs", type=int, default=2000, help="#HVGs to keep")
    p.add_argument(
        "--max_cells_pretrain",
        type=int,
        default=1e5,
        help="Upper bound on sampled pre‑training cells to load",
    )
    p.add_argument(
        "--max_cells_finetune",
        type=int,
        default=1e6,  # Changed from 1_000_000 to 50_000
        help="Upper bound on sampled fine-tuning cells to load (default: 50k for memory efficiency)",
    )
    main(p.parse_args())
