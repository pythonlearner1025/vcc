#!/usr/bin/env python3
"""
UMI-level datasets for VAE pre-training and fine-tuning.

The loaders mirror the existing HVG-based datasets but retain the **full gene
count space** (no HVG filtering) because the VAE must learn a global latent
representation that can later be tokenised.

There are two flavours:
    1. `UMITrainLoader`    – unsupervised single-cell UMI counts used to fit
       the VAE.  Returns a **float tensor** of log-normalised counts
       (B, G).
    2. `UMIFinetuneLoader` – wrapper around the existing VCC paired dataset so
       the downstream diffusion model can keep using the same fields while the
       VAE sees raw counts.  If some optional keys are not present we skip
       them gracefully so that the DataLoader can reuse the current collator
       without modifications.

Both datasets apply *per-cell normalisation* to 10 000 counts and ``log1p`` in
`__getitem__` – thereby supporting random access without an upfront expensive
normalisation pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helper – on-the-fly normalisation ------------------------------------------------
# ---------------------------------------------------------------------------

def _cp10k_log1p(x: np.ndarray) -> np.ndarray:
    """CP10K normalisation followed by ``log1p`` – vectorised for a single cell."""
    # ensure float32 for numerical stability
    x = x.astype(np.float32, copy=False)
    libsize = x.sum() or 1.0  # avoid division by zero
    x = x / libsize * 1e4
    np.log1p(x, out=x)
    return x


# ---------------------------------------------------------------------------
# 1) Unsupervised single-cell loader (training the VAE) -----------------------
# ---------------------------------------------------------------------------

class UMITrainLoader(Dataset):
    """All single-cell runs used for VAE training.

    Expects a directory containing batch_*.h5 files (same layout as HVG loader)
    where each file has an `X` dataset holding the raw UMI counts (cells × genes).
    """

    def __init__(self, data_dir: str | Path, normalize: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_files = sorted(self.data_dir.glob("batch_*.h5"))
        self.normalize = normalize
        if not self.batch_files:
            raise FileNotFoundError(f"No batch_*.h5 files in {data_dir}")

        # determine global gene dimension + build cell index → (file_idx, local_idx)
        self._index: List[tuple[int, int]] = []
        for f_idx, h5_path in enumerate(self.batch_files):
            with h5py.File(h5_path, "r") as f:
                n_cells = f["X"].shape[0]
            self._index.extend([(f_idx, i) for i in range(n_cells)])

        # gene dimension
        with h5py.File(self.batch_files[0], "r") as f:
            self.n_genes = f["X"].shape[1]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._index[idx]
        with h5py.File(self.batch_files[file_idx], "r") as f:
            counts = f["X"][local_idx]  # (G,)
        if self.normalize: 
            counts = _cp10k_log1p(counts)
        return torch.from_numpy(counts).float()


# ---------------------------------------------------------------------------
# 2) Fine-tune loader – VCC paired sets with raw counts -----------------------
# ---------------------------------------------------------------------------

class UMIFinetuneLoader(Dataset):
    """Thin wrapper around `VCCPairedDataset` that exposes raw expression.

    It keeps all dictionary keys returned by `VCCPairedDataset.__getitem__` but
    replaces the expression matrices with CP10K-normalised log counts.
    """

    def __init__(
        self,
        vcc_dataset,  # instance of VCCPairedDataset
    ):
        self.ds = vcc_dataset

    # delegate --------------------------------------------------------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.ds[idx].copy()
        # normalise in-place (numpy then torch)
        for key in ("perturbed_expr", "control_expr"):
            if key not in sample:
                continue
            arr: torch.Tensor = sample[key]
            if torch.is_tensor(arr):
                np_arr = arr.numpy()
            else:
                np_arr = arr
            np_arr = np.array([_cp10k_log1p(row) for row in np_arr])
            sample[key] = torch.from_numpy(np_arr).float()
        return sample


# ---------------------------------------------------------------------------
# Optional collate function – paddings/truncation ----------------------------
# ---------------------------------------------------------------------------

def umi_collate(batch: List[torch.Tensor]):
    """Simple collation because all items have identical gene dimension."""
    return torch.stack(batch, dim=0)  # (B, G)
