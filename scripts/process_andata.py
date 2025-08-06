#!/usr/bin/env python3
"""
process_andata.py

Parallel-safe, low-memory processing of a huge backed .h5ad file
into ≤ chunk_size cell batches.

Changes vs previous attempt
---------------------------
* Workers receive only picklable objects (no functions, no h5py handles).
* Symbol→Ensembl dict, orig_vars and new_vars are passed as plain data.
* Pandas `groupby` called with observed=False to silence future warning.
"""

from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import scanpy as sc
import anndata as ad
from joblib import Parallel, delayed


# ---------- simple helpers ---------- #

def load_lookup(path: Path) -> Dict[str, str]:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as fh:
        return pickle.load(fh)


def detect_control_label(labels: np.ndarray, explicit: str | None) -> str:
    if explicit:
        return explicit
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    for lab, _ in sorted(counts.items(), key=lambda x: -x[1]):
        low = lab.lower()
        if any(tok in low for tok in ("non", "control", "nt")):
            return lab
    return max(counts, key=counts.get)


def write_json(path: Path, **payload):
    path.write_text(json.dumps(payload, indent=2))


# ---------- worker ---------- #

def process_batch(input_h5ad: str,
                  out_dir: str,
                  idx_list: List[int],
                  gts: List[str],
                  orig_vars: List[str],
                  new_vars: List[str],
                  bid: int):
    """
    *All* arguments are plain Python objects ⇒ picklable.
    """
    A = sc.read_h5ad(input_h5ad, backed="r")
    chunk: ad.AnnData = A[np.asarray(idx_list, dtype=np.int64)].to_memory()

    chunk.var["gene_symbol"] = orig_vars
    chunk.var_names = new_vars

    sc.pp.normalize_total(chunk, target_sum=50000, inplace=True)
    sc.pp.log1p(chunk)
    chunk.X = chunk.X.astype(np.float32)

    dest = Path(out_dir) / f"batch_{bid:04d}.h5ad"
    chunk.write(dest, compression="gzip", compression_opts=1)

    write_json(dest.with_suffix(".json"),
               file=str(dest),
               gene_targets=gts,
               n_cells=int(chunk.n_obs))

    print(f"[{bid:04d}] {len(idx_list):,} cells → {dest}")


# ---------- main ---------- #

def main():
    ap = argparse.ArgumentParser(description="Optimised AnnData batching")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--lookup", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--chunk_size", default=5000, type=int)
    ap.add_argument("--n_jobs", default=1, type=int)
    ap.add_argument("--control_label", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    symbol2ens = load_lookup(args.lookup)
    print(f"Loaded {len(symbol2ens):,} symbol→Ensembl mappings.")
    A = sc.read_h5ad(args.input, backed="r")

    ctrl = detect_control_label(A.obs["gene_target"].values, args.control_label)
    print(f"Using '{ctrl}' as non-targeting control label.")

    # one-time gene-id mapping
    orig_vars = A.var_names.astype(str).tolist()
    new_vars = [v if v.startswith("ENS") else symbol2ens.get(v, v)
                for v in orig_vars]

    # build batches
    groups = A.obs.groupby("gene_target", observed=False).indices

    # --- 1) emit control-cell-only batches first --------------------------
    ctrl_idx = groups.pop(ctrl)  # remove control group from further processing
    ctrl_ids = ctrl_idx.tolist()
    batches = [
        (ctrl_ids[i:i + args.chunk_size], [ctrl])
        for i in range(0, len(ctrl_ids), args.chunk_size)
    ]

    # --- 2) fill remaining batches with the other gene targets ------------
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))  # largest first

    cur_idx, cur_gts = [], []
    for gt, idx in ordered:
        ids = idx.tolist()
        if cur_idx and len(cur_idx) + len(ids) > args.chunk_size:
            batches.append((cur_idx, cur_gts))
            cur_idx, cur_gts = [], []
        cur_idx.extend(ids)
        cur_gts.append(gt)
    if cur_idx:
        batches.append((cur_idx, cur_gts))

    print(f"Planned {len(batches)} batch file(s) of ≤ {args.chunk_size:,} cells each.")

    # kick off workers
    Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_batch)(
            str(args.input), str(args.out_dir),
            idxs, gts, orig_vars, new_vars, bid
        )
        for bid, (idxs, gts) in enumerate(batches)
    )

    print("All done.")

if __name__ == "__main__":
    main()
