#!/usr/bin/env python3
"""
Precompute HVG-pruned arrays for scRNA batches to an external cache.

For each input HDF5 batch with datasets:
  - X: (cells, n_genes)
  - genes: (n_genes,) byte/string gene IDs (Ensembl)

This script writes per-batch files to a cache directory (no in-place writes):
  - {cache_dir}/{batch_stem}.npy: (cells, n_hvgs) float16 in the supplied HVG order
  - {cache_dir}/hvg_genes.txt: HVG list used
  - {cache_dir}/meta.json: metadata
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


def read_hvgs(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def compute_hvg_for_file(batch_path: Path, hvgs: List[str]) -> Tuple[np.ndarray, List[str]]:
    with h5py.File(batch_path, 'r') as f:
        genes = [g.decode('utf-8') if isinstance(g, (bytes, np.bytes_)) else str(g) for g in f['genes'][:]]
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        present_hvgs = [g for g in hvgs if g in gene_to_idx]
        X = f['X']
        rows = X.shape[0]
        # Build column indices in the supplied HVG order
        col_idx = np.array([gene_to_idx[g] for g in present_hvgs], dtype=np.int64)
        # Copy in chunks of rows to limit peak memory
        step = 8192
        out = np.empty((rows, len(col_idx)), dtype=np.float16)
        for start in range(0, rows, step):
            end = min(rows, start + step)
            order = np.argsort(col_idx)
            sorted_idx = col_idx[order]
            slab_sorted = X[start:end, sorted_idx]
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            slab = slab_sorted[:, inv]
            out[start:end, :] = slab.astype(np.float16, copy=False)
    return out, present_hvgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Path to scRNA processed dir containing batch_*.h5 and config.json')
    ap.add_argument('--hvg_info_path', required=True, help='Path to HVG list file (one Ensembl ID per line)')
    ap.add_argument('--out_dir', default=None, help='Optional output cache dir; default: <data_dir>/../hvg_cache/<hvg_tag>')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    hvgs = read_hvgs(args.hvg_info_path)
    files = sorted(data_dir.glob('batch_*.h5'))
    if not files:
        raise SystemExit(f"No batch_*.h5 files found under {data_dir}")
    # Tag cache dir by HVG list hash to avoid collisions when HVGs change
    import hashlib, json as _json
    tag = hashlib.sha1('\n'.join(hvgs).encode('utf-8')).hexdigest()[:12]
    out_root = Path(args.out_dir) if args.out_dir else (data_dir.parent / 'hvg_cache' / tag)
    out_root.mkdir(parents=True, exist_ok=True)
    # Save metadata
    (out_root / 'hvg_genes.txt').write_text('\n'.join(hvgs))
    meta = {'hvg_tag': tag, 'num_hvgs': len(hvgs), 'source_dir': str(data_dir)}
    (out_root / 'meta.json').write_text(_json.dumps(meta, indent=2))
    for f in files:
        out_path = out_root / (f.stem + '.npy')
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        arr, present = compute_hvg_for_file(f, hvgs)
        np.save(out_path, arr)
        print(f"[ok] wrote {out_path.name}: {arr.shape[0]} x {arr.shape[1]}")


if __name__ == '__main__':
    main()

