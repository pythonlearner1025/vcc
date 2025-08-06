#!/usr/bin/env python3
"""Plot heat-map of per-gene attention scores.

Given the TSV produced by `export_attention.py`, this script
1. Loads the per-gene maximum attention scores.
2. Reads `gene_symbol2ensg.json` (mapping of priority symbols → ENSG IDs).
3. Adds 100 random additional HVGs (reproducible seed=42).
4. Builds a square matrix M where M[i,j] = (score_i + score_j) / 2.0 so that
   both rows and columns correspond to the same ordered gene list.
5. Plots M as a heat-map (seaborn/matplotlib), writing to *PNG*.

Priority genes occupy indices 0…len(priority)-1 on both axes, making it easy to
restrict the plot later (e.g. using a whitelist).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", default="attn.tsv", help="Per-gene attention TSV from export_attention.py")
    p.add_argument("--json", default="gene_symbol2ensg.json", help="Priority gene mapping JSON")
    p.add_argument("--out", default="heatmap.png", help="Output PNG path")
    p.add_argument("--random", type=int, default=100, help="Number of extra random genes")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(42)

    # Load TSV with attention scores -------------------------------------------------
    try:
        df = pd.read_csv(args.tsv, sep="\t")  # header present
        score_map = dict(zip(df["ensembl_id"], df["max_attention"]))
    except (ValueError, KeyError):
        # Fallback: file has no header
        df = pd.read_csv(
            args.tsv,
            sep="\t",
            header=None,
            names=["ensembl_id", "rank", "max_attention"],
        )
        score_map = dict(zip(df["ensembl_id"], df["max_attention"]))

    # Priority genes ---------------------------------------------------------------
    sym2ensg = json.load(open(args.json))
    priority_ensg: List[str] = [sym2ensg[sym] for sym in sym2ensg if sym2ensg[sym] in score_map]

    # Randomly draw additional genes not in priority list ---------------------------
    remaining = [gid for gid in score_map.keys() if gid not in priority_ensg]
    random_extra = rng.sample(remaining, k=min(args.random, len(remaining)))

    ordered_genes = priority_ensg + random_extra
    scores = np.array([score_map[g] for g in ordered_genes], dtype=np.float32)

    # Build symmetric matrix for heat-map -------------------------------------------
    mat = (scores[:, None] + scores[None, :]) / 2.0  # simple outer-average

    # Plot --------------------------------------------------------------------------
    sns.set(context="paper", style="white")
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(mat, cmap="viridis", square=True, cbar_kws={"label": "avg max-attention"})
    # Tick labels only for priority genes to avoid clutter
    labels = list(sym2ensg.keys())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Heat-map written to {args.out}")


if __name__ == "__main__":
    main()
