#!/usr/bin/env python3
"""Minimal test for SYN/MT covariance encoding.

This script evaluates whether a diffusion-transformer model trained on brain
scRNA-seq implicitly captures the expected *synaptic vs mitochondrial* trade-off
observed in real data.

It compares the **sign** of three quantities between real and model-generated
expression on randomly sampled cells:

1. Mean pair-wise Spearman correlation within SYN genes
2. Mean pair-wise Spearman correlation within MT genes
3. Mean pair-wise Spearman correlation between SYN and MT genes

The final metric is the fraction of the above where the model's sign matches
the real data (range ∈ {0, 0.33, 0.67, 1}).

Usage
-----
$ python eval_covariance.py --n 40000   # defaults to 40 k if omitted
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Biology – hard-coded gene sets & (unused) log2FC dictionary
# -----------------------------------------------------------------------------
SYN_GENES: List[str] = [
    # unc-13 paralogues
    "UNC13A", "UNC13B", "UNC13C", "UNC13D",
    # Appl
    "APP",
    # brp paralogues
    "ERC2", "ERC1",
    # Nrx-1 paralogues
    "NRXN1", "NRXN2", "NRXN3",
    # Ank2 paralogues
    "ANK2", "ANK3",
    # Dscam4 paralogues
    "DSCAM", "DSCAML1",
]

MT_GENES: List[str] = [
    "UQCRC1",  # UQCR-C1
    "IFI30",   # CG41378
    "PLA2G3",  # CG3009
    "ERAP1", "LNPEP",  # CG4467
]

# Log2 fold-changes kept for completeness (not used here)
HUMAN_LOG2_FC: Dict[str, float] = {
    "UNC13A": -0.914260682,
    "UNC13B": -0.914260682,
    "UNC13C": -0.914260682,
    "UNC13D": -0.914260682,
    "APP": -0.957257636,
    "UQCRC1": 0.922456005,
    "PTPRN": -1.343366601,
    "PTPRN2": -1.343366601,
    "RGS7": -0.974478239,
    "RGS6": -0.974478239,
    "RGS9": -0.974478239,
    "RGS11": -0.974478239,
    "ERC2": -1.133742246,
    "ERC1": -1.133742246,
    "IFI30": 0.798824958,
    "NRXN1": -0.822721794,
    "NRXN2": -0.822721794,
    "NRXN3": -0.822721794,
    "ANK2": -0.726232487,
    "ANK3": -0.726232487,
    "DSCAM": -0.980582517,
    "DSCAML1": -0.980582517,
    "PLA2G3": 0.903135873,
    "TMPRSS2": -0.978279778,
    "TMPRSS4": -0.978279778,
    "SPEN": -0.557713325,
    "CHD7": -0.984824691,
    "CHD8": -0.984824691,
    "CACNA1A": -0.729046867,
    "ERAP1": -1.165815775,
    "LNPEP": -1.165815775,
    "CPEB1": -1.459195155,
    "PCBP1": -0.842228875,
    "PCBP2": -0.842228875,
    "PCBP3": -0.842228875,
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

CKPT_DIR = Path("checkpoints/run_20250802_193251")
CFG_PATH = CKPT_DIR / "config.json"


def _load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR   cannot find config.json at {path}")
    return json.load(open(path))


def _hvg_list(cfg: dict) -> List[str]:
    hvg_path = Path(cfg["hvg_info_path"])
    if not hvg_path.exists():
        sys.exit(f"ERROR   hvg file not found: {hvg_path}")
    with open(hvg_path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _find_latest_ckpt(ckpt_dir: Path) -> Path:
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        sys.exit("ERROR   no .pt checkpoint files found")
    latest = max(pts, key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
    return latest


def _avg_pairwise(mat: np.ndarray) -> float:
    """Average upper-triangle Spearman correlation (excluding diagonal)."""
    rho = spearmanr(mat, axis=0).correlation  # (G,G) or scalar
    if not hasattr(rho, "ndim") or rho.ndim != 2:  # Happens if <2 valid columns
        return float(rho) if np.isfinite(rho) else 0.0
    iu = np.triu_indices_from(rho, k=1)
    return float(np.nanmean(rho[iu]))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", "-n", type=int, default=50000, help="Number of cells to sample (default 40k)")
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    # 1) Config & HVGs -----------------------------------------------------
    cfg = _load_config(CFG_PATH)
    hvgs = _hvg_list(cfg)  # list of ENSG IDs

    # ------------------------------------------------------------------
    # Map gene SYMBOL -> HVG index using Ensembl REST (cached locally)
    # ------------------------------------------------------------------
    cache_path = Path("gene_symbol2ensg.json")
    if cache_path.exists():
        symbol2ensg: Dict[str, str] = json.load(open(cache_path))
    else:
        symbol2ensg = {}

    def _lookup(symbol: str) -> str:
        if symbol in symbol2ensg:
            return symbol2ensg[symbol]
        import requests
        url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol}"
        resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        if not resp.ok or not resp.json():
            sys.exit(f"ERROR   failed to look-up ENSG ID for {symbol}")
        for entry in resp.json():
            if entry.get("type") == "gene" and entry["id"].startswith("ENSG"):
                ensg = entry["id"].split(".")[0]
                symbol2ensg[symbol] = ensg
                return ensg
        sys.exit(f"ERROR   no ENSG ID found for {symbol}")

    # Build index mapping
    gene2idx: Dict[str, int] = {}
    missing: List[str] = []
    for sym in SYN_GENES + MT_GENES:
        ensg = _lookup(sym)
        if ensg not in hvgs:
            missing.append(sym)
            continue  # skip genes outside the 2 k HVG list
        gene2idx[sym] = hvgs.index(ensg)

    if missing:
        print(
            f"WARNING  {len(missing)} gene(s) not in HVGs and will be ignored: "
            + ", ".join(missing)
        )

    # Require at least two genes per set so a correlation is defined
    syn_idx = [gene2idx[g] for g in SYN_GENES if g in gene2idx]
    mt_idx = [gene2idx[g] for g in MT_GENES if g in gene2idx]
    if len(syn_idx) < 2 or len(mt_idx) < 2:
        sys.exit("ERROR   fewer than two SYN or MT genes left after filtering – aborting")

    # Persist cache (non-critical)
    try:
        with open(cache_path, "w") as fh:
            json.dump(symbol2ensg, fh)
    except Exception:
        pass

    syn_idx = [gene2idx[g] for g in SYN_GENES]
    mt_idx = [gene2idx[g] for g in MT_GENES]

    # 2) Sample expression counts ----------------------------------------
    from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs

    data_dir = Path(cfg.get("pretrain_data_dir") or cfg.get("data_path"))
    if not data_dir.exists():
        sys.exit(f"ERROR   data directory not found: {data_dir}")

    ds = ScRNADatasetWithHVGs(
        data_dir=str(data_dir),
        hvg_genes=hvgs,
        normalize=False,
        use_cache=True,
    )

    if args.n > len(ds):
        sys.exit(f"ERROR   requested n={args.n} exceeds dataset size {len(ds)}")

    rows = rng.choice(len(ds), size=args.n, replace=False)
    X = np.stack([ds[i].numpy() for i in rows], dtype=np.float32)  # (N,2000)

    # 3) Real correlations ------------------------------------------------
    real_syn = X[:, syn_idx]
    real_mt = X[:, mt_idx]

    real_within_syn = _avg_pairwise(real_syn)
    real_within_mt = _avg_pairwise(real_mt)
    real_between = float(np.nanmean(spearmanr(real_syn, real_mt, axis=0).correlation))

    # 4) Model inference --------------------------------------------------
    from ..tokenizer import create_delta_tokenizer
    from models.diffusion import ConditionalDiffusionTransformer, ConditionalModelConfig

    tokenizer, detokenizer = create_delta_tokenizer(cfg["vocab_size"])
    # For covariance eval, treat input X as baseline (control) and mask SYN/MT positions to predict Δ, then reconstruct
    tokens_ctrl = tokenizer(np.zeros_like(X)).long()
    tokens = tokens_ctrl.clone()
    mask_token = cfg["vocab_size"] - 1
    for idx in syn_idx + mt_idx:
        tokens[:, idx] = mask_token

    model_cfg = ConditionalModelConfig(**cfg)
    model = ConditionalDiffusionTransformer(model_cfg)
    ckpt_path = _find_latest_ckpt(CKPT_DIR)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)

    vocab_size = cfg["vocab_size"]
    # Build expected Δ per token (pad mask id with last bin centre)
    detok_vec = detokenizer(torch.arange(vocab_size - 1))  # (V-1,)
    detok_vec = torch.cat([detok_vec, detok_vec[-1:].clone()])


    batch = 32
    pred_expr = np.empty_like(X)
    with torch.no_grad():
        for i in range(0, args.n, batch):
            tb = tokens[i : i + batch].to(device)
            logits = model(tb, torch.zeros(tb.size(0), dtype=torch.long, device=device))
            probs = torch.softmax(logits, dim=-1)  # (B,N,V)
            delta_vals = torch.einsum("bnv,v->bn", probs, detok_vec.to(device))
            pred_expr[i : i + batch] = delta_vals.cpu().numpy()

    pred_syn = pred_expr[:, syn_idx]
    pred_mt = pred_expr[:, mt_idx]

    pred_within_syn = _avg_pairwise(pred_syn)
    pred_within_mt = _avg_pairwise(pred_mt)
    pred_between = float(np.nanmean(spearmanr(pred_syn, pred_mt, axis=0).correlation))

    # 5) Sign-match score --------------------------------------------------
    def _sgn(x: float) -> int:
        return int(np.sign(x))

    matches = [
        _sgn(real_within_syn) == _sgn(pred_within_syn),
        _sgn(real_within_mt) == _sgn(pred_within_mt),
        _sgn(real_between) == _sgn(pred_between),
    ]
    score = np.mean(matches)

    # 6) Report -----------------------------------------------------------
    def _fmt(x: float) -> str:
        return f"{x:+.2f}"

    print(f"N cells evaluated   : {args.n}")
    print("=================== REAL  ===================")
    print(f"Within-SYN ρ¯       : {_fmt(real_within_syn)}")
    print(f"Within-MT  ρ¯       : {_fmt(real_within_mt)}")
    print(f"Between    ρ¯       : {_fmt(real_between)}")
    print("================  MODEL  ====================")
    print(f"Within-SYN ρ¯       : {_fmt(pred_within_syn)}   (sign match {'✔' if matches[0] else '✘'})")
    print(f"Within-MT  ρ¯       : {_fmt(pred_within_mt)}   (sign match {'✔' if matches[1] else '✘'})")
    print(f"Between    ρ¯       : {_fmt(pred_between)}   (sign match {'✔' if matches[2] else '✘'})")
    print("--------------------------------------------")
    print(f"Sign-match score    : {score:.2f}")


if __name__ == "__main__":
    main()
