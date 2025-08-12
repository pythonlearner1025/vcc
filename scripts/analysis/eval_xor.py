#!/usr/bin/env python3
"""Evaluate RBFOX3 ⊕ GFAP XOR task.

Usage
-----
$ python eval_xor.py --n 50000  [--q 0.8]

The script prints balanced accuracy, AUROC and confusion matrix for a random
subset of single-cell expression profiles using the latest diffusion
transformer checkpoint found in
`checkpoints/run_20250802_193251/`.

The implementation intentionally re-uses existing data/model utilities to avoid
duplicate I/O and boiler-plate.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

# -----------------------------------------------------------------------------
# Constants – ENSG identifiers for the two target genes
# -----------------------------------------------------------------------------
GFAP_ID = "ENSG00000131095"
RBFOX3_ID = "ENSG00000143842"  # NeuN / Fox-3

CKPT_DIR = Path("checkpoints/run_20250802_193251")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_hvg_list(path: Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {ckpt_dir}")

    def _step(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(pts, key=_step)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "-n", type=int, required=True, help="Number of cells to sample")
    parser.add_argument("--q", "-q", type=float, default=0.6, help="Quantile threshold (default 0.8)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------------
    # 1. Load training/config information
    # ---------------------------------------------------------------------
    cfg_path = CKPT_DIR / "config.json"
    cfg_dict = json.load(open(cfg_path))

    data_dir = Path(cfg_dict.get("pretrain_data_dir") or cfg_dict.get("data_path"))
    if not data_dir.exists():
        raise FileNotFoundError(f"Pre-training data directory not found: {data_dir}")

    hvg_path = Path(cfg_dict.get("hvg_info_path"))
    hvg = _load_hvg_list(hvg_path)

    try:
        idx_rbfox3 = hvg.index(RBFOX3_ID)
        idx_gfap = hvg.index(GFAP_ID)
    except ValueError as e:
        raise RuntimeError("Required gene not present in HVG list – aborting") from e

    # ---------------------------------------------------------------------
    # 2. Sample expression matrix (N, 2000) with existing dataset loader
    # ---------------------------------------------------------------------
    from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs

    ds = ScRNADatasetWithHVGs(
        data_dir=str(data_dir),
        hvg_genes=hvg,
        normalize=False,
        use_cache=True,
    )

    if args.n > len(ds):
        raise ValueError(f"Requested n={args.n} exceeds dataset size {len(ds)}")

    indices = rng.choice(len(ds), size=args.n, replace=False)

    expr = np.stack([ds[i].numpy() for i in indices], dtype=np.float32)
    # (N, 2000)

    # ---------------------------------------------------------------------
    # 3. Binarise expression for ground-truth XOR
    # ---------------------------------------------------------------------
    def _binarise(col: np.ndarray, q: float) -> np.ndarray:
        nz = col[col > 0]
        tau = np.quantile(nz, q) if nz.size else 0.0
        return (col > tau).astype(np.uint8)

    x_bin = _binarise(expr[:, idx_rbfox3], args.q)
    y_bin = _binarise(expr[:, idx_gfap], args.q)
    y_true = np.bitwise_xor(x_bin, y_bin)

    # ---------------------------------------------------------------------
    # 4. Tokenise counts and mask the two genes
    # ---------------------------------------------------------------------
    from ..tokenizer import create_delta_tokenizer

    tokenizer, _ = create_delta_tokenizer(cfg_dict["vocab_size"])
    # Use zero baseline (control) and mask target genes to predict Δ
    tokens = tokenizer(np.zeros_like(expr)).long()  # (N, 2000)
    mask_token = cfg_dict["vocab_size"] - 1
    tokens[:, idx_rbfox3] = mask_token
    tokens[:, idx_gfap] = mask_token

    # ---------------------------------------------------------------------
    # 5. Instantiate model and load latest checkpoint
    # ---------------------------------------------------------------------
    from models.diffusion import ConditionalDiffusionTransformer, ConditionalModelConfig

    model_cfg = ConditionalModelConfig(**cfg_dict)
    model = ConditionalDiffusionTransformer(model_cfg)
    ckpt_path = _find_latest_checkpoint(CKPT_DIR)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)

    # ---------------------------------------------------------------------
    # 6. Run inference in mini-batches
    # ---------------------------------------------------------------------
    high_token = mask_token - 1  # highest expression bin
    batch_size = 32
    probs_rb = np.empty(args.n, dtype=np.float32)
    probs_gf = np.empty(args.n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, args.n, batch_size):
            end = min(start + batch_size, args.n)
            tb = tokens[start:end].to(device)
            timesteps = torch.zeros(tb.size(0), dtype=torch.long, device=device)
            logits = model(tb, timesteps)
            p = torch.softmax(logits, dim=-1)
            probs_rb[start:end] = p[:, idx_rbfox3, high_token].cpu().numpy()
            probs_gf[start:end] = p[:, idx_gfap, high_token].cpu().numpy()

    p_xor = probs_rb * (1 - probs_gf) + probs_gf * (1 - probs_rb)
    y_pred = (p_xor >= 0.5).astype(np.uint8)

    # ---------------------------------------------------------------------
    # 7. Metrics
    # ---------------------------------------------------------------------
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, p_xor) if len(np.unique(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred)

    # ---------------------------------------------------------------------
    # 8. Reporting
    # ---------------------------------------------------------------------
    print(f"N cells evaluated     : {args.n}")
    print(f"Balanced accuracy     : {bal_acc:.3f}")
    print(f"AUROC                 : {auroc:.3f}")
    print("Confusion matrix")
    print(cm)
