#!/usr/bin/env python3
"""Export per-HVG maximum attention (first transformer layer).

The script samples *N* random cells from the pre-training brain dataset,
passes them through the diffusion transformer, captures the first-layer
self-attention and writes a tab-separated table that can be used to plot
heat-maps or filter by gene whitelist later.

Output columns
--------------
1. ensembl_id   – HVG ENSG identifier (exact order of the 2 k HVG list)
2. rank         – 0-based HVG index
3. max_attn     – average over sampled cells of the token’s **maximum**
                  attention weight across all 12 heads and all source tokens.

Example usage
-------------
$ python export_attention.py -n 10000 > attn_first_layer.tsv

You can subsequently restrict the heat-map to a whitelist:
$ grep -F -f whitelist.txt attn_first_layer.tsv > subset.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Locations (model + config)
# -----------------------------------------------------------------------------
CKPT_DIR = Path("checkpoints/run_20250802_193251")
CFG_PATH = CKPT_DIR / "config.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_cfg() -> dict:
    if not CFG_PATH.exists():
        sys.exit("config.json not found – cannot continue")
    return json.load(open(CFG_PATH))


def _latest_ckpt() -> Path:
    pts = list(CKPT_DIR.glob("*.pt"))
    if not pts:
        sys.exit("No checkpoint files found in ckpt dir")
    return max(pts, key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))


def _hvg(cfg: dict) -> List[str]:
    hvg_path = Path(cfg["hvg_info_path"])
    if not hvg_path.exists():
        sys.exit(f"HVG file not found: {hvg_path}")
    with open(hvg_path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# -----------------------------------------------------------------------------
# Attention capture hook
# -----------------------------------------------------------------------------
class FirstLayerAttentionRecorder:
    """Forward hook that stores max-over-heads&src attention per token."""

    def __init__(self):
        self.values: List[torch.Tensor] = []  # (B,N) per batch

    def __call__(self, module, inp, out):  # noqa: ANN001
        # module is MultiQueryAttention; we need its internal attn
        x = inp[0]  # (B,N,D)
        B, N, _ = x.shape
        # Re-compute attention weights using module internals (no flash path).
        q = module.q_proj(x).view(B, N, module.n_head, module.head_dim)
        kv = module.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, N, 1, module.head_dim).expand(-1, -1, module.n_head, -1)
        v = v.view(B, N, 1, module.head_dim).expand(-1, -1, module.n_head, -1)
        q = q.permute(0, 2, 1, 3)  # B,H,N,HD
        k = k.permute(0, 2, 3, 1)  # B,H,HD,N
        scores = torch.matmul(q, k) / (module.head_dim**0.5)  # B,H,N,N
        attn = torch.softmax(scores, dim=-1)  # B,H,N,N
        # Max over heads & src tokens --> (B,N)
        max_attn = attn.max(dim=1).values.max(dim=-1).values  # (B,N)
        self.values.append(max_attn.detach().cpu())

    def reset(self):
        self.values.clear()

    def stacked(self) -> np.ndarray:
        if not self.values:
            return np.empty((0, 0), dtype=np.float32)
        return torch.cat(self.values, dim=0).numpy()


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--n", "-n", type=int, default=16000, help="Number of cells to sample (default 40k)")
    args = par.parse_args()

    cfg = _load_cfg()
    hvgs = _hvg(cfg)

    # Dataset ----------------------------------------------------------------
    from dataset.scrna_hvg_dataset import ScRNADatasetWithHVGs

    data_dir = Path(cfg.get("pretrain_data_dir") or cfg.get("data_path"))
    ds = ScRNADatasetWithHVGs(str(data_dir), hvgs, normalize=False, use_cache=True)
    if args.n > len(ds):
        sys.exit(f"Requested n={args.n} exceeds dataset size {len(ds)}")

    rng = np.random.default_rng(42)
    rows = rng.choice(len(ds), size=args.n, replace=False)
    X = np.stack([ds[i].numpy() for i in rows], dtype=np.float32)

    # Tokenise ----------------------------------------------------------------
    from ..tokenizer import create_logbin_tokenizer

    tokenizer, detokenizer = create_logbin_tokenizer(cfg["vocab_size"])
    tokens = tokenizer(X).long()

    # Model -------------------------------------------------------------------
    from models.diffusion import ConditionalDiffusionTransformer, ConditionalModelConfig
    ckpt = _latest_ckpt()
    model_cfg = ConditionalModelConfig(**cfg)
    model = ConditionalDiffusionTransformer(model_cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)

    # Disable flash-attention in layer 0 self-attn to get weights
    first_attn = model.blocks[0].self_attn
    first_attn.use_flash_attn = False

    recorder = FirstLayerAttentionRecorder()
    hook = first_attn.register_forward_hook(recorder)

    # Run in batches ----------------------------------------------------------
    bs = 32
    with torch.no_grad():
        for i in range(0, args.n, bs):
            tb = tokens[i : i + bs].to(device)
            model(tb, torch.zeros(tb.size(0), dtype=torch.long, device=device))

    hook.remove()
    attn_mat = recorder.stacked()  # (N,2000)

    # Average max-attention per gene -----------------------------------------
    gene_scores = attn_mat.mean(axis=0)  # (2000,)

    # Output TSV --------------------------------------------------------------
    print("ensembl_id\trank\tmax_attention")
    for idx, (gid, score) in enumerate(zip(hvgs, gene_scores)):
        print(f"{gid}\t{idx}\t{score:.6f}")


if __name__ == "__main__":
    main()
