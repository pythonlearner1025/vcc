#!/usr/bin/env python3
"""
Evaluation script for a Conditional Discrete Diffusion Transformer trained with
`train.py` on the VCC dataset.

The script runs two evaluation phases:
1. Validation  – compare model predictions against the held-out validation set
   used during fine-tuning.  A separate prediction/ground-truth `.h5ad` file is
   written per validation gene and the *cell-eval* toolkit is invoked to compute
   the full suite of metrics.
2. Test        – generate predictions for a list of unseen perturbation genes
   (no ground truth available).  All generated cells are concatenated into a
   single `.h5ad` file ready for submission / downstream analysis.

Outputs are organised under a timestamped folder inside the project-level
`evals/` directory, e.g. `evals/eval_20250728_123456/`.

Typical usage
-------------
$ python evaluate.py \
      --ckpt_dir checkpoints/run_20250727_181500 \
      --val_genes_number 32 \
      --val_set_size 16 \
      --test_genes_path data/vcc_data/pert_counts_Validation.csv \
      --test_sample_size 64 \
      --test_gene_names_path data/vcc_data/gene_names.csv \
      --hvg_ids_path hvg_seuratv3_2000.txt

Requirements
------------
- The *cell-eval* package must be installed in the current environment.
  `pip install cell-eval` or see README in the repository.
- GPU with enough memory to sample `set_size × vocab_size` tokens in one go.
- All dataset paths from the original training run must still be valid.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
import torch

# -----------------------------------------------------------------------------
# Local imports – reuse code from the training script and model package
# -----------------------------------------------------------------------------

from train import create_simple_tokenizer  # noqa: E402  – defined at top-level
from models.diffusion import (
    ConditionalDiffusionTransformer,
    ConditionalModelConfig,
    PartialMaskingDiffusion,
)
from dataset.vcc_paired_dataloader import (
    create_train_val_dataloaders,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _load_checkpoint(ckpt_dir: Path, device: torch.device):
    # 1. read config ---------------------------------------------------------
    cfg_dict = json.load(open(ckpt_dir / "config.json"))
    config = ConditionalModelConfig(**cfg_dict)

    # 2. build model **on the CPU**
    model = ConditionalDiffusionTransformer(config).cpu()   # <-- no .to(device)

    # 3. load weights on CPU
    ckpt_path = ckpt_dir / "checkpoint_st_final.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)

    # 4. only now move the whole model in one go
    model = model.to(device, non_blocking=True)
    model.eval().requires_grad_(False)

    diffusion = PartialMaskingDiffusion(config)
    return config, model, diffusion

def _prepare_gene_mappings(
    adata_path: Path, hvg_ids_path: Path
) -> tuple[List[str], np.ndarray, List[str]]:
    """Return `(gene_names_18k, hvg_idx→gene_name_idx)`.

    This avoids loading the massive `gene_names.csv` directly by relying on the
    AnnData object from the VCC dataset which already contains gene-symbol ↔︎
    Ensembl-ID mappings.
    """

    adata_path = Path(adata_path).expanduser()
    print(f"Reading VCC AnnData from {adata_path} (only .var loaded)")
    adata = sc.read_h5ad(adata_path, backed="r")

    # Gene symbols are the AnnData var index, Ensembl IDs in `gene_id` column
    gene_names = list(adata.var_names)
    gene_name_to_idx: Dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    gene_id_to_name: Dict[str, str] = {}
    for gid, gsym in zip(adata.var["gene_id"].values, adata.var.index.values):
        base_gid = gid.split(".")[0] if isinstance(gid, str) else gid
        gene_id_to_name[base_gid] = gsym

    # HVG → full-gene-list index mapping ------------------------------------
    with open(hvg_ids_path) as f:
        hvg_gene_ids = [ln.strip() for ln in f if ln.strip()]

    hvg_to_full = []
    missing = 0
    for gid in hvg_gene_ids:
        gname = gene_id_to_name.get(gid)
        if gname is None:
            missing += 1
            hvg_to_full.append(-1)
        else:
            hvg_to_full.append(gene_name_to_idx[gname])
    if missing:
        print(f"WARNING: {missing}/{len(hvg_gene_ids)} HVG IDs not found in AnnData")
    return gene_names, np.asarray(hvg_to_full, dtype=np.int32), hvg_gene_ids


def _tokens_to_expression(tokens: torch.LongTensor, detokenizer) -> np.ndarray:
    """Convert token tensor (B,N) to numpy float32 (B,N)."""
    with torch.no_grad():
        vals = detokenizer(tokens.cpu())  # (B,N)
    return vals.numpy().astype(np.float32)


def _build_18k_matrix(
    expr_2k: np.ndarray, hvg_to_full: np.ndarray, n_full: int
) -> np.ndarray:
    """Insert HVG expression into zero-initialised 18,080-gene matrix."""
    B, N = expr_2k.shape
    out = np.zeros((B, n_full), dtype=np.float32)
    # since if gname not in hvg, its -1 
    valid_mask = hvg_to_full >= 0
    out[:, hvg_to_full[valid_mask]] = expr_2k[:, valid_mask]
    return out


def _write_anndata(
    X: np.ndarray,
    var_names: Sequence[str],
    pert_label: str,
    out_path: Path,
):
    """Save expression matrix **X** to `out_path` as AnnData (csr compressed)."""
    import pandas as pd

    adata = ad.AnnData(
        X,
        obs=pd.DataFrame({"target_gene": [pert_label] * X.shape[0]}),
        var=pd.DataFrame(index=list(var_names)),
    )
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Wrote {out_path.name}  shape={X.shape}")


# -----------------------------------------------------------------------------
# Validation evaluation
# -----------------------------------------------------------------------------

def _run_validation_merged(
    cfg: ConditionalModelConfig,
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    tokenizer,
    detokenizer,
    gene_names: List[str],
    hvg_to_full: np.ndarray,
    hvg_gene_ids: List[str],
    args,
    device: torch.device,
    out_dir: Path,
):
    """Generate one merged prediction/ground-truth pair for all
    validation genes and run cell-eval once."""
    import pandas as pd
    mask_token = cfg.vocab_size - 1

    (_, _), (val_ds, val_dl) = create_train_val_dataloaders(
        adata_path=cfg.finetune_data_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=args.val_set_size,
        batch_size=1,
        n_samples_per_gene_val=1,
        train_split=0.8,
        tokenizer=None,
        num_workers=0,
        normalize=False,
        blacklist_path=args.blacklist_path,
    )

    X_true_parts = []
    X_pred_parts = []
    obs_labels: list[str] = []

    for sample in val_dl:
        pert_expr = sample["perturbed_expr"].squeeze(0)  # (S,N)
        ctrl_expr = sample["control_expr"].squeeze(0)
        gene_name: str = sample["target_gene"][0]
        gene_idx = int(sample.get("gene_idx", sample.get("target_gene_idx")).item())
        batches = sample["pert_batches"]
        S, N = pert_expr.shape

        # Tokenise
        tokens_pert = tokenizer(pert_expr).to(device)
        tokens_ctrl = tokenizer(ctrl_expr).to(device)
        ctrl_expanded = tokens_ctrl.unsqueeze(0).expand(S, -1, -1)

        batch_to_idx = val_ds.batch_to_idx
        batch_idx = torch.tensor([batch_to_idx.get(b, 0) for b in batches], dtype=torch.long, device=device)
        target_gene_idx = torch.full((S,), gene_idx, dtype=torch.long, device=device)

        pred_tokens = diffusion.p_sample_loop(
            model,
            mask_token=mask_token,
            shape=(S, cfg.n_genes),
            control_set=ctrl_expanded,
            target_gene_idx=target_gene_idx,
            batch_idx=batch_idx,
            device=device,
        )

        # Convert tokens → expression.  For Δ-training we need to add the predicted
        # change back to the control-cell baseline.
        if cfg.target_is_delta:
            pred_delta_2k = _tokens_to_expression(pred_tokens, detokenizer)
            pred_expr_2k = np.clip(ctrl_expr.numpy() + pred_delta_2k, 0, None)
        else:
            pred_expr_2k = _tokens_to_expression(pred_tokens, detokenizer)
        true_expr_2k = pert_expr.numpy()

        # Map to 18 k space
        pred_full = _build_18k_matrix(pred_expr_2k, hvg_to_full, len(gene_names))
        true_full = _build_18k_matrix(true_expr_2k, hvg_to_full, len(gene_names))
        ctrl_full = _build_18k_matrix(ctrl_expr.numpy(), hvg_to_full, len(gene_names))

        X_true_parts.extend([ctrl_full, true_full])
        X_pred_parts.extend([ctrl_full, pred_full])
        obs_labels.extend(["non-targeting"] * S + [gene_name] * S)

        if len(X_true_parts) // 2 >= args.val_genes_number:
            break

    X_true = np.concatenate(X_true_parts, axis=0)
    X_pred = np.concatenate(X_pred_parts, axis=0)
    obs_df = pd.DataFrame({"target_gene": obs_labels})

    pred_path = (out_dir / "val_pred.h5ad").resolve()
    true_path = (out_dir / "val_true.h5ad").resolve()
    ad.AnnData(X_pred, obs=obs_df.copy(), var=pd.DataFrame(index=gene_names)).write_h5ad(pred_path, compression="gzip")
    ad.AnnData(X_true, obs=obs_df.copy(), var=pd.DataFrame(index=gene_names)).write_h5ad(true_path, compression="gzip")

    print("Running cell-eval on merged validation set…")
    cmd = [
        "cell-eval", "run",
        "-ap", str(pred_path),
        "-ar", str(true_path),
        "--num-threads", "8",
        "--profile", "full",
    ]
    subprocess.run(cmd, check=True)
    # Move cell-eval results to correct output directory
    import shutil
    if Path("cell-eval-outdir").exists():
        cell_eval_dest = out_dir / "cell-eval-outdir"
        if cell_eval_dest.exists():
            shutil.rmtree(cell_eval_dest)
        shutil.move("cell-eval-outdir", cell_eval_dest)
    print(f"Validation complete – results in {out_dir}/cell-eval-outdir/")


# -----------------------------------------------------------------------------
# Test-set generation
# -----------------------------------------------------------------------------

def _run_test_generation(
    cfg: ConditionalModelConfig,
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    detokenizer,
    gene_names: List[str],
    hvg_to_full: np.ndarray,
    hvg_gene_ids: List[str],
    args,
    device: torch.device,
    out_dir: Path,
):
    import pandas as pd

    test_df = pd.read_csv(args.test_genes_path)
    # ---------------------------------------------------------------------
    # Build full gene list **including** the negative control label
    # ---------------------------------------------------------------------
    target_genes = test_df["target_gene"].values.tolist()
    print(f"Generating predictions for {len(target_genes)} test genes (incl. non-targeting)")
    mask_token = cfg.vocab_size - 1

    # ---------------------------------------------------------------------
    # Build helper objects: tokenizer and control-cell iterator
    # ---------------------------------------------------------------------
    from train import create_simple_tokenizer  # local import to avoid circular deps
    import itertools

    tokenizer, _ = create_simple_tokenizer(cfg.vocab_size)

    # Re-use the paired dataloader to fetch real control cells (non-targeting)
    (_, _), (val_ds, val_dl) = create_train_val_dataloaders(
        adata_path=cfg.finetune_data_path,
        hvg_gene_ids=hvg_gene_ids,
        set_size=cfg.vcc_set_size,
        batch_size=1,
        n_samples_per_gene_val=1,
        train_split=0.8,
        tokenizer=None,
        num_workers=0,
        normalize=False,
        blacklist_path=args.blacklist_path,
    )
    # Endless iterator cycling over validation samples
    control_cycle = itertools.cycle(val_dl)

    def _next_control_expr():
        """Return a control-expression matrix (S,N) from a non-targeting sample."""
        for sample in control_cycle:
            # sample["control_expr"] has shape (1,S,N)
            if sample["target_gene"][0] == "non-targeting":
                return sample["control_expr"].squeeze(0)
        # Should never happen but keeps mypy happy
        raise RuntimeError("Could not find non-targeting control sample")

    all_expr_full = []
    all_obs_labels = []

    # We re-use gene-name→HVG idx mapping from val_ds (by Ensembl id)
    # Build name→idx (HVG)
    gene_name_to_hvg_idx: Dict[str, int] = {}
    for i, name_idx in enumerate(hvg_to_full):
        if name_idx >= 0:
            gene_name_to_hvg_idx[gene_names[name_idx]] = i

    for gene in target_genes:
        # -----------------------------------------------------------------
        # Special handling for the negative-control class.  For evaluation
        # purposes we only need the label to be present – but to remain
        # consistent with validation we output **real control cells**,
        # sampled from the dataset via the helper iterator defined above.
        # -----------------------------------------------------------------
        if gene == "non-targeting":
            S = cfg.vcc_set_size
            n_needed = args.test_sample_size
            n_batches = math.ceil(n_needed / S)
            for _ in range(n_batches):
                ctrl_expr = _next_control_expr()  # (S,N)
                expr_full = _build_18k_matrix(ctrl_expr.numpy(), hvg_to_full, len(gene_names))

                take = min(S, n_needed - len(all_obs_labels))
                all_expr_full.append(expr_full[:take])
                all_obs_labels.extend(["non-targeting"] * take)
            continue

        if gene not in gene_name_to_hvg_idx:
            print(f"WARNING  gene {gene} not part of HVG list – skipping")
            continue

        hvg_idx = gene_name_to_hvg_idx[gene]

        # How many batches of size S to produce?
        S = cfg.vcc_set_size
        n_needed = args.test_sample_size
        n_batches = math.ceil(n_needed / S)

        for _ in range(n_batches):
            # Fetch a fresh control set (sampled from non-targeting cells)
            ctrl_expr = _next_control_expr()  # (S,N)
            tokens_ctrl = tokenizer(ctrl_expr).to(device)
            ctrl_expanded = tokens_ctrl.unsqueeze(0).expand(S, -1, -1)

            # Always sample with batch size S to match training, then truncate later
            pred_tokens = diffusion.p_sample_loop(
                model,
                mask_token=mask_token,
                shape=(S, cfg.n_genes),
                control_set=ctrl_expanded,
                target_gene_idx=torch.full((S,), hvg_idx, dtype=torch.long, device=device),
                batch_idx=torch.zeros(S, dtype=torch.long, device=device),
                device=device,
            )
            expr_2k = _tokens_to_expression(pred_tokens, detokenizer)  # (S,N)
            expr_full = _build_18k_matrix(expr_2k, hvg_to_full, len(gene_names))

            # If final batch overshoots, truncate
            take = min(S, n_needed - len(all_obs_labels))
            all_expr_full.append(expr_full[:take])
            all_obs_labels.extend([gene] * take)
        # end for batches
    # end for gene

    if not all_expr_full:
        print("No test predictions generated – aborting")
        return

    X = np.concatenate(all_expr_full, axis=0)

    import pandas as pd

    adata_pred = ad.AnnData(
        X,
        obs=pd.DataFrame({"target_gene": all_obs_labels}),
        var=pd.DataFrame(index=gene_names),
    )

    out_path = out_dir / "test_predictions.h5ad"
    adata_pred.write_h5ad(out_path, compression="gzip")
    print(f"Saved aggregated test predictions → {out_path}")


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate VCC diffusion model")
    p.add_argument("--ckpt_dir", default="/run_20250731_072803", type=Path)
    p.add_argument("--val_genes_number", type=int, default=10)
    p.add_argument("--val_set_size", type=int, default=8)
    p.add_argument("--test_genes_path", default="data/vcc_data/pert_counts_Validation.csv", type=Path)
    p.add_argument("--test_sample_size", type=int, default=64)
    p.add_argument("--test_gene_names_path", default="data/vcc_data/gene_names.csv", type=Path)
    p.add_argument("--hvg_ids_path", default="hvg_seuratv3_global_scRNA1e5_2000.txt", type=Path)
    p.add_argument("--blacklist_path", default="data/blacklist.txt", type=Path)
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evals") / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save argument snapshot -------------------------------------------------
    with open(out_dir / "args.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # 1. Load model & diffusion
    # ---------------------------------------------------------------------
    cfg, model, diffusion = _load_checkpoint(args.ckpt_dir, device)

    # ---------------------------------------------------------------------
    # 2. Tokeniser / detokeniser
    # ---------------------------------------------------------------------
    tokenizer, detokenizer = create_simple_tokenizer(cfg.vocab_size)

    # ---------------------------------------------------------------------
    # 3. Gene mappings (18k list + HVG idx mapping)
    # ---------------------------------------------------------------------
    gene_names, hvg_to_full, hvg_gene_ids = _prepare_gene_mappings(cfg.finetune_data_path, args.hvg_ids_path)

    # ---------------------------------------------------------------------
    # 4. Validation evaluation
    # ---------------------------------------------------------------------
    _run_validation_merged(
        cfg,
        model,
        diffusion,
        tokenizer,
        detokenizer,
        gene_names,
        hvg_to_full,
        hvg_gene_ids,
        args,
        device,
        out_dir,
    )

    # ---------------------------------------------------------------------
    # 5. Test-set generation
    # ---------------------------------------------------------------------
    _run_test_generation(
        cfg,
        model,
        diffusion,
        detokenizer,
        gene_names,
        hvg_to_full,
        hvg_gene_ids,
        args,
        device,
        out_dir,
    )

    print("\nAll done – results stored in", out_dir)


if __name__ == "__main__":
    main()

'''
cell-eval prep -i /workspace/vcc/evals/eval_20250802_051921/test_predictions.h5ad -g /workspace/vcc/data/vcc_data/gene_names.csv
'''