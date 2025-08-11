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
from tqdm import tqdm

import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# Local imports – reuse code from the training script and model package
# -----------------------------------------------------------------------------

from tokenizer import (
    create_logbin_tokenizer
)
      # noqa: E402  – defined at top-level
from models.diffusion import (
    ConditionalDiffusionTransformer,
    ConditionalModelConfig,
    PartialMaskingDiffusion,
)
from dataset.vcc_paired_dataloader import (
    create_vcc_train_val_dataloaders
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
    """Convert token tensor (..., N) to numpy float32 with the same leading shape.

    Accepts either 2D or higher-rank tensors; the output numpy array preserves
    the input shape and converts values to float32.
    """
    with torch.no_grad():
        vals = detokenizer(tokens.cpu())  # (B,N)
    return vals.numpy().astype(np.float32)


def _build_18k_matrix(
    expr_2k: np.ndarray, hvg_to_full: np.ndarray, n_full: int
) -> np.ndarray:
    """Insert HVG expression into zero-initialised 18,080-gene matrix."""
    print(expr_2k.shape)
    S, N = expr_2k.shape
    out = np.zeros((S, n_full), dtype=np.float32)
    # since if gname not in hvg, its -1 
    valid_mask = hvg_to_full >= 0
    out[:, hvg_to_full[valid_mask]] = expr_2k[:, valid_mask]
    return out


def _build_global_batch_mapping(cfg: ConditionalModelConfig, vcc_batches: Sequence[str]) -> Dict[str, int]:
    """Create merged batch_to_idx mapping identical to `train_orion.py` logic."""
    from dataset.orion_paired import OrionPairedDataset  # local import to avoid heavy deps unless needed
    orion_ds = OrionPairedDataset(cfg.pretrain_data_dir,
                                  set_size=cfg.vcc_set_size,
                                  hvg_gene_ids=None,
                                  seed=0)
    unique_batches = sorted(set(vcc_batches) | set(orion_ds.unique_batches))
    return {b: idx for idx, b in enumerate(unique_batches)}


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
    batch_to_idx: Dict[str, int] | None = None,
    val_ds=None,
    val_dl=None,
    max_genes: int = 10
):
    """Generate one merged prediction/ground-truth pair for all
    validation genes and run cell-eval once."""
    mask_token = cfg.vocab_size - 1
    # Default: if caller did not supply val_ds/dl we will create VCC val loader below

    # ------------------------------------------------------------------
    # Build merged batch_to_idx mapping identical to training
    # ------------------------------------------------------------------
    if batch_to_idx is None:
        if hasattr(val_ds, 'adata'):
            vcc_batches = sorted(list(set(val_ds.adata.obs['batch'].values)))
        else:
            # Orion Subset case
            base_ds = val_ds.dataset if hasattr(val_ds, 'dataset') else val_ds
            vcc_batches = []
        batch_to_idx = _build_global_batch_mapping(cfg, vcc_batches)
    setattr(val_ds, 'batch_to_idx', batch_to_idx)

    X_true_parts = []
    X_pred_parts = []
    obs_labels: list[str] = []

    steps = 0
    for sample in val_dl:
        #if steps >= 1:
        #    break
        steps += 1
        print(f"num genes: {steps}")
        # Tokens from collator (already tokenised)
        pert_tok = sample["tokens"]   # (B,S,N) – perturbed tokens
        ctrl_tok  = sample["control"]  # (B,S,N) – control tokens
        gene_idxs = sample["target_gene_idx"].to(device, non_blocking=True)  # (B,S)
        batch_idx = sample["batch_idx"].to(device, non_blocking=True)        # (B,S)
        B, S, N = pert_tok.shape
        print(f"pert_tok.shape: {pert_tok.shape}")

        # Control set tokens are already provided by the collator
        tokens_ctrl = ctrl_tok.to(device, non_blocking=True)
        # Build float control baseline via detokeniser for metrics (CPU numpy)
        ctrl_expr = _tokens_to_expression(ctrl_tok, detokenizer)  # (B,S,N) float32

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Sampler expects explicit (B,S,N) shape
            pred_tokens = diffusion.p_sample_loop(
                model,
                shape=(B, S, cfg.n_genes),
                control_set=tokens_ctrl,
                target_gene_idx=gene_idxs,
                batch_idx=batch_idx
            )

        pred_expr_2k = _tokens_to_expression(pred_tokens, detokenizer)  # (B,S,N)
        # Ground-truth perturbed expression from provided perturbed tokens
        true_expr_2k = _tokens_to_expression(pert_tok, detokenizer)     # (B,S,N)

        # Map to 18 k space per batch element
        for b in range(B):
            pred_full = _build_18k_matrix(pred_expr_2k[b], hvg_to_full, len(gene_names))
            true_full = _build_18k_matrix(true_expr_2k[b], hvg_to_full, len(gene_names))
            ctrl_full = _build_18k_matrix(ctrl_expr[b], hvg_to_full, len(gene_names))

            X_true_parts.extend([ctrl_full, true_full])
            X_pred_parts.extend([ctrl_full, pred_full])

            # Append labels for control and perturbed cells (S each) for this batch item
            batch_gene_idxs = gene_idxs[b].view(-1).tolist()
            obs_label = ["non-targeting"] * S + [gene_names[i] for i in batch_gene_idxs]
            obs_labels.extend(obs_label)
    
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
# ---------------------------------------------------------------------------
# Fast test‑set generation (GPU‑centric, minimal host/device transfers)
# ---------------------------------------------------------------------------
def _run_test_generation(
    cfg: ConditionalModelConfig,
    model: torch.nn.Module,
    diffusion: PartialMaskingDiffusion,
    detokenizer,                           # delta detokeniser passed in main()
    gene_names: List[str],
    hvg_to_full: np.ndarray,
    hvg_gene_ids: List[str],
    args,
    device: torch.device,
    out_dir: Path,
    test_sample_size: int,
):
    """
    Much faster version of the original routine.

    ‣ Performs sampling *and* detokenisation entirely on the GPU.
    ‣ Only the final (already‑dense) 18 k‑gene expression matrix is copied
      back to the host – once per mini‑batch.
      # required
      - cfg.vocab_size
      - cfg.vcc_set_size 
      - cfg.n_genes
    """
    import pandas as pd
    import scanpy as sc
    from tqdm import tqdm

    # ------------------------------------------------------------------
    # Static / constant tensors (created **once**; stay on GPU)
    # ------------------------------------------------------------------
    hvg_to_full_t = torch.from_numpy(hvg_to_full).to(device, non_blocking=True)
    valid_mask_t  = (hvg_to_full_t >= 0)
    full_idx_t    = hvg_to_full_t[valid_mask_t].long()
    n_full        = len(gene_names)

    # Helper to scatter n k expression into 18 k space on GPU
    def _scatter_to_full(expr_nk: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(expr_nk.size(0), n_full,
                           dtype=expr_nk.dtype,
                           device=expr_nk.device)
        full.index_copy_(1, full_idx_t, expr_nk[:, valid_mask_t])
        return full

    # Use config-driven sizes (args may not define these)
    vocab_size = int(getattr(cfg, "vocab_size"))
    set_size = int(getattr(cfg, "vcc_set_size", 64))
    batch_size_cfg = int(getattr(cfg, "vcc_batch_size", 1))

    tokenizer, _ = create_logbin_tokenizer(vocab_size)

    # Precompute GPU detokenisation bin centres once
    # bins length == vocab_size; bins[1:] are log edges; last centre uses upper bound
    bins_dev = tokenizer.bins.to(device)
    bin_centres_dev = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    bin_centres_dev[0] = 0.0
    if vocab_size > 2:
        bin_centres_dev[1:-1] = (bins_dev[1:-1] + bins_dev[2:]) * 0.5
    bin_centres_dev[-1] = bins_dev[-1]

    # ------------------------------------------------------------------
    # Load control cells once, keep in backed mode
    # ------------------------------------------------------------------
    adata_ctrl = sc.read_h5ad("/competition_train.h5", backed="r")
    ctrl_idx   = np.where(adata_ctrl.obs["target_gene"].values == "non-targeting")[0]
    if len(ctrl_idx) == 0:
        raise RuntimeError("Dataset contains no non‑targeting control cells")

    # Build HVG index map (once)
    gid_to_name = { (gid.split(".")[0] if isinstance(gid, str) else gid): gname
                    for gid, gname in zip(adata_ctrl.var["gene_id"].values,
                                          adata_ctrl.var_names) }
    hvg_names   = [gid_to_name.get(gid) for gid in hvg_gene_ids]
    hvg_idx     = np.asarray(
        [adata_ctrl.var_names.get_loc(g) for g in hvg_names if g is not None],
        dtype=np.int32,
    )

    # Materialise control HVG matrix once into host RAM to avoid repeated CSR->dense conversions
    ctrl_full_block = adata_ctrl[ctrl_idx, hvg_idx].X
    if not isinstance(ctrl_full_block, np.ndarray):
        ctrl_full_block = ctrl_full_block.toarray()
    ctrl_full_block = ctrl_full_block.astype(np.float32, copy=False)

    # RNG for control‑cell sampler
    rng = np.random.default_rng(0)
    def _next_control_expr_cpu() -> np.ndarray:
        """Sample S control cells → (S,2000) FP32 numpy (CPU)."""
        sel_pos = rng.choice(ctrl_full_block.shape[0], size=S, replace=False)
        return ctrl_full_block[sel_pos]
    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------
    test_genes = pd.read_csv(args.test_genes_path)["target_gene"].tolist()
    gene_name_to_hvg = {
        gene_names[idx]: i for i, idx in enumerate(hvg_to_full) if idx >= 0
    }

    all_expr_full_cpu: List[np.ndarray] = []
    all_obs_labels:   List[str]        = []

    mask_token = vocab_size
    S          = set_size 

    model.eval().requires_grad_(False)
    torch.cuda.empty_cache()

    # Allow batching multiple target genes per diffusion call (B > 1)
    B_cfg = batch_size_cfg
    for i in tqdm(range(0, len(test_genes), B_cfg), desc="test‑genes"):
        batch_genes = test_genes[i:i + B_cfg]

        # Prepare control sets for each batch element: (B_eff, S, 2000)
        B_eff = len(batch_genes)
        for _ in range(test_sample_size):
            ctrl_expr_cpu_batch = np.stack([
                _next_control_expr_cpu() for _ in range(B_eff)
            ], axis=0)  # (B_eff, S, 2000)

            # Tokenise controls directly on GPU to avoid CPU round-trips
            ctrl_expr_t = torch.from_numpy(ctrl_expr_cpu_batch).to(device, non_blocking=True)
            tokens_ctrl_t = tokenizer(ctrl_expr_t).long()

            # Split into predicted vs control-only entries
            pred_indices: List[int] = []
            pred_gene_ids: List[int] = []
            control_indices: List[int] = []
            for b, g in enumerate(batch_genes):
                gid = gene_name_to_hvg.get(g)
                if gid is None:
                    print(f"[WARN] gene '{g}' not in HVG list – skipping")
                    # treat as no-op for this position
                    continue
                pred_indices.append(b)
                pred_gene_ids.append(gid)

            # Generate predictions for non-control targets in the batch
            if pred_indices:
                B_pred = len(pred_indices)
                tgt_idx_t = torch.as_tensor(pred_gene_ids, dtype=torch.long, device=device)
                tgt_idx_t = tgt_idx_t.view(B_pred, 1).expand(B_pred, S)  # (B_pred,S)
                batch_idx_t = torch.zeros((B_pred, S), dtype=torch.long, device=device)
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_tokens = diffusion.p_sample_loop(
                        model,
                        shape=(B_pred, S, cfg.n_genes),
                        control_set=tokens_ctrl_t.index_select(0, torch.as_tensor(pred_indices, device=device)),
                        target_gene_idx=tgt_idx_t,
                        batch_idx=batch_idx_t
                    )  # (B_pred,S,N)

                # Detokenise on GPU using precomputed bin centres
                pred_tokens_clamped = pred_tokens.clamp_(0, vocab_size - 1)
                pred_expr_2k = bin_centres_dev[pred_tokens_clamped]  # (B_pred,S,N) float
                # Scatter each predicted target to 18k and append
                for j, b in enumerate(pred_indices):
                    expr_full_cpu = _scatter_to_full(pred_expr_2k[j].view(S, -1)).detach().cpu().numpy()
                    all_expr_full_cpu.append(expr_full_cpu)
                    all_obs_labels.extend([batch_genes[b]] * S)

            # Append control-only entries directly
            for b in control_indices:
                expr_full_cpu = _scatter_to_full(
                     torch.from_numpy(ctrl_expr_cpu_batch[b]).to(device)
                 ).detach().cpu().numpy()
                all_expr_full_cpu.append(expr_full_cpu)
                all_obs_labels.extend(["non-targeting"] * S)

    # ------------------------------------------------------------------
    # Concatenate & save
    # ------------------------------------------------------------------
    if not all_expr_full_cpu:
        print("No predictions generated – aborting.")
        return

    X_full = np.concatenate(all_expr_full_cpu, axis=0)
    
    # ------------------------------------------------------------------
    # Append real control cells if not already included
    # ------------------------------------------------------------------
    # Check if we already have control cells from test_genes
    n_existing_controls = all_obs_labels.count("non-targeting")
    
    if n_existing_controls < set_size:
        # Add control cells to reach at least args.val_set_size total
        n_to_add = set_size - n_existing_controls
        print(f"Appending {n_to_add} real control cells (existing: {n_existing_controls})...")
        
        # Sample control cells
        ctrl_sel = rng.choice(ctrl_idx, size=n_to_add, replace=False)
        ctrl_X = adata_ctrl[ctrl_sel, hvg_idx].X
        ctrl_expr_2k = ctrl_X.toarray().astype(np.float32) if not isinstance(ctrl_X, np.ndarray) else ctrl_X
        
        # Convert to full gene space on GPU
        ctrl_expr_full = _scatter_to_full(
            torch.from_numpy(ctrl_expr_2k).to(device)
        ).detach().cpu().numpy()
        
        # Append to existing data
        X_full = np.vstack([X_full, ctrl_expr_full])
        all_obs_labels.extend(["non-targeting"] * n_to_add)
    else:
        print(f"Already have {n_existing_controls} control cells, no additional controls needed.")
    
    # Create and save final AnnData
    import pandas as pd
    ad.AnnData(
        X_full,
        obs=pd.DataFrame({"target_gene": all_obs_labels}),
        var=pd.DataFrame(index=gene_names),
    ).write_h5ad(out_dir / "test_predictions.h5ad", compression="gzip")

    final_n_controls = all_obs_labels.count("non-targeting")
    print(f"Saved → {out_dir/'test_predictions.h5ad'}   "
          f"(n_cells={X_full.shape[0]:,}, including {final_n_controls} control cells)")

# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate VCC diffusion model")
    p.add_argument("--ckpt_dir", default="/run_20250731_072803", type=Path)
    p.add_argument("--test_prep", default=0)
    p.add_argument("--val_genes_number", type=int, default=10)
    p.add_argument("--val_set_size", type=int, default=8)
    p.add_argument("--test_genes_path", default="data/vcc_data/pert_counts_Validation.csv", type=Path)
    p.add_argument("--test_sample_size", type=int, default=1)
    p.add_argument("--test_gene_names_path", default="data/vcc_data/gene_names.csv", type=Path)
    p.add_argument("--hvg_ids_path", default="hvg_seuratv3_global_scRNA1e5_2000.txt", type=Path)
    p.add_argument("--blacklist_path", default="assets/blacklist.txt", type=Path)
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
    tokenizer, detokenizer = create_logbin_tokenizer(cfg.vocab_size)
    # 3. Gene mappings (18k list + HVG idx mapping)
    # ---------------------------------------------------------------------
    gene_names, hvg_to_full, hvg_gene_ids = _prepare_gene_mappings(cfg.finetune_data_path, args.hvg_ids_path)

    # ---------------------------------------------------------------------
    # 4. Validation evaluation
    # ---------------------------------------------------------------------
    if not args.test_prep:
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
        args.test_sample_size
    )
    print("\nAll done – results stored in", out_dir)

if __name__ == "__main__":
    main()

'''
cell-eval prep -i checkpoints/scrna_to_vcc_diffusion_20250811_002641/eval_ep9/test_predictions.h5ad -g /workspace/vcc/data/vcc_data/gene_names.csv
'''