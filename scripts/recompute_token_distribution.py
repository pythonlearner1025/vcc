#!/usr/bin/env python3
"""
Recompute token distribution for frequency-aware masking.

This script replicates the data loading pipeline used in `train.py` to produce
an updated `token_distribution.json` that matches the current
`ConditionalModelConfig.vocab_size`.

The resulting JSON maps a token id (as a string) to the number of times that
token appears across *both* the pre-training scRNA dataset and the fine-tuning
VCC paired dataset (train **and** validation splits).

Usage (from repository root):

    python scripts/recompute_token_distribution.py --output token_distribution.json

You can override any of the dataset paths or other parameters via the command
line; run with ``--help`` to see all options.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Local imports – keep them here to avoid import-time GPU allocations in PyTorch
# -----------------------------------------------------------------------------
from models.diffusion import ConditionalModelConfig
from tokenizer import create_delta_tokenizer
from dataset.scrna_hvg_dataset import (
    ScRNADatasetWithHVGs,
    create_scrna_hvg_dataloader,
)
from dataset.vcc_paired_dataloader import (
    create_vcc_train_val_dataloaders,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _accumulate_counts(token_tensor: torch.Tensor, counts: torch.Tensor):
    """Accumulate token counts from *token_tensor* into *counts* (in-place).

    Parameters
    ----------
    token_tensor : torch.Tensor
        Tensor containing token ids (any shape, integer dtype).
    counts : torch.Tensor
        1-D tensor of shape ``(vocab_size,)`` holding running counts.
    """
    token_tensor = token_tensor.reshape(-1).to(torch.int64)
    batch_counts = torch.bincount(token_tensor, minlength=counts.numel())
    counts += batch_counts


def _iterate_dataloader(dl: DataLoader, counts: torch.Tensor, *, include_control: bool = False):
    """Iterate over a dataloader and update *counts*.

    If *include_control* is True, control-set tokens present under the "control"
    key in each batch will also be counted.
    """
    for batch in dl:
        if isinstance(batch, dict):
            # Tokenised batches from VCCCollator or similar structure
            _accumulate_counts(batch["tokens"], counts)
            if include_control and "control" in batch:
                _accumulate_counts(batch["control"], counts)
        else:
            # Plain tensor batch from pretraining DataLoader (TokenisedScRNADataset)
            _accumulate_counts(batch, counts)

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recompute token distribution JSON for masking.")
    parser.add_argument("--output", type=str, default="token_distribution.json", help="Path to output JSON file")

    # Allow overriding any dataset-related parameters if needed
    parser.add_argument("--pretrain_data_dir", type=str, default="/scRNA_norm/processed")
    parser.add_argument("--finetune_adata_path", type=str, default="/competition_train.h5")
    parser.add_argument("--hvg_info_path", type=str, default="assets/hvg_seuratv3_3000.txt")
    parser.add_argument("--vocab_size", type=int, default=128, help="Override vocab size; defaults to config setting")
    parser.add_argument("--debug_pretrain_max_cells", type=int, default=1e5, help="Limit number of pretrain cells (for quick runs)")
    parser.add_argument("--debug_finetune_max_cells", type=int, default=1e5, help="Limit number of fine-tune samples (for quick runs)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config & tokenizer
    # ------------------------------------------------------------------
    config = ConditionalModelConfig()
    if args.vocab_size is not None:
        config.vocab_size = args.vocab_size
    vocab_size = config.vocab_size

    tokenizer, _ = create_delta_tokenizer(vocab_size)

    # Ensure output directory exists
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load HVG gene list
    # ------------------------------------------------------------------
    hvg_info_path = Path(args.hvg_info_path).expanduser().resolve()
    if not hvg_info_path.exists():
        raise FileNotFoundError(f"HVG info file not found: {hvg_info_path}")

    with open(hvg_info_path, "r") as f:
        hvg_genes: List[str] = [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------
    # 1. Pretrain dataset (single cells)
    # ------------------------------------------------------------------
    scrna_dataset, _ = create_scrna_hvg_dataloader(
        data_dir=args.pretrain_data_dir,
        hvg_genes=hvg_genes,
        batch_size=1,   # temp; we will wrap later so batch size not important here
        shuffle=False,
        num_workers=0,
        use_cache=True,
        normalize=False,
    )

    from train import TokenizedScRNADataset as _TokDS  # class defined in train.py
    pretrain_dataset = _TokDS(scrna_dataset, tokenizer)

    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 2. Fine-tune dataset (VCC paired sets – train & val)
    # ------------------------------------------------------------------
    (train_dataset, train_loader), (val_dataset, val_loader) = create_vcc_train_val_dataloaders(
        adata_path=args.finetune_adata_path,
        hvg_gene_ids=hvg_genes,
        set_size=config.vcc_set_size,
        batch_size=1,  # We are interested in counts, not GPU utilisation
        n_samples_per_gene_train=10,
        n_samples_per_gene_val=1,
        train_split=0.8,
        num_workers=4,
        tokenizer=tokenizer,
        random_seed=42,
        normalize=False,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Accumulate counts
    # ------------------------------------------------------------------
    counts = torch.zeros(vocab_size, dtype=torch.int64)

    # 2.1 Pretrain tokens
    processed_cells = 0
    for batch in pretrain_loader:
        _iterate_dataloader([batch] if isinstance(batch, (torch.Tensor, dict)) else batch, counts)
        processed_cells += batch.size(0) if isinstance(batch, torch.Tensor) else batch["tokens"].size(0)
        if args.debug_pretrain_max_cells is not None and processed_cells >= args.debug_pretrain_max_cells:
            break

    # 2.2 Fine-tune train tokens (include control sets)
    processed_batches = 0
    for batch in train_loader:
        _iterate_dataloader([batch], counts, include_control=True)
        processed_batches += 1
        if args.debug_finetune_max_cells is not None and processed_batches >= args.debug_finetune_max_cells:
            break

    # 2.3 Fine-tune validation tokens (include control sets)
    processed_batches = 0
    for batch in val_loader:
        _iterate_dataloader([batch], counts, include_control=True)
        processed_batches += 1
        if args.debug_finetune_max_cells is not None and processed_batches >= args.debug_finetune_max_cells:
            break

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    token_counts: Dict[str, int] = {str(i): int(c) for i, c in enumerate(counts.tolist()) if c > 0}

    with open(output_path, "w") as f:
        json.dump(token_counts, f, indent=2)

    total = counts.sum().item()
    print(f"Saved token distribution for {total:,} tokens to {output_path}")


if __name__ == "__main__":
    main()
