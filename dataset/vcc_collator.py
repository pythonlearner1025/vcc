#!/usr/bin/env python3
"""VCC collate function that performs expensive CPU work (tokenisation, control-set expansion,
   batch-index mapping, etc.) inside the DataLoader worker processes so that the training loop
   only receives ready-to-use tensors.
"""

from typing import List, Dict
import torch

class VCCCollator:
    """Callable collate_fn for torch DataLoader.

    It expects that each element in *batch_list* is a dictionary produced by
    ``VCCPairedDataset.__getitem__`` with keys:
        - 'perturbed_expr': FloatTensor (S, N)
        - 'control_expr':   FloatTensor (S, N)
        - 'target_gene_idx': int
        - 'pert_batches':   List[str] of length S  (batch name for each perturbed cell)

    The collator stacks the sets, tokenises everything in a vectorised fashion,
    expands the control sets to shape (B·S, S, N), and maps the batch names to
    integer indices used by the model.
    """

    def __init__(self, tokenizer, batch_to_idx: Dict[str, int], set_size: int):
        self.tokenizer = tokenizer
        self.batch_to_idx = batch_to_idx
        self.set_size = set_size

    def __call__(self, batch_list: List[Dict]):
        # (B, S, N) float
        pert = torch.stack([b["perturbed_expr"] for b in batch_list])
        ctrl = torch.stack([b["control_expr"] for b in batch_list])

        # Compute Δ = perturbed − mean(control) before tokenisation
        ctrl_mean = ctrl.mean(1, keepdim=True)  # (B,1,N)
        delta_expr = pert - ctrl_mean            # (B,S,N)

        # Vectorised tokenisation on CPU (runs in DataLoader worker)
        delta_tok = self.tokenizer(delta_expr)
        ctrl_tok = self.tokenizer(ctrl)

        B, S, N = delta_tok.shape  # S == self.set_size

        # Flatten perturbed tokens -> (B·S, N)
        tokens = delta_tok.view(B * S, N)

        # Expand control sets so each perturbed cell gets the full control set
        ctrl_tok_expanded = (
            ctrl_tok.unsqueeze(1)          # (B, 1, S, N)
                   .expand(-1, S, -1, -1)  # (B, S, S, N)
                   .reshape(B * S, S, N)   # (B·S, S, N)
        )

        # Batch indices  (B·S,)
        batch_idx_list = [
            self.batch_to_idx.get(batch_name, 0)
            for sample in batch_list
            for batch_name in sample["pert_batches"]
        ]
        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long)

        # Target-gene indices  (B·S,)
        tgt_gene_idx = torch.tensor(
            [sample["target_gene_idx"] for sample in batch_list for _ in range(self.set_size)],
            dtype=torch.long,
        )

        return {
            "tokens": tokens.long(),              # (B·S, N)
            "control": ctrl_tok_expanded.long(),  # (B·S, S, N)
            "batch_idx": batch_idx,               # (B·S,)
            "target_gene_idx": tgt_gene_idx,      # (B·S,)
        }
