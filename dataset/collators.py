#!/usr/bin/env python3
"""VCC collate function that performs expensive CPU work (tokenisation, control-set expansion,
   batch-index mapping, etc.) inside the DataLoader worker processes so that the training loop
   only receives ready-to-use tensors.
"""

from typing import List, Dict
import torch

class OrionCollator:
    """Collate function for Orion datasets (batch_size is always 1).

    Each element in *batch_list* is a dict with keys:
        - 'perturbed_expr': FloatTensor (S, N)
        - 'control_expr':   FloatTensor (S, N)
        - 'target_gene_idx': int
        - 'pert_batches':   List[str] length S

    We convert expression matrices to tokens, build a (S, S, N) control tensor
    and map batch names to integer indices on-the-fly.
    """

    def __init__(self, tokenizer, set_size: int, batch_to_idx: Dict[str, int] | None = None):
        self.tokenizer = tokenizer
        self.set_size = set_size
        # use external global (+vcc) batch_to_idx map
        if batch_to_idx is None:
            self._batch_to_idx: Dict[str, int] = {}
            self._use_external_mapping = False
        else:
            self._batch_to_idx = batch_to_idx
            self._use_external_mapping = True

    def _map_batch(self, name: str) -> int:
        # use external global (+vcc) batch_to_idx map
        if self._use_external_mapping:
            return self._batch_to_idx.get(name, 0)
        else:
            if name not in self._batch_to_idx:
                self._batch_to_idx[name] = len(self._batch_to_idx)
            return self._batch_to_idx[name]

    def __call__(self, batch_list: List[Dict]):
        # batch_size is guaranteed to be 1
        sample = batch_list[0]
        pert = sample["perturbed_expr"]  # (S,N)
        ctrl = sample["control_expr"]    # (S,N)
        S, N = pert.shape

        # Tokenise (vectorised over set)
        delta_tok = self.tokenizer((pert - ctrl).unsqueeze(0)).squeeze(0)  # (S,N)
        ctrl_tok = self.tokenizer(ctrl.unsqueeze(0)).squeeze(0)            # (S,N)

        # Expand control set so each perturbed cell gets full control matrix
        ctrl_tok_expanded = ctrl_tok.unsqueeze(0).expand(S, -1, -1)  # (S,S,N)

        # Batch indices
        batch_idx = torch.tensor([self._map_batch(b) for b in sample["pert_batches"]], dtype=torch.long)

        return {
            "tokens": delta_tok.long(),              # (S,N)
            "control": ctrl_tok_expanded.long(),     # (S,S,N)
            "batch_idx": batch_idx,                  # (S,)
            "target_gene_idx": torch.tensor(sample["target_gene_idx"], dtype=torch.long).repeat(S),
        }

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
        # batch_size == 1 in this training setup
        sample = batch_list[0]
        pert = sample["perturbed_expr"].squeeze()  # (S,N)
        ctrl = sample["control_expr"].squeeze()    # (S,N)
        S, N = pert.shape

        delta_expr = pert - ctrl  # (S,N)
        delta_tok = self.tokenizer(delta_expr.unsqueeze(0)).squeeze(0)  # (S,N)
        ctrl_tok = self.tokenizer(ctrl.unsqueeze(0)).squeeze(0)         # (S,N)

        ctrl_tok_expanded = ctrl_tok.unsqueeze(0).expand(S, -1, -1)  # (S,S,N)

        # Map batch names to integer indices
        batch_idx = torch.tensor([self.batch_to_idx.get(b, 0) for b in sample["pert_batches"]], dtype=torch.long)

        return {
            "tokens": delta_tok.long(),
            "control": ctrl_tok_expanded.long(),
            "batch_idx": batch_idx,  # (S,)
            "target_gene_idx": torch.tensor(sample["target_gene_idx"], dtype=torch.long).repeat(S),
        }

        # ----------------------------------------------------------------
        # Old code retained for reference (multi-batch variant)
        # ----------------------------------------------------------------
        '''
        batch_idx_list = [
            self.batch_to_idx.get(batch_name, 0)
            for sample in batch_list
            for batch_name in sample["pert_batches"]
        ]

        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long)
        '''

        # Target-gene indices  (B·S,)
        '''
        tgt_gene_idx = torch.tensor(
            [sample["target_gene_idx"] for sample in batch_list for _ in range(self.set_size)],
            dtype=torch.long,
        )

        return {
            "tokens": tokens.long(),              # (S, N)
            "control": ctrl_tok_expanded.long(),  # (S, S, N)
            "batch_idx": batch_idx.long(),               # (S,)
            #"batch_idx": batch_idx_list.long(),               # (B*S,)
            #"target_gene_idx": tgt_gene_idx.long(),      # (B*S,)
            "target_gene_idx": target_gene_idx.long(),      # (S,)
        }
        '''