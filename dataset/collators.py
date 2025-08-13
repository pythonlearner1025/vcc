#!/usr/bin/env python3
"""VCC collate function that performs expensive CPU work (tokenisation, control-set expansion,
   batch-index mapping, etc.) inside the DataLoader worker processes so that the training loop
   only receives ready-to-use tensors.
"""

from typing import List, Dict
import torch

class OrionCollator:
    """Collate for Orion datasets, supports batch_size >= 1.

    Input per item:
      - perturbed_expr: (S, N)
      - control_expr:   (S, N)
      - target_gene_idx: int
      - pert_batches:   List[str] length S

    Output tensors are flattened to shape (B*S, N) for compatibility with the
    current training/loss stack. delta_mean is returned only for B==1.
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

    # Allow runtime injection of global mapping via attribute set
    @property
    def batch_to_idx(self) -> Dict[str, int]:
        return self._batch_to_idx

    @batch_to_idx.setter
    def batch_to_idx(self, mapping: Dict[str, int]) -> None:
        self._batch_to_idx = mapping
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
        B = len(batch_list)
        perts, ctrls, batch_ids, tgt_ids = [], [], [], []
        delta_means = []
        for sample in batch_list:
            pert = sample["perturbed_expr"]  # (S,N)
            ctrl = sample["control_expr"]    # (S,N)
            S, N = pert.shape
            # Tokenise
            delta = pert - ctrl
            pert_tok = self.tokenizer(delta)  # (S,N)
            ctrl_tok = self.tokenizer(ctrl)  # (S,N)
            perts.append(pert_tok.long())
            ctrls.append(ctrl_tok.long())
            batch_ids.append(torch.tensor([self._map_batch(b) for b in sample["pert_batches"]], dtype=torch.long))
            tgt_ids.append(torch.tensor(sample["target_gene_idx"], dtype=torch.long).repeat(S))
            # delta_mean only when unbatched to keep semantics (N,)
            p_mean = pert.mean(0)
            c_mean = ctrl.mean(0)
            delta_means.append(p_mean - c_mean)
        tokens = torch.stack(perts, dim=0)   # (B,S,N)
        control = torch.stack(ctrls, dim=0)  # (B,S,N)
        batch_idx = torch.stack(batch_ids, dim=0)  # (B,S,)
        target_gene_idx = torch.stack(tgt_ids, dim=0)  # (B,S,)
        assert (target_gene_idx >= 0).all(), "target_gene_idx contains negative values"
        delta_means = torch.stack(delta_means, dim=0)  # (B,S,)
        out = {
            "tokens": tokens,
            "control": control,
            "batch_idx": batch_idx,
            "target_gene_idx": target_gene_idx,
            "tokenizer": self.tokenizer,
            "delta_means": delta_means
        }
        return out

class VCCCollator:
    """Collate for VCC dataset, supports batch_size >= 1 and flattens to (B*S, N).

    delta_mean is included only when B==1 (shape (N,)).
    """

    def __init__(self, tokenizer, batch_to_idx: Dict[str, int], set_size: int):
        self.tokenizer = tokenizer
        self.batch_to_idx = batch_to_idx
        self.set_size = set_size

    def __call__(self, batch_list: List[Dict]):
        B = len(batch_list)
        perts, ctrls, batch_ids, tgt_ids = [], [], [], []
        delta_means = []
        for sample in batch_list:
            pert = sample["perturbed_expr"].squeeze()  # (S,N)
            ctrl = sample["control_expr"].squeeze()    # (S,N)
            S, N = pert.shape
            # Tokenise targets as deltas and context as control
            delta = pert - ctrl
            pert_tok = self.tokenizer(delta)  # (S,N) – targets are Δ tokens
            ctrl_tok = self.tokenizer(ctrl)   # (S,N) – context encodes control
            perts.append(pert_tok.long())
            ctrls.append(ctrl_tok.long())
            batch_ids.append(torch.tensor([self.batch_to_idx.get(b, 0) for b in sample["pert_batches"]], dtype=torch.long))
            tgt_ids.append(torch.tensor(sample["target_gene_idx"], dtype=torch.long).repeat(S))
            p_mean = pert.mean(0)
            c_mean = ctrl.mean(0)
            delta_means.append(p_mean - c_mean)
        tokens = torch.stack(perts, dim=0)   # (B,S,N)
        control = torch.stack(ctrls, dim=0)  # (B,S,N)
        batch_idx = torch.stack(batch_ids, dim=0)  # (B,S,)
        target_gene_idx = torch.stack(tgt_ids, dim=0)  # (B,S,)
        delta_means = torch.stack(delta_means, dim=0)  # (B,S,)
        out = {
            "tokens": tokens,
            "control": control,
            "batch_idx": batch_idx,
            "target_gene_idx": target_gene_idx,
            "tokenizer": self.tokenizer,
            "delta_means": delta_means
        }
        return out
