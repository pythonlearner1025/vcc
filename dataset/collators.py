#!/usr/bin/env python3
"""VCC collate function that performs expensive CPU work (tokenisation, control-set expansion,
   batch-index mapping, etc.) inside the DataLoader worker processes so that the training loop
   only receives ready-to-use tensors.
"""

from typing import List, Dict
import torch
import pickle
import os
from datetime import datetime

def _centers_from_edges(edges: torch.Tensor) -> torch.Tensor:
    """Compute representative bin centers from a sorted 1D tensor of bin edges.

    Token indices are produced via torch.bucketize(x, edges) ∈ [0, K] then clamped to [0, K-1],
    where K = len(edges). Effective classes:
      - class 0:              (-inf, edges[0]]
      - class i in 1..K-2:    (edges[i-1], edges[i]]
      - class K-1:            (edges[K-2], +inf)

    We use extrapolated centers for both tails and midpoints for interior intervals.
    """
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError("edges must be 1D with length >= 2")
    K = edges.numel()
    centers = torch.empty_like(edges)
    # Interior midpoints (classes 0..K-2)
    centers[:-1] = 0.5 * (edges[:-1] + edges[1:])
    # Right-most tail uses the upper bound, matching detokenize
    centers[-1] = edges[-1]
    # If there is an explicit zero edge, force the neutral bin center to 0
    zero_locs = (edges == 0).nonzero(as_tuple=False)
    if zero_locs.numel() > 0:
        neutral_idx = int(zero_locs[0].item())
        if 0 <= neutral_idx < K:
            centers[neutral_idx] = torch.tensor(0.0, dtype=edges.dtype, device=edges.device)
    return centers

def gaussian_targets_from_set(delta_set: torch.Tensor, bin_centers: torch.Tensor, sigma_floor: float = 1e-3, save_debug: bool = False):
    """
    delta_set:   (S, N) continuous Δ values for one set (pert - ctrl in log1p space)
    bin_centers: (K,) Δ value at center of each tokenizer bin
    sigma_floor: minimum σ to avoid degenerate peaks
    save_debug:  if True, save debug info for top 16 genes (controlled by global counter)
    Returns:
        target_dist: (N, K) tensor where each row is a probability distribution over bins for that gene
    """
    # Mean and std per gene from the S replicates
    mu = delta_set.mean(dim=0)                # (N,)
    sigma = delta_set.std(dim=0, unbiased=True)  # (N,)
    sigma_unclamped = sigma.clone()  # Save unclamped version for debug
    sigma = torch.clamp(sigma, min=sigma_floor)

    # Save debug info for the first batch with top 16 genes
    # Use a global counter to save only occasionally
    if not hasattr(gaussian_targets_from_set, '_debug_counter'):
        gaussian_targets_from_set._debug_counter = 0
    
    if save_debug and gaussian_targets_from_set._debug_counter == 0:
        # Find top 16 genes by absolute mean delta value (not max!)
        abs_mean_delta = mu.abs()  # Use the already computed mean
        max_abs_delta = delta_set.abs().max(dim=0)[0]  # Still save this for reference
        top_k = 16
        if delta_set.shape[1] >= top_k:  # Ensure we have at least 16 genes
            top_indices = torch.topk(abs_mean_delta, min(top_k, delta_set.shape[1]), largest=True).indices
            
            # Prepare debug data
            debug_data = {
                'timestamp': datetime.now().isoformat(),
                'top_gene_indices': top_indices.cpu().numpy(),
                'delta_distributions': delta_set[:, top_indices].cpu().numpy(),  # (S, 16)
                'mu_values': mu[top_indices].cpu().numpy(),  # (16,)
                'abs_mu_values': abs_mean_delta[top_indices].cpu().numpy(),  # (16,) - ranking criterion
                'sigma_unclamped': sigma_unclamped[top_indices].cpu().numpy(),  # (16,)
                'sigma_clamped': sigma[top_indices].cpu().numpy(),  # (16,)
                'max_abs_delta': max_abs_delta[top_indices].cpu().numpy(),  # (16,) - for reference
                'sigma_floor': sigma_floor,
                'bin_centers': bin_centers.cpu().numpy(),  # (K,)
                'S_replicates': delta_set.shape[0],
                'N_genes': delta_set.shape[1],
                'ranking_method': 'abs_mean_delta'  # Document the ranking method
            }
            
            # Save to file
            os.makedirs('debug_gaussian', exist_ok=True)
            filename = f'debug_gaussian/top16_genes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(debug_data, f)
            print(f"\n🔍 Debug: Saved top-16 genes (ranked by |μ|) to {filename}")
            print(f"   Absolute mean deltas (|μ|): {abs_mean_delta[top_indices][:5].cpu().numpy()}")
            print(f"   Actual mean deltas (μ): {mu[top_indices][:5].cpu().numpy()}")
            print(f"   Max |Δ| values: {max_abs_delta[top_indices][:5].cpu().numpy()}")
            print(f"   Sigma values (clamped): {sigma[top_indices][:5].cpu().numpy()}\n")
    
    gaussian_targets_from_set._debug_counter = (gaussian_targets_from_set._debug_counter + 1) % 1  # Save every 100 batches

    # Align centers device/dtype with inputs
    if bin_centers.device != delta_set.device:
        bin_centers = bin_centers.to(delta_set.device)
    bin_centers = bin_centers.to(delta_set.dtype)

    # Compute Gaussian weights for each bin center, per gene
    # Shape after broadcasting: (N, K)
    diffs = bin_centers.unsqueeze(0) - mu.unsqueeze(1)       # (N,K)
    weights = torch.exp(-0.5 * (diffs / sigma.unsqueeze(1))**2)

    # Normalize to sum=1 per gene
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
    return weights  # (N,K)

class VCCCollator:
    """Collate for VCC dataset, supports batch_size >= 1 and flattens to (B*S, N).

    delta_mean is included only when B==1 (shape (N,)).
    """

    def __init__(self, tokenizer, batch_to_idx: Dict[str, int], set_size: int):
        self.tokenizer = tokenizer
        self.batch_to_idx = batch_to_idx
        self.set_size = set_size
        # Resolve true bin centers once to avoid repeated computation
        centers = getattr(self.tokenizer, 'centers', None)
        if centers is None:
            bins = getattr(self.tokenizer, 'bins', None)
            if bins is None:
                raise AttributeError("Tokenizer must expose `bins` or `centers` for soft-targets.")
            self._bin_centers = _centers_from_edges(bins)
        else:
            self._bin_centers = centers

    def __call__(self, batch_list: List[Dict]):
        B = len(batch_list)
        perts, ctrls, batch_ids, tgt_ids = [], [], [], []
        delta_means = []
        soft_targets = []
        for sample in batch_list:
            pert = sample["perturbed_expr"].squeeze().mean(0).unsqueeze(0)  # (1,N)
            ctrl = sample["control_expr"].squeeze().mean(0).unsqueeze(0)    # (1,N)
            S, N = pert.shape
            # Tokenise targets as deltas and context as control
            pert_tok = self.tokenizer(pert)  # (S,N) – targets are Δ tokens
            ctrl_tok = self.tokenizer(ctrl)   # (S,N) – context encodes control
            perts.append(pert_tok.long())
            ctrls.append(ctrl_tok.long())
            batch_ids.append(torch.tensor([self.batch_to_idx.get(b, 0) for b in sample["pert_batches"]], dtype=torch.long))
            tgt_ids.append(torch.tensor(sample["target_gene_idx"], dtype=torch.long))
            delta_means.append(pert - ctrl)
            # Build Gaussian soft targets over true bin centers (not edges)
            #soft_target = gaussian_targets_from_set(delta, self._bin_centers, save_debug=False)
            #soft_targets.append(soft_target)
        tokens = torch.stack(perts, dim=0)   # (B,1,N)
        control = torch.stack(ctrls, dim=0)  # (B,1,N)
        assert tokens.shape == (B, S, N)
        #control = torch.zeros_like(control)
        batch_idx = torch.stack(batch_ids, dim=0)  # (B,S,)
        target_gene_idx = torch.stack(tgt_ids, dim=0)  # (B,1,)
        #target_gene_idx = torch.zeros_like(target_gene_idx)
        delta_means = torch.stack(delta_means, dim=0).squeeze()  # (B,S,)
        assert delta_means.shape == (B,N)
        
        # Sharpen targets (lower entropy) with numerical stability
        # Default to a safer 3–5 range; allow override via env for annealing/tuning
        '''
        soft_targets = torch.stack(soft_targets, dim=0) # (B,N,K)
        target_gamma = float(os.getenv("VCC_TARGET_GAMMA", "2.0"))
        soft_targets_orig = soft_targets.clone()  # Save original for debugging
        
        # Method 1: Temperature-based sharpening (more stable)
        # Convert to log space for numerical stability
        log_soft_targets = torch.log(soft_targets.clamp_min(1e-8))
        # Apply temperature scaling in log space
        temperature = 1.0 / target_gamma
        log_st = log_soft_targets / temperature
        # Subtract max for numerical stability before exp
        log_st = log_st - log_st.max(dim=-1, keepdim=True)[0]
        # Convert back to probabilities
        st = torch.exp(log_st)
        st = st / st.sum(dim=-1, keepdim=True)  # (B,N,K)
        
        # Add small epsilon to prevent exact zeros
        st = st * 0.9999 + 1e-6 / st.shape[-1]
        st = st / st.sum(dim=-1, keepdim=True)
        '''
        
        out = {
            "tokens": tokens,
            "control": control,
            "batch_idx": batch_idx,
            "target_gene_idx": target_gene_idx,
            "tokenizer": self.tokenizer,
            "delta_means": delta_means,
            "soft_targets": None,   # <- sharpened targets
            #"soft_targets_orig": soft_targets_orig  # <- original for debug
        }
        return out


class OrionCollator(VCCCollator):
    """Collate for Orion datasets, supports batch_size >= 1.
    
    Inherits from VCCCollator but provides additional flexibility for batch mapping.
    When called, it behaves exactly like VCCCollator but with the ability to:
    - Work without an external batch_to_idx mapping (creates one dynamically)
    - Allow runtime injection of batch mapping via property setter

    Input per item:
      - perturbed_expr: (S, N)
      - control_expr:   (S, N)
      - target_gene_idx: int
      - pert_batches:   List[str] length S

    Output tensors are flattened to shape (B*S, N) for compatibility with the
    current training/loss stack. delta_mean is returned only for B==1.
    """

    def __init__(self, tokenizer, set_size: int, batch_to_idx: Dict[str, int] | None = None):
        # Handle the flexible batch mapping initialization
        if batch_to_idx is None:
            self._batch_to_idx: Dict[str, int] = {}
            self._use_external_mapping = False
            # Create a temporary empty dict for parent initialization
            temp_batch_to_idx = {}
        else:
            self._batch_to_idx = batch_to_idx
            self._use_external_mapping = True
            temp_batch_to_idx = batch_to_idx
        
        # Initialize parent class with the batch mapping
        super().__init__(tokenizer, temp_batch_to_idx, set_size)
        
        # Override parent's batch_to_idx with our flexible version
        self.batch_to_idx = self._batch_to_idx

    # Allow runtime injection of global mapping via attribute set
    @property
    def batch_to_idx(self) -> Dict[str, int]:
        return self._batch_to_idx

    @batch_to_idx.setter
    def batch_to_idx(self, mapping: Dict[str, int]) -> None:
        self._batch_to_idx = mapping
        self._use_external_mapping = True

    def _map_batch(self, name: str) -> int:
        """Map batch name to index with automatic creation if not using external mapping."""
        if self._use_external_mapping:
            return self._batch_to_idx.get(name, 0)
        else:
            if name not in self._batch_to_idx:
                self._batch_to_idx[name] = len(self._batch_to_idx)
            return self._batch_to_idx[name]

    def __call__(self, batch_list: List[Dict]):
        """Process batch using parent VCCCollator logic with our batch mapping."""
        # Create a temporary mapping that uses our _map_batch logic
        temp_mapping = {}
        for sample in batch_list:
            for batch_name in sample["pert_batches"]:
                if batch_name not in temp_mapping:
                    temp_mapping[batch_name] = self._map_batch(batch_name)
        
        # Temporarily override batch_to_idx with our mapped values
        original_batch_to_idx = self.batch_to_idx
        self.batch_to_idx = temp_mapping
        
        # Call parent's __call__ method
        result = super().__call__(batch_list)
        
        # Restore our flexible batch_to_idx
        self.batch_to_idx = original_batch_to_idx
        
        return result