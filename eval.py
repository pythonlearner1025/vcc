#!/usr/bin/env python3
"""
Evaluation module for VCC (Virtual Cell Challenge) dataset.
Handles loading VCC data, computing perturbation effects, and evaluating generated samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VCCConfig:
    """Configuration for VCC evaluation."""
    vcc_data_path: str = "/Users/minjunes/Downloads/vcc_data"
    n_genes: int = 18080  # Total genes in VCC
    n_hvgs: int = 2000  # Number of HVGs to use
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def train_path(self) -> Path:
        return Path(self.vcc_data_path) / "adata_Training.h5ad"
    
    @property
    def val_path(self) -> Path:
        return Path(self.vcc_data_path) / "pert_counts_Validation.csv"


class VCCDataset:
    """Handler for VCC dataset loading and preprocessing."""
    
    def __init__(self, config: VCCConfig):
        self.config = config
        self._adata = None
        self._val_df = None
        self._control_mean = None
        self._gene_to_idx = None
        self._hvg_indices = None
        
    @property
    def adata(self):
        """Lazy load training data."""
        if self._adata is None:
            print(f"Loading VCC training data from {self.config.train_path}")
            self._adata = sc.read_h5ad(self.config.train_path)
            # Build gene index mapping
            self._gene_to_idx = {gene: idx for idx, gene in enumerate(self._adata.var.index)}
        return self._adata
    
    @property
    def val_df(self):
        """Lazy load validation data."""
        if self._val_df is None:
            self._val_df = pd.read_csv(self.config.val_path)
        return self._val_df
    
    @property
    def control_mean(self) -> np.ndarray:
        """Get mean expression of control cells."""
        if self._control_mean is None:
            control_mask = self.adata.obs['target_gene'] == 'non-targeting'
            control_expr = self.adata[control_mask].X
            if hasattr(control_expr, 'toarray'):
                self._control_mean = np.array(control_expr.mean(axis=0)).flatten()
            else:
                self._control_mean = control_expr.mean(axis=0)
        return self._control_mean
    
    def get_gene_index(self, gene_name: str) -> Optional[int]:
        """Get the index of a gene in the expression matrix."""
        return self._gene_to_idx.get(gene_name)
    
    def get_perturbed_cells(self, target_gene: str) -> Optional[np.ndarray]:
        """Get expression data for cells with a specific perturbation."""
        mask = self.adata.obs['target_gene'] == target_gene
        if mask.sum() == 0:
            return None
        
        expr = self.adata[mask].X
        if hasattr(expr, 'toarray'):
            return expr.toarray()
        return expr
    
    def compute_perturbation_effect(self, target_gene: str) -> Optional[Dict]:
        """Compute the effect of a perturbation compared to control."""
        pert_expr = self.get_perturbed_cells(target_gene)
        if pert_expr is None:
            return None
        
        pert_mean = pert_expr.mean(axis=0)
        log_fc = np.log2((pert_mean + 1) / (self.control_mean + 1))
        
        # Get target gene knockdown effect
        gene_idx = self.get_gene_index(target_gene)
        target_kd = None
        if gene_idx is not None:
            target_kd = log_fc[gene_idx]
        
        return {
            'mean_expression': pert_mean,
            'log_fc': log_fc,
            'target_knockdown': target_kd,
            'n_cells': len(pert_expr)
        }
    
    def get_hvg_indices(self, hvg_info_path: Optional[str] = None) -> np.ndarray:
        """Get HVG indices, either from file or compute them."""
        if self._hvg_indices is not None:
            return self._hvg_indices
            
        if hvg_info_path and Path(hvg_info_path).exists():
            # Load precomputed HVGs
            with open(hvg_info_path, 'r') as f:
                hvg_info = json.load(f)
            
            # Note: The HVGs were computed on a different dataset (36,601 genes)
            # while VCC has 18,080 genes. We need to map the indices.
            # For now, we'll compute HVGs from VCC data directly.
            print("Warning: HVG file found but gene spaces may differ.")
            print("Computing HVGs from VCC data for consistency...")
            
        # Compute HVGs from VCC data
        print("Computing HVGs from VCC training data...")
        # Use scanpy's HVG selection
        adata_copy = self.adata.copy()
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=self.config.n_hvgs)
        self._hvg_indices = np.where(adata_copy.var['highly_variable'])[0]
            
        return self._hvg_indices
    
    def map_to_hvg_space(self, expression: np.ndarray, hvg_indices: np.ndarray) -> np.ndarray:
        """Map full gene expression to HVG space."""
        return expression[:, hvg_indices]


class PerturbationEvaluator:
    """Evaluates generated perturbation effects against expected patterns."""
    
    def __init__(self, vcc_dataset: VCCDataset):
        self.vcc = vcc_dataset
        self.metrics = {}
        
    def evaluate_perturbation(
        self, 
        generated: np.ndarray, 
        target_gene: str,
        hvg_indices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate generated perturbation against expected effects.
        
        Args:
            generated: Generated expression (n_cells, n_genes or n_hvgs)
            target_gene: Target gene being perturbed
            hvg_indices: If provided, indicates generated is in HVG space
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get expected perturbation effect from training data
        expected_effect = self.vcc.compute_perturbation_effect(target_gene)
        
        if expected_effect is None:
            # Target gene not in training perturbations (expected for validation)
            # Use general perturbation patterns
            return self._evaluate_novel_perturbation(generated, target_gene, hvg_indices)
        
        # Convert to full gene space if needed
        if hvg_indices is not None:
            # Expand HVG to full gene space (simple approach: zeros for non-HVGs)
            full_generated = np.zeros((generated.shape[0], self.vcc.config.n_genes))
            full_generated[:, hvg_indices] = generated
            generated = full_generated
        
        # Compute generated statistics
        gen_mean = generated.mean(axis=0)
        gen_log_fc = np.log2((gen_mean + 1) / (self.vcc.control_mean + 1))
        
        # Metrics
        metrics = {
            'mse_expression': mean_squared_error(expected_effect['mean_expression'], gen_mean),
            'mae_expression': mean_absolute_error(expected_effect['mean_expression'], gen_mean),
            'mse_log_fc': mean_squared_error(expected_effect['log_fc'], gen_log_fc),
            'mae_log_fc': mean_absolute_error(expected_effect['log_fc'], gen_log_fc),
            'correlation': np.corrcoef(expected_effect['mean_expression'], gen_mean)[0, 1],
            'log_fc_correlation': np.corrcoef(expected_effect['log_fc'], gen_log_fc)[0, 1],
        }
        
        # Target gene knockdown accuracy
        gene_idx = self.vcc.get_gene_index(target_gene)
        if gene_idx is not None and expected_effect['target_knockdown'] is not None:
            metrics['target_kd_error'] = abs(gen_log_fc[gene_idx] - expected_effect['target_knockdown'])
            metrics['target_kd_expected'] = expected_effect['target_knockdown']
            metrics['target_kd_generated'] = gen_log_fc[gene_idx]
        
        return metrics
    
    def _evaluate_novel_perturbation(
        self, 
        generated: np.ndarray, 
        target_gene: str,
        hvg_indices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate perturbation for genes not in training set."""
        # For novel perturbations, evaluate general properties
        
        # Convert to full gene space if needed
        if hvg_indices is not None:
            full_generated = np.zeros((generated.shape[0], self.vcc.config.n_genes))
            full_generated[:, hvg_indices] = generated
            generated = full_generated
            
        gen_mean = generated.mean(axis=0)
        gen_log_fc = np.log2((gen_mean + 1) / (self.vcc.control_mean + 1))
        
        # Check if target gene exists in expression matrix
        gene_idx = self.vcc.get_gene_index(target_gene)
        
        metrics = {
            'mean_expression': gen_mean.mean(),
            'sparsity': (generated == 0).mean(),
            'log_fc_magnitude': np.abs(gen_log_fc).mean(),
            'n_upregulated': (gen_log_fc > 0.5).sum(),
            'n_downregulated': (gen_log_fc < -0.5).sum(),
        }
        
        if gene_idx is not None:
            # Evaluate target gene knockdown
            metrics['target_kd_generated'] = gen_log_fc[gene_idx]
            metrics['target_in_control'] = self.vcc.control_mean[gene_idx]
            metrics['target_in_generated'] = gen_mean[gene_idx]
            # Expected knockdown for CRISPR is typically -1 to -3 log2FC
            metrics['target_kd_plausible'] = -3 < gen_log_fc[gene_idx] < -0.5
        
        return metrics
    
    def evaluate_batch(
        self,
        model: nn.Module,
        diffusion,
        target_genes: List[str],
        n_samples_per_gene: int = 100,
        hvg_indices: Optional[np.ndarray] = None,
        tokenizer = None,
        gene_to_idx: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on a batch of target genes.
        
        Args:
            model: Trained diffusion model
            diffusion: Diffusion process object
            target_genes: List of genes to evaluate
            n_samples_per_gene: Number of cells to generate per gene
            hvg_indices: HVG indices if using HVG space
            tokenizer: Tokenizer for converting between continuous and discrete
            gene_to_idx: Mapping from gene names to indices for conditioning
            
        Returns:
            Dictionary mapping gene names to evaluation metrics
        """
        model.eval()
        results = {}
        
        # If gene_to_idx not provided, create a simple mapping
        if gene_to_idx is None:
            # Use VCC gene names if available
            all_genes = list(self.vcc.adata.var.index)
            gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
        
        with torch.no_grad():
            for gene in target_genes:
                print(f"Evaluating perturbation: {gene}")
                
                # Generate samples
                n_genes = len(hvg_indices) if hvg_indices is not None else self.vcc.config.n_genes
                # Get device from model parameters
                device = next(model.parameters()).device
                
                # Prepare conditioning for knockdown (typical use case for VCC)
                from train_conditional_diffusion import prepare_perturbation_conditioning
                gene_idx, magnitude, sign = prepare_perturbation_conditioning(
                    gene,
                    gene_to_idx,
                    perturbation_type="knockdown",
                    magnitude=None,  # Let it sample typical knockdown values
                    batch_size=n_samples_per_gene,
                    device=device
                )
                
                # Generate samples with new conditioning
                samples = diffusion.p_sample_loop(
                    model,
                    shape=(n_samples_per_gene, n_genes),
                    target_gene_idx=gene_idx,
                    perturb_magnitude=magnitude,
                    perturb_sign=sign,
                    device=device
                )
                
                # Convert from tokens to expression values
                if tokenizer is not None:
                    # Assuming tokenizer[1] is detokenize function
                    samples_expr = tokenizer[1](samples).cpu().numpy()
                else:
                    samples_expr = samples.cpu().numpy()
                
                # Evaluate
                metrics = self.evaluate_perturbation(
                    samples_expr, 
                    gene,
                    hvg_indices
                )
                results[gene] = metrics
                
        return results


def create_vcc_evaluator(
    vcc_data_path: str = "/Users/minjunes/Downloads/vcc_data",
    hvg_info_path: Optional[str] = None
) -> Tuple[VCCDataset, PerturbationEvaluator]:
    """
    Create VCC dataset and evaluator instances.
    
    Args:
        vcc_data_path: Path to VCC data directory
        hvg_info_path: Optional path to precomputed HVG info
        
    Returns:
        Tuple of (VCCDataset, PerturbationEvaluator)
    """
    config = VCCConfig(vcc_data_path=vcc_data_path)
    vcc_dataset = VCCDataset(config)
    evaluator = PerturbationEvaluator(vcc_dataset)
    
    return vcc_dataset, evaluator


def evaluate_on_vcc_validation(
    model: nn.Module,
    diffusion,
    tokenizer,
    vcc_data_path: str = "/Users/minjunes/Downloads/vcc_data",
    hvg_info_path: Optional[str] = "hvg_info.json",
    n_samples: int = 100,
    max_genes: int = 10,
    gene_vocabulary: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Main evaluation function for VCC validation set.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        tokenizer: Tokenizer (tokenize, detokenize, vocab_size)
        vcc_data_path: Path to VCC data
        hvg_info_path: Path to HVG info file
        n_samples: Number of samples per gene
        max_genes: Maximum number of validation genes to evaluate
        gene_vocabulary: Optional list of all genes in model vocabulary
        
    Returns:
        Aggregated evaluation metrics
    """
    # Create evaluator
    vcc_dataset, evaluator = create_vcc_evaluator(vcc_data_path, hvg_info_path)
    
    # Get validation genes
    val_genes = vcc_dataset.val_df['target_gene'].tolist()[:max_genes]
    
    # Get HVG indices
    hvg_indices = None
    if hvg_info_path and Path(hvg_info_path).exists():
        hvg_indices = vcc_dataset.get_hvg_indices(hvg_info_path)
    
    # Create gene to index mapping
    if gene_vocabulary is not None:
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_vocabulary)}
    else:
        # Use VCC genes as vocabulary
        gene_to_idx = {gene: idx for idx, gene in enumerate(vcc_dataset.adata.var.index)}
    
    # Evaluate
    results = evaluator.evaluate_batch(
        model, 
        diffusion, 
        val_genes,
        n_samples_per_gene=n_samples,
        hvg_indices=hvg_indices,
        tokenizer=tokenizer,
        gene_to_idx=gene_to_idx
    )
    
    # Aggregate metrics
    aggregated = {}
    metric_names = set()
    for gene_results in results.values():
        metric_names.update(gene_results.keys())
    
    for metric in metric_names:
        values = [results[gene].get(metric, np.nan) for gene in val_genes]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[f'vcc_{metric}_mean'] = np.mean(values)
            aggregated[f'vcc_{metric}_std'] = np.std(values)
    
    # Add summary statistics
    aggregated['vcc_n_genes_evaluated'] = len(val_genes)
    aggregated['vcc_n_samples_per_gene'] = n_samples
    
    return aggregated


# Utility functions for integration with training

def log_vcc_metrics(metrics: Dict[str, float], step: int, wandb_log: bool = True):
    """Log VCC evaluation metrics."""
    print("\nVCC Evaluation Results:")
    print("-" * 50)
    
    for key, value in sorted(metrics.items()):
        if '_mean' in key:
            base_key = key.replace('_mean', '')
            std_key = base_key + '_std'
            if std_key in metrics:
                print(f"{base_key}: {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"{key}: {value:.4f}")
        elif '_std' not in key:
            print(f"{key}: {value:.4f}")
    
    if wandb_log:
        import wandb
        wandb.log({**metrics, 'step': step})


if __name__ == "__main__":
    # Test the evaluation module
    print("Testing VCC evaluation module...")
    
    # Create evaluator
    vcc_dataset, evaluator = create_vcc_evaluator()
    
    # Test loading
    print(f"Training data shape: {vcc_dataset.adata.shape}")
    print(f"Validation genes: {len(vcc_dataset.val_df)}")
    print(f"Control mean shape: {vcc_dataset.control_mean.shape}")
    
    # Test perturbation effect computation
    test_gene = "TMSB4X"  # A gene that exists in training
    effect = vcc_dataset.compute_perturbation_effect(test_gene)
    if effect:
        print(f"\nPerturbation effect for {test_gene}:")
        print(f"  Target knockdown: {effect['target_knockdown']:.2f}")
        print(f"  Number of cells: {effect['n_cells']}")
    
    print("\nVCC evaluation module ready!") 