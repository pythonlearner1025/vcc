#!/usr/bin/env python3
"""
Inference script for the Conditional VAE.

This script demonstrates how to use a trained VAE model for:
1. Cell type prediction
2. Perturbation effect prediction
3. Generation of synthetic cells
4. Latent space analysis
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig
from train_vae import ScRNAVAEDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str, device: str = 'cuda') -> Tuple[ConditionalVAE, Dict, Dict]:
    """
    Load a trained VAE model.
    
    Returns:
        model: Trained ConditionalVAE
        config: Model configuration
        vocabularies: Metadata vocabularies
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    config_dict = checkpoint['config']
    config = VAEConfig(**config_dict)
    
    model = ConditionalVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    vocabularies = checkpoint['vocabularies']
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, config, vocabularies


def predict_cell_types(model: ConditionalVAE, 
                      data_loader,
                      vocabularies: Dict,
                      device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict cell types for all cells in the dataset.
    
    Returns:
        predicted_types: Predicted cell type IDs
        true_types: True cell type IDs
    """
    model.eval()
    
    predicted_types = []
    true_types = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting cell types"):
            x = batch['x'].to(device)
            true_cell_type_ids = batch['cell_type_ids'].to(device)
            perturbation_ids = batch['perturbation_ids'].to(device)
            experiment_ids = batch['experiment_ids'].to(device)
            
            # Predict cell types by trying all possibilities
            batch_size = x.size(0)
            n_cell_types = len(vocabularies['cell_type'])
            
            best_cell_types = []
            
            for i in range(batch_size):
                x_single = x[i:i+1].expand(n_cell_types, -1)
                pert_single = perturbation_ids[i:i+1].expand(n_cell_types)
                exp_single = experiment_ids[i:i+1].expand(n_cell_types)
                
                all_cell_types = torch.arange(n_cell_types, device=device)
                
                outputs = model(x_single, all_cell_types, pert_single, exp_single)
                
                # Compute reconstruction errors
                recon_errors = torch.nn.functional.mse_loss(
                    outputs['reconstructed'], x_single, reduction='none'
                ).sum(dim=1)
                
                best_cell_type = torch.argmin(recon_errors)
                best_cell_types.append(best_cell_type.cpu().numpy())
            
            predicted_types.extend(best_cell_types)
            true_types.extend(true_cell_type_ids.cpu().numpy())
    
    return np.array(predicted_types), np.array(true_types)


def analyze_perturbation_effects(model: ConditionalVAE,
                                data_loader,
                                vocabularies: Dict,
                                device: str = 'cuda',
                                control_perturbation: str = 'control') -> Dict:
    """
    Analyze perturbation effects by comparing with control condition.
    
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    # Get control perturbation ID
    control_id = vocabularies['perturbation'].get(control_perturbation, 0)
    
    perturbation_effects = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Analyzing perturbation effects"):
            x = batch['x'].to(device)
            cell_type_ids = batch['cell_type_ids'].to(device)
            perturbation_ids = batch['perturbation_ids'].to(device)
            experiment_ids = batch['experiment_ids'].to(device)
            
            # Only analyze non-control perturbations
            non_control_mask = perturbation_ids != control_id
            if not non_control_mask.any():
                continue
            
            x_perturbed = x[non_control_mask]
            cell_type_ids_perturbed = cell_type_ids[non_control_mask]
            perturbation_ids_perturbed = perturbation_ids[non_control_mask]
            experiment_ids_perturbed = experiment_ids[non_control_mask]
            
            # Encode perturbed state
            mu, logvar = model.encode(x_perturbed, cell_type_ids_perturbed, 
                                    perturbation_ids_perturbed, experiment_ids_perturbed)
            z = model.reparameterize(mu, logvar)
            
            # Generate control prediction
            control_perturbation_ids = torch.full_like(perturbation_ids_perturbed, control_id)
            x_control_pred = model.decode(z, cell_type_ids_perturbed, 
                                        control_perturbation_ids, experiment_ids_perturbed)
            
            # Compute perturbation effect (log fold change)
            log_fold_change = torch.log2((x_perturbed + 1) / (x_control_pred + 1))
            
            # Store results by perturbation type
            for i, pert_id in enumerate(perturbation_ids_perturbed):
                pert_name = None
                for name, id_val in vocabularies['perturbation'].items():
                    if id_val == pert_id.item():
                        pert_name = name
                        break
                
                if pert_name and pert_name != control_perturbation:
                    if pert_name not in perturbation_effects:
                        perturbation_effects[pert_name] = []
                    
                    perturbation_effects[pert_name].append(log_fold_change[i].cpu().numpy())
    
    # Summarize effects
    summary = {}
    for pert_name, effects_list in perturbation_effects.items():
        effects = np.array(effects_list)
        summary[pert_name] = {
            'mean_effect': np.mean(effects, axis=0),
            'std_effect': np.std(effects, axis=0),
            'n_cells': len(effects_list)
        }
    
    return summary


def generate_synthetic_cells(model: ConditionalVAE,
                           vocabularies: Dict,
                           n_samples: int = 100,
                           cell_type: str = 'T_cell',
                           perturbation: str = 'control',
                           experiment: str = None,
                           device: str = 'cuda') -> np.ndarray:
    """
    Generate synthetic cells with specified conditions.
    
    Returns:
        Generated gene expression data [n_samples, n_genes]
    """
    model.eval()
    
    # Get condition IDs
    cell_type_id = vocabularies['cell_type'].get(cell_type, 0)
    perturbation_id = vocabularies['perturbation'].get(perturbation, 0)
    
    # Use first experiment if not specified
    if experiment is None:
        experiment_id = 0
    else:
        experiment_id = vocabularies['srx_accession'].get(experiment, 0)
    
    # Create condition tensors
    cell_type_ids = torch.full((n_samples,), cell_type_id, device=device)
    perturbation_ids = torch.full((n_samples,), perturbation_id, device=device)
    experiment_ids = torch.full((n_samples,), experiment_id, device=device)
    
    # Generate samples
    with torch.no_grad():
        generated = model.generate(cell_type_ids, perturbation_ids, experiment_ids)
    
    return generated.cpu().numpy()


def analyze_latent_space(model: ConditionalVAE,
                        data_loader,
                        vocabularies: Dict,
                        device: str = 'cuda',
                        max_cells: int = 5000) -> Dict:
    """
    Analyze the latent space representations.
    
    Returns:
        Dictionary with latent representations and metadata
    """
    model.eval()
    
    latent_representations = []
    metadata = {
        'cell_type_ids': [],
        'perturbation_ids': [],
        'experiment_ids': []
    }
    
    n_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting latent representations"):
            if n_processed >= max_cells:
                break
            
            x = batch['x'].to(device)
            cell_type_ids = batch['cell_type_ids'].to(device)
            perturbation_ids = batch['perturbation_ids'].to(device)
            experiment_ids = batch['experiment_ids'].to(device)
            
            # Encode to latent space
            mu, logvar = model.encode(x, cell_type_ids, perturbation_ids, experiment_ids)
            
            # Use mean of latent distribution
            latent_representations.append(mu.cpu().numpy())
            metadata['cell_type_ids'].extend(cell_type_ids.cpu().numpy())
            metadata['perturbation_ids'].extend(perturbation_ids.cpu().numpy())
            metadata['experiment_ids'].extend(experiment_ids.cpu().numpy())
            
            n_processed += len(x)
    
    latent_representations = np.vstack(latent_representations)
    
    return {
        'latent_representations': latent_representations,
        'metadata': metadata
    }


def visualize_latent_space(latent_data: Dict, vocabularies: Dict, output_dir: str):
    """Create visualizations of the latent space."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latent_reps = latent_data['latent_representations']
    metadata = latent_data['metadata']
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(latent_reps)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(latent_reps[:2000])  # Limit for speed
    
    # Create reverse vocabularies for labeling
    reverse_vocabs = {}
    for key, vocab in vocabularies.items():
        reverse_vocabs[key] = {v: k for k, v in vocab.items()}
    
    # Plot PCA colored by cell type
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cell_type_labels = [reverse_vocabs['cell_type'].get(id, f'Unknown_{id}') 
                       for id in metadata['cell_type_ids']]
    unique_types = list(set(cell_type_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    
    for i, cell_type in enumerate(unique_types):
        mask = np.array(cell_type_labels) == cell_type
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                   c=[colors[i]], label=cell_type, alpha=0.6, s=1)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA - Colored by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot PCA colored by perturbation
    plt.subplot(1, 2, 2)
    pert_labels = [reverse_vocabs['perturbation'].get(id, f'Unknown_{id}') 
                  for id in metadata['perturbation_ids']]
    unique_perts = list(set(pert_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_perts)))
    
    for i, pert in enumerate(unique_perts):
        mask = np.array(pert_labels) == pert
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                   c=[colors[i]], label=pert, alpha=0.6, s=1)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA - Colored by Perturbation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # t-SNE plot (subset)
    plt.figure(figsize=(12, 5))
    
    subset_size = min(2000, len(tsne_coords))
    cell_type_labels_subset = cell_type_labels[:subset_size]
    pert_labels_subset = pert_labels[:subset_size]
    
    plt.subplot(1, 2, 1)
    for i, cell_type in enumerate(unique_types):
        mask = np.array(cell_type_labels_subset) == cell_type
        if mask.any():
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                       c=[colors[i]], label=cell_type, alpha=0.6, s=1)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE - Colored by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_perts)))
    for i, pert in enumerate(unique_perts):
        mask = np.array(pert_labels_subset) == pert
        if mask.any():
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                       c=[colors[i]], label=pert, alpha=0.6, s=1)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE - Colored by Perturbation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved latent space visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="VAE inference and analysis")
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing batch_*.h5 files')
    parser.add_argument('--output-dir', type=str, default='output/vae_inference',
                       help='Output directory for results')
    parser.add_argument('--hvg-file', type=str, default=None,
                       help='File containing HVG gene names')
    
    # Analysis options
    parser.add_argument('--predict-cell-types', action='store_true',
                       help='Predict cell types')
    parser.add_argument('--analyze-perturbations', action='store_true',
                       help='Analyze perturbation effects')
    parser.add_argument('--generate-cells', action='store_true',
                       help='Generate synthetic cells')
    parser.add_argument('--analyze-latent', action='store_true',
                       help='Analyze latent space')
    
    # Generation parameters
    parser.add_argument('--n-generate', type=int, default=100,
                       help='Number of cells to generate')
    parser.add_argument('--generate-cell-type', type=str, default='T_cell',
                       help='Cell type for generation')
    parser.add_argument('--generate-perturbation', type=str, default='control',
                       help='Perturbation for generation')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--max-cells', type=int, default=10000,
                       help='Maximum cells to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model, config, vocabularies = load_trained_model(args.model_path, args.device)
    
    # Load HVG genes if provided
    hvg_genes = None
    if args.hvg_file and os.path.exists(args.hvg_file):
        with open(args.hvg_file, 'r') as f:
            hvg_genes = [line.strip() for line in f if line.strip()]
    
    # Create dataset and data loader
    dataset = ScRNAVAEDataset(
        data_dir=args.data_dir,
        hvg_genes=hvg_genes,
        max_cells_per_batch=args.max_cells
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Created dataset with {len(dataset)} cells")
    
    # Run requested analyses
    if args.predict_cell_types:
        logger.info("Predicting cell types...")
        predicted_types, true_types = predict_cell_types(model, data_loader, vocabularies, args.device)
        
        accuracy = (predicted_types == true_types).mean()
        logger.info(f"Cell type prediction accuracy: {accuracy:.4f}")
        
        # Save results
        np.save(output_dir / 'predicted_cell_types.npy', predicted_types)
        np.save(output_dir / 'true_cell_types.npy', true_types)
    
    if args.analyze_perturbations:
        logger.info("Analyzing perturbation effects...")
        pert_effects = analyze_perturbation_effects(model, data_loader, vocabularies, args.device)
        
        with open(output_dir / 'perturbation_effects.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_effects = {}
            for pert, effects in pert_effects.items():
                json_effects[pert] = {
                    'mean_effect': effects['mean_effect'].tolist(),
                    'std_effect': effects['std_effect'].tolist(),
                    'n_cells': effects['n_cells']
                }
            json.dump(json_effects, f, indent=2)
        
        logger.info(f"Analyzed {len(pert_effects)} perturbation types")
    
    if args.generate_cells:
        logger.info("Generating synthetic cells...")
        generated_cells = generate_synthetic_cells(
            model, vocabularies, args.n_generate,
            args.generate_cell_type, args.generate_perturbation, device=args.device
        )
        
        np.save(output_dir / 'generated_cells.npy', generated_cells)
        logger.info(f"Generated {len(generated_cells)} synthetic cells")
    
    if args.analyze_latent:
        logger.info("Analyzing latent space...")
        latent_data = analyze_latent_space(model, data_loader, vocabularies, args.device)
        
        # Save latent representations
        np.save(output_dir / 'latent_representations.npy', latent_data['latent_representations'])
        with open(output_dir / 'latent_metadata.json', 'w') as f:
            json.dump(latent_data['metadata'], f)
        
        # Create visualizations
        try:
            visualize_latent_space(latent_data, vocabularies, output_dir)
        except ImportError as e:
            logger.warning(f"Could not create visualizations: {e}")
        
        logger.info(f"Extracted latent representations for {len(latent_data['latent_representations'])} cells")
    
    logger.info("Analysis completed!")


if __name__ == "__main__":
    main()
