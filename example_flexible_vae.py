#!/usr/bin/env python3
"""
Comprehensive example demonstrating the Flexible VAE pipeline.

This script showcases the complete workflow:
1. Data preparation and validation
2. Phase 1: Self-supervised pretraining
3. Phase 2: Perturbation fine-tuning
4. Inference and perturbation injection
5. Latent space analysis and visualization

Features:
- Synthetic data generation for testing
- Complete training pipeline demonstration
- Perturbation injection examples
- Latent space visualization
- Model evaluation metrics

Usage:
    python example_flexible_vae.py --mode synthetic    # Use synthetic data
    python example_flexible_vae.py --mode real --data_path data/real_data.npz
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flexible_vae import (
    FlexibleVAE, VAEConfig, LearnedGeneEmbedding, 
    PretrainedGeneEmbedding, flexible_vae_loss
)
from dataset.flexible_dataloader import (
    FlexibleDataset, PairedPerturbationDataset, 
    create_flexible_dataloaders
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_synthetic_data(n_experiments: int = 5,
                         n_perturbations: int = 10,
                         n_cells_per_condition: int = 200,
                         n_genes: int = 1000,
                         n_cell_types: int = 3,
                         perturbation_strength: float = 2.0,
                         noise_level: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Create realistic synthetic single-cell RNA-seq data with known perturbation effects.
    
    This function generates data that mimics real scRNA-seq characteristics:
    - Cell type-specific expression patterns
    - Experiment/batch effects
    - Targeted perturbation effects on specific genes
    - Realistic count distributions
    
    Returns:
        Dictionary containing expression data, IDs, and metadata
    """
    print("Generating synthetic data...")
    
    # Define cell type signatures (genes preferentially expressed in each type)
    cell_type_signatures = []
    genes_per_signature = n_genes // n_cell_types
    
    for i in range(n_cell_types):
        start_idx = i * genes_per_signature
        end_idx = min((i + 1) * genes_per_signature, n_genes)
        signature = np.zeros(n_genes)
        signature[start_idx:end_idx] = np.random.exponential(2.0, end_idx - start_idx)
        cell_type_signatures.append(signature)
    
    # Storage for all data
    all_expression = []
    all_experiment_ids = []
    all_perturbation_ids = []
    all_target_gene_ids = []
    all_cell_types = []
    
    for exp_id in range(n_experiments):
        # Experiment-specific effects (batch effects)
        exp_effect = np.random.normal(0, 0.5, n_genes)
        
        # Generate control cells for this experiment
        for cell_type in range(n_cell_types):
            n_cells = n_cells_per_condition
            
            # Cell type base expression
            base_expression = cell_type_signatures[cell_type].copy()
            
            for cell_idx in range(n_cells):
                # Individual cell variation
                cell_expression = base_expression + np.random.normal(0, noise_level, n_genes)
                cell_expression += exp_effect  # Batch effect
                cell_expression = np.maximum(cell_expression, 0)  # No negative expression
                
                # Convert to count-like data
                cell_expression = np.random.poisson(cell_expression * 100) / 100.0
                
                all_expression.append(cell_expression)
                all_experiment_ids.append(exp_id)
                all_perturbation_ids.append(0)  # Control perturbation ID
                all_target_gene_ids.append(0)  # No target gene for control
                all_cell_types.append(cell_type)
        
        # Generate perturbed cells for this experiment
        for pert_id in range(1, n_perturbations + 1):
            # Each perturbation targets specific genes
            target_genes = np.random.choice(n_genes, size=3, replace=False)
            target_gene_id = target_genes[0]  # Primary target for this perturbation
            
            for cell_type in range(n_cell_types):
                n_cells = n_cells_per_condition
                
                # Cell type base expression
                base_expression = cell_type_signatures[cell_type].copy()
                
                for cell_idx in range(n_cells):
                    # Individual cell variation
                    cell_expression = base_expression + np.random.normal(0, noise_level, n_genes)
                    cell_expression += exp_effect  # Batch effect
                    
                    # Apply perturbation effect
                    perturbation_effect = np.zeros(n_genes)
                    for target_gene in target_genes:
                        # Primary effect on target gene
                        effect_strength = perturbation_strength * (2 if target_gene == target_gene_id else 1)
                        perturbation_effect[target_gene] = np.random.normal(effect_strength, 0.3)
                        
                        # Secondary effects on related genes (pathway effects)
                        related_genes = np.random.choice(n_genes, size=5, replace=False)
                        for related_gene in related_genes:
                            perturbation_effect[related_gene] += np.random.normal(0, 0.5)
                    
                    cell_expression += perturbation_effect
                    cell_expression = np.maximum(cell_expression, 0)  # No negative expression
                    
                    # Convert to count-like data
                    cell_expression = np.random.poisson(cell_expression * 100) / 100.0
                    
                    all_expression.append(cell_expression)
                    all_experiment_ids.append(exp_id)
                    all_perturbation_ids.append(pert_id)
                    all_target_gene_ids.append(target_gene_id)
                    all_cell_types.append(cell_type)
    
    # Convert to numpy arrays
    data = {
        'expression': np.array(all_expression, dtype=np.float32),
        'experiment_ids': np.array(all_experiment_ids, dtype=np.int64),
        'perturbation_ids': np.array(all_perturbation_ids, dtype=np.int64),
        'target_gene_ids': np.array(all_target_gene_ids, dtype=np.int64),
        'cell_types': np.array(all_cell_types, dtype=np.int64),
        'gene_names': [f'Gene_{i}' for i in range(n_genes)]
    }
    
    print(f"Generated data: {data['expression'].shape[0]} cells, {data['expression'].shape[1]} genes")
    print(f"Experiments: {len(np.unique(data['experiment_ids']))}")
    print(f"Perturbations: {len(np.unique(data['perturbation_ids']))}")
    print(f"Cell types: {len(np.unique(data['cell_types']))}")
    
    return data


def demonstrate_phase1_training(data: Dict[str, np.ndarray], 
                               config: VAEConfig,
                               num_epochs: int = 20) -> FlexibleVAE:
    """Demonstrate Phase 1 pretraining."""
    print("\n" + "="*50)
    print("PHASE 1: SELF-SUPERVISED PRETRAINING")
    print("="*50)
    
    # Create Phase 1 dataset (no perturbation labels)
    dataset = FlexibleDataset(
        expression_data=data['expression'],
        experiment_ids=data['experiment_ids'],
        gene_names=data['gene_names'],
        normalize=True,
        log_transform=True
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model with learned gene embeddings
    gene_embedding = LearnedGeneEmbedding(config.n_genes, config.target_gene_embed_dim)
    model = FlexibleVAE(config, gene_embedding)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        epoch_train_loss = 0.0
        for batch in train_loader:
            expression = batch['expression']
            experiment_ids = batch['experiment_id']
            
            # Forward pass (no perturbation for Phase 1)
            outputs = model(expression, experiment_ids)
            
            # Compute loss
            loss_dict = flexible_vae_loss(outputs, expression, config)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                expression = batch['expression']
                experiment_ids = batch['experiment_id']
                
                outputs = model(expression, experiment_ids)
                loss_dict = flexible_vae_loss(outputs, expression, config)
                epoch_val_loss += loss_dict['total_loss'].item()
        
        model.train()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    print(f"Phase 1 training completed!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    return model


def demonstrate_phase2_training(data: Dict[str, np.ndarray], 
                               pretrained_model: FlexibleVAE,
                               config: VAEConfig,
                               num_epochs: int = 15) -> FlexibleVAE:
    """Demonstrate Phase 2 fine-tuning with perturbation data."""
    print("\n" + "="*50)
    print("PHASE 2: PERTURBATION FINE-TUNING")
    print("="*50)
    
    # Create Phase 2 dataset (with perturbation labels)
    dataset = PairedPerturbationDataset(
        expression_data=data['expression'],
        experiment_ids=data['experiment_ids'],
        perturbation_ids=data['perturbation_ids'],
        target_gene_ids=data['target_gene_ids'],
        control_perturbation_id=0,
        min_cells_per_condition=10,
        balance_pairs=True,
        normalize=True,
        log_transform=True
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Use pretrained model
    model = pretrained_model
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * 0.1)
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    print(f"Fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        epoch_train_loss = 0.0
        for batch in train_loader:
            control_expression = batch['control_expression']
            perturbed_expression = batch['perturbed_expression']
            experiment_ids = batch['experiment_id']
            target_gene_ids = batch['target_gene_id']
            
            # Forward pass with perturbation
            outputs = model(control_expression, experiment_ids, target_gene_ids=target_gene_ids)
            
            # Compute loss against perturbed expression
            loss_dict = flexible_vae_loss(outputs, perturbed_expression, config)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                control_expression = batch['control_expression']
                perturbed_expression = batch['perturbed_expression']
                experiment_ids = batch['experiment_id']
                target_gene_ids = batch['target_gene_id']
                
                outputs = model(control_expression, experiment_ids, target_gene_ids=target_gene_ids)
                loss_dict = flexible_vae_loss(outputs, perturbed_expression, config)
                epoch_val_loss += loss_dict['total_loss'].item()
        
        model.train()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    print(f"Phase 2 fine-tuning completed!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    return model


def demonstrate_perturbation_injection(data: Dict[str, np.ndarray], 
                                     model: FlexibleVAE,
                                     config: VAEConfig):
    """Demonstrate perturbation injection and analysis."""
    print("\n" + "="*50)
    print("PERTURBATION INJECTION ANALYSIS")
    print("="*50)
    
    model.eval()
    
    # Select test data (control cells)
    control_mask = data['perturbation_ids'] == 0
    control_expression = data['expression'][control_mask]
    control_experiment_ids = data['experiment_ids'][control_mask]
    control_cell_types = data['cell_types'][control_mask]
    
    # Select a subset for analysis
    n_test_cells = min(100, len(control_expression))
    test_indices = np.random.choice(len(control_expression), n_test_cells, replace=False)
    
    test_expression = torch.FloatTensor(control_expression[test_indices])
    test_experiment_ids = torch.LongTensor(control_experiment_ids[test_indices])
    test_cell_types = control_cell_types[test_indices]
    
    print(f"Testing perturbation injection on {n_test_cells} control cells...")
    
    # Test different perturbations
    perturbation_results = {}
    
    with torch.no_grad():
        # Original (control) predictions
        control_outputs = model(test_expression, test_experiment_ids)
        control_predictions = control_outputs['reconstructed'].numpy()
        
        # Test each perturbation
        unique_perturbations = np.unique(data['target_gene_ids'])
        unique_perturbations = unique_perturbations[unique_perturbations > 0]  # Exclude control
        
        for pert_gene_id in unique_perturbations[:5]:  # Test first 5 perturbations
            target_gene_ids = torch.LongTensor([pert_gene_id] * len(test_expression))
            
            # Inject perturbation
            pert_outputs = model(test_expression, test_experiment_ids, target_gene_ids=target_gene_ids)
            pert_predictions = pert_outputs['reconstructed'].numpy()
            
            # Calculate perturbation effect
            perturbation_effect = pert_predictions - control_predictions
            
            perturbation_results[pert_gene_id] = {
                'predictions': pert_predictions,
                'effect': perturbation_effect,
                'target_gene': pert_gene_id
            }
            
            # Print summary statistics
            mean_effect = np.mean(np.abs(perturbation_effect), axis=0)
            max_effect_gene = np.argmax(mean_effect)
            
            print(f"Perturbation {pert_gene_id}: Max effect on gene {max_effect_gene} ({mean_effect[max_effect_gene]:.3f})")
    
    return perturbation_results


def visualize_latent_space(data: Dict[str, np.ndarray], 
                          model: FlexibleVAE,
                          config: VAEConfig):
    """Visualize the learned latent space."""
    print("\n" + "="*50)
    print("LATENT SPACE VISUALIZATION")
    print("="*50)
    
    model.eval()
    
    # Sample data for visualization
    n_viz_cells = min(1000, len(data['expression']))
    viz_indices = np.random.choice(len(data['expression']), n_viz_cells, replace=False)
    
    viz_expression = torch.FloatTensor(data['expression'][viz_indices])
    viz_experiment_ids = torch.LongTensor(data['experiment_ids'][viz_indices])
    viz_perturbation_ids = data['perturbation_ids'][viz_indices]
    viz_cell_types = data['cell_types'][viz_indices]
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(viz_expression, viz_experiment_ids)
        latent_means = mu.numpy()
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_means)
    
    # Create visualization plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Color by cell type
    scatter = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=viz_cell_types, 
                             cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title('Latent Space - Cell Types')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0], label='Cell Type')
    
    # Plot 2: Color by experiment
    scatter = axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=viz_experiment_ids, 
                             cmap='viridis', alpha=0.6, s=10)
    axes[1].set_title('Latent Space - Experiments')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[1], label='Experiment ID')
    
    # Plot 3: Color by perturbation
    scatter = axes[2].scatter(latent_2d[:, 0], latent_2d[:, 1], c=viz_perturbation_ids, 
                             cmap='plasma', alpha=0.6, s=10)
    axes[2].set_title('Latent Space - Perturbations')
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[2], label='Perturbation ID')
    
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Latent space visualization saved as 'latent_space_visualization.png'")
    print(f"Latent space captures {pca.explained_variance_ratio_.sum():.2%} of variance in 2D")


def evaluate_model_performance(data: Dict[str, np.ndarray], 
                              model: FlexibleVAE,
                              config: VAEConfig):
    """Evaluate model performance with comprehensive metrics."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    model.eval()
    
    # Test reconstruction quality on control cells
    control_mask = data['perturbation_ids'] == 0
    test_expression = torch.FloatTensor(data['expression'][control_mask][:500])  # First 500 control cells
    test_experiment_ids = torch.LongTensor(data['experiment_ids'][control_mask][:500])
    
    with torch.no_grad():
        outputs = model(test_expression, test_experiment_ids)
        reconstructed = outputs['reconstructed'].numpy()
        original = test_expression.numpy()
    
    # Calculate metrics
    r2_scores = []
    pearson_correlations = []
    mse_scores = []
    
    for i in range(len(original)):
        r2 = r2_score(original[i], reconstructed[i])
        pearson_r, _ = pearsonr(original[i], reconstructed[i])
        mse = mean_squared_error(original[i], reconstructed[i])
        
        r2_scores.append(r2)
        pearson_correlations.append(pearson_r)
        mse_scores.append(mse)
    
    # Print summary statistics
    print(f"Reconstruction Quality (n={len(original)} cells):")
    print(f"  Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"  Mean Pearson r: {np.mean(pearson_correlations):.4f} ± {np.std(pearson_correlations):.4f}")
    print(f"  Mean MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    
    # Gene-wise analysis
    gene_r2_scores = []
    for gene_idx in range(original.shape[1]):
        gene_r2 = r2_score(original[:, gene_idx], reconstructed[:, gene_idx])
        gene_r2_scores.append(gene_r2)
    
    print(f"\nGene-wise Reconstruction:")
    print(f"  Mean gene R²: {np.mean(gene_r2_scores):.4f} ± {np.std(gene_r2_scores):.4f}")
    print(f"  Best reconstructed genes: {np.argsort(gene_r2_scores)[-5:]}")
    print(f"  Worst reconstructed genes: {np.argsort(gene_r2_scores)[:5]}")
    
    return {
        'cell_r2_scores': r2_scores,
        'cell_pearson_correlations': pearson_correlations,
        'cell_mse_scores': mse_scores,
        'gene_r2_scores': gene_r2_scores
    }


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description='Flexible VAE Example')
    parser.add_argument('--mode', type=str, choices=['synthetic', 'real'], default='synthetic',
                       help='Data mode: synthetic or real')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to real data file (required for real mode)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained models')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent space dimension')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FLEXIBLE VAE COMPREHENSIVE EXAMPLE")
    print("="*60)
    
    if args.mode == 'synthetic':
        # Generate synthetic data
        data = create_synthetic_data(
            n_experiments=3,
            n_perturbations=5,
            n_cells_per_condition=100,
            n_genes=500,
            n_cell_types=3
        )
        
        # Save synthetic data for later use
        os.makedirs('examples/data', exist_ok=True)
        np.savez('examples/data/synthetic_data.npz', **data)
        print("Synthetic data saved to 'examples/data/synthetic_data.npz'")
        
    else:
        # Load real data
        if not args.data_path:
            raise ValueError("--data_path required for real mode")
        
        from dataset.flexible_dataloader import load_data_from_file
        expression_data, experiment_ids, metadata = load_data_from_file(args.data_path)
        
        data = {
            'expression': expression_data,
            'experiment_ids': experiment_ids,
            **metadata
        }
    
    # Create configuration
    config = VAEConfig(
        input_dim=data['expression'].shape[1],
        latent_dim=args.latent_dim,
        n_experiments=len(np.unique(data['experiment_ids'])),
        n_genes=data['expression'].shape[1],  # Use full gene vocabulary for embeddings
        learning_rate=1e-3,
        batch_size=64
    )
    
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {config.input_dim}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Number of experiments: {config.n_experiments}")
    print(f"  Number of genes: {config.n_genes}")
    
    # Demonstrate complete pipeline
    try:
        # Phase 1: Pretraining
        model = demonstrate_phase1_training(data, config, num_epochs=10)
        
        if args.save_model:
            os.makedirs('examples/models', exist_ok=True)
            torch.save(model.state_dict(), 'examples/models/phase1_model.pt')
            print("Phase 1 model saved to 'examples/models/phase1_model.pt'")
        
        # Phase 2: Fine-tuning (if perturbation data available)
        if 'perturbation_ids' in data and len(np.unique(data['perturbation_ids'])) > 1:
            model = demonstrate_phase2_training(data, model, config, num_epochs=8)
            
            if args.save_model:
                torch.save(model.state_dict(), 'examples/models/phase2_model.pt')
                print("Phase 2 model saved to 'examples/models/phase2_model.pt'")
            
            # Perturbation injection analysis
            perturbation_results = demonstrate_perturbation_injection(data, model, config)
        
        # Latent space visualization
        visualize_latent_space(data, model, config)
        
        # Model evaluation
        performance_metrics = evaluate_model_performance(data, model, config)
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files generated:")
        print("  - latent_space_visualization.png")
        if args.mode == 'synthetic':
            print("  - examples/data/synthetic_data.npz")
        if args.save_model:
            print("  - examples/models/phase1_model.pt")
            if 'perturbation_ids' in data:
                print("  - examples/models/phase2_model.pt")
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
