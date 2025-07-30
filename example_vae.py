#!/usr/bin/env python3
"""
Example script demonstrating VAE usage with synthetic data.

This script creates synthetic single-cell data and shows how to train and use
the Conditional VAE for cell type prediction and perturbation analysis.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig, VAETrainer, vae_loss_function


def create_synthetic_data(n_cells: int = 1000, 
                         n_genes: int = 2000,
                         n_cell_types: int = 5,
                         n_perturbations: int = 3,
                         n_experiments: int = 10) -> Dict[str, torch.Tensor]:
    """
    Create synthetic single-cell RNA-seq data for testing.
    
    Returns:
        Dictionary with synthetic data tensors
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create cell type-specific expression patterns
    cell_type_ids = torch.randint(0, n_cell_types, (n_cells,))
    perturbation_ids = torch.randint(0, n_perturbations, (n_cells,))
    experiment_ids = torch.randint(0, n_experiments, (n_cells,))
    
    # Generate base expression profiles
    x = torch.randn(n_cells, n_genes) * 0.5 + 2.0  # Log-normal-ish expression
    
    # Add cell type effects
    for cell_type in range(n_cell_types):
        mask = cell_type_ids == cell_type
        if mask.any():
            # Each cell type has specific genes upregulated
            start_gene = cell_type * (n_genes // n_cell_types)
            end_gene = min((cell_type + 1) * (n_genes // n_cell_types), n_genes)
            x[mask, start_gene:end_gene] += 1.0
    
    # Add perturbation effects
    for pert in range(1, n_perturbations):  # Skip control (0)
        mask = perturbation_ids == pert
        if mask.any():
            # Perturbations affect random subsets of genes
            affected_genes = np.random.choice(n_genes, size=n_genes//10, replace=False)
            effect_size = np.random.normal(0, 0.5, size=len(affected_genes))
            x[mask, affected_genes] += torch.tensor(effect_size, dtype=torch.float32)
    
    # Add experiment effects (batch effects)
    for exp in range(n_experiments):
        mask = experiment_ids == exp
        if mask.any():
            batch_effect = torch.randn(n_genes) * 0.2
            x[mask] += batch_effect
    
    # Ensure non-negative and add noise
    x = torch.relu(x) + torch.randn_like(x) * 0.1
    x = torch.clamp(x, min=0)
    
    return {
        'x': x,
        'cell_type_ids': cell_type_ids,
        'perturbation_ids': perturbation_ids,
        'experiment_ids': experiment_ids
    }


def create_vocabularies(n_cell_types: int, n_perturbations: int, n_experiments: int) -> Dict[str, Dict]:
    """Create synthetic vocabularies for metadata."""
    
    cell_types = [f'CellType_{i}' for i in range(n_cell_types)]
    perturbations = ['control'] + [f'Treatment_{i}' for i in range(1, n_perturbations)]
    experiments = [f'Experiment_{i}' for i in range(n_experiments)]
    
    return {
        'cell_type': {name: i for i, name in enumerate(cell_types)},
        'perturbation': {name: i for i, name in enumerate(perturbations)},
        'srx_accession': {name: i for i, name in enumerate(experiments)}
    }


def train_example_vae(data: Dict[str, torch.Tensor], 
                     config: VAEConfig,
                     n_epochs: int = 50,
                     device: str = 'cpu') -> ConditionalVAE:
    """Train VAE on synthetic data."""
    
    print("Training VAE on synthetic data...")
    
    # Create model and trainer
    model = ConditionalVAE(config)
    trainer = VAETrainer(model, config, device)
    
    # Training loop
    n_cells = len(data['x'])
    batch_size = min(config.batch_size, n_cells)
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Simple batching (no data loader for simplicity)
        for i in range(0, n_cells, batch_size):
            end_i = min(i + batch_size, n_cells)
            batch = {key: tensor[i:end_i] for key, tensor in data.items()}
            
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)
        
        # Print progress
        avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
        avg_recon = np.mean([l['reconstruction_loss'] for l in epoch_losses])
        avg_kld = np.mean([l['kld_loss'] for l in epoch_losses])
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KLD={avg_kld:.4f}")
    
    print("Training completed!")
    return model


def test_cell_type_prediction(model: ConditionalVAE, 
                            test_data: Dict[str, torch.Tensor],
                            config: VAEConfig,
                            device: str = 'cpu') -> float:
    """Test cell type prediction accuracy."""
    
    print("\nTesting cell type prediction...")
    
    model.eval()
    trainer = VAETrainer(model, config, device)
    
    # Predict cell types
    x = test_data['x']
    true_cell_types = test_data['cell_type_ids']
    perturbation_ids = test_data['perturbation_ids']
    experiment_ids = test_data['experiment_ids']
    
    predicted_cell_types = trainer.predict_cell_type(x, perturbation_ids, experiment_ids)
    
    # Calculate accuracy
    accuracy = (predicted_cell_types == true_cell_types).float().mean().item()
    
    print(f"Cell type prediction accuracy: {accuracy:.4f}")
    return accuracy


def test_perturbation_analysis(model: ConditionalVAE,
                             test_data: Dict[str, torch.Tensor],
                             vocabularies: Dict[str, Dict],
                             config: VAEConfig,
                             device: str = 'cpu'):
    """Test perturbation effect analysis."""
    
    print("\nTesting perturbation analysis...")
    
    model.eval()
    trainer = VAETrainer(model, config, device)
    
    # Find cells with perturbations
    control_id = vocabularies['perturbation']['control']
    perturbed_mask = test_data['perturbation_ids'] != control_id
    
    if not perturbed_mask.any():
        print("No perturbed cells found in test data")
        return
    
    # Get perturbed cells
    x_perturbed = test_data['x'][perturbed_mask]
    cell_type_ids = test_data['cell_type_ids'][perturbed_mask]
    experiment_ids = test_data['experiment_ids'][perturbed_mask]
    
    # Predict control state
    x_control_pred = trainer.predict_perturbation_effect(
        x_perturbed, cell_type_ids, experiment_ids, control_id
    )
    
    # Compute fold changes
    fold_changes = torch.log2((x_perturbed + 1) / (x_control_pred + 1))
    
    print(f"Analyzed {len(x_perturbed)} perturbed cells")
    print(f"Mean absolute fold change: {torch.abs(fold_changes).mean():.4f}")
    print(f"Max fold change: {torch.abs(fold_changes).max():.4f}")


def test_generation(model: ConditionalVAE,
                   vocabularies: Dict[str, Dict],
                   config: VAEConfig,
                   device: str = 'cpu'):
    """Test synthetic cell generation."""
    
    print("\nTesting synthetic cell generation...")
    
    model.eval()
    
    # Generate cells for each cell type
    for cell_type_name, cell_type_id in vocabularies['cell_type'].items():
        generated = model.generate(
            cell_type_ids=torch.tensor([cell_type_id]),
            perturbation_ids=torch.tensor([0]),  # control
            experiment_ids=torch.tensor([0]),    # first experiment
            n_samples=10
        )
        
        print(f"{cell_type_name}: Generated {len(generated)} cells, "
              f"mean expression: {generated.mean():.4f}, "
              f"std: {generated.std():.4f}")


def test_latent_analysis(model: ConditionalVAE,
                       test_data: Dict[str, torch.Tensor],
                       config: VAEConfig,
                       device: str = 'cpu'):
    """Test latent space analysis."""
    
    print("\nTesting latent space analysis...")
    
    model.eval()
    
    with torch.no_grad():
        # Encode test data
        mu, logvar = model.encode(
            test_data['x'],
            test_data['cell_type_ids'],
            test_data['perturbation_ids'],
            test_data['experiment_ids']
        )
        
        # Analyze latent representations
        print(f"Latent space shape: {mu.shape}")
        print(f"Latent mean: {mu.mean(dim=0).mean():.4f}")
        print(f"Latent std: {mu.std(dim=0).mean():.4f}")
        
        # Check if different cell types cluster in latent space
        for cell_type in range(config.n_cell_types):
            mask = test_data['cell_type_ids'] == cell_type
            if mask.any():
                cell_type_latent = mu[mask]
                centroid = cell_type_latent.mean(dim=0)
                print(f"Cell type {cell_type} latent centroid norm: {torch.norm(centroid):.4f}")


def main():
    """Main example function."""
    
    print("VAE Example with Synthetic Data")
    print("=" * 40)
    
    # Configuration
    config = VAEConfig(
        input_dim=2000,
        latent_dim=64,  # Smaller for faster training
        hidden_dims=[256, 128],
        n_cell_types=5,
        n_perturbations=3,
        n_experiments=10,
        learning_rate=1e-3,
        batch_size=64
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    train_data = create_synthetic_data(
        n_cells=1000,
        n_genes=config.input_dim,
        n_cell_types=config.n_cell_types,
        n_perturbations=config.n_perturbations,
        n_experiments=config.n_experiments
    )
    
    test_data = create_synthetic_data(
        n_cells=200,
        n_genes=config.input_dim,
        n_cell_types=config.n_cell_types,
        n_perturbations=config.n_perturbations,
        n_experiments=config.n_experiments
    )
    
    vocabularies = create_vocabularies(
        config.n_cell_types, config.n_perturbations, config.n_experiments
    )
    
    print(f"Created training data: {len(train_data['x'])} cells")
    print(f"Created test data: {len(test_data['x'])} cells")
    
    # Move data to device
    for key in train_data:
        train_data[key] = train_data[key].to(device)
        test_data[key] = test_data[key].to(device)
    
    # Train VAE
    model = train_example_vae(train_data, config, n_epochs=50, device=device)
    
    # Test functionality
    accuracy = test_cell_type_prediction(model, test_data, config, device)
    test_perturbation_analysis(model, test_data, vocabularies, config, device)
    test_generation(model, vocabularies, config, device)
    test_latent_analysis(model, test_data, config, device)
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print(f"Final cell type prediction accuracy: {accuracy:.4f}")
    
    if accuracy > 0.7:
        print("✓ Model successfully learned to distinguish cell types")
    else:
        print("⚠ Model had difficulty with cell type prediction")
        print("  This is expected with synthetic data - try with real data for better results")


if __name__ == "__main__":
    main()
