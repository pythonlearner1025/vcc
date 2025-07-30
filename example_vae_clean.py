#!/usr/bin/env python3
"""
Example usage of the VAE for predicting perturbation effects.

This script demonstrates how to:
1. Load and train the VAE on paired control/perturbation data
2. Use the model to predict perturbation effects 
3. Analyze latent representations
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig
from dataset.vae_paired_dataloader import VAEPairedDataset, load_vae_data


def create_synthetic_data(n_experiments=10, n_perturbations=5, n_cells_per_condition=100, n_genes=1000):
    """Create synthetic data for demonstration."""
    np.random.seed(42)
    
    # Create synthetic gene expression data
    all_expression = []
    all_experiment_ids = []
    all_perturbation_ids = []
    
    control_id = 0  # Control perturbation ID
    
    for exp_id in range(n_experiments):
        for pert_id in range(n_perturbations):
            # Base expression profile for this experiment (batch effect)
            base_expression = np.random.normal(0, 1, n_genes)
            
            # Perturbation effect (only for non-control)
            if pert_id == control_id:
                pert_effect = np.zeros(n_genes)
            else:
                # Different perturbations affect different genes
                pert_effect = np.zeros(n_genes)
                # Each perturbation affects 100 random genes
                affected_genes = np.random.choice(n_genes, 100, replace=False)
                pert_effect[affected_genes] = np.random.normal(0, 2, 100)
            
            # Generate cells for this condition
            for _ in range(n_cells_per_condition):
                # Add noise to each cell
                cell_expression = base_expression + pert_effect + np.random.normal(0, 0.5, n_genes)
                
                all_expression.append(cell_expression)
                all_experiment_ids.append(exp_id)
                all_perturbation_ids.append(pert_id)
    
    expression_data = np.array(all_expression)
    experiment_ids = np.array(all_experiment_ids)
    perturbation_ids = np.array(all_perturbation_ids)
    
    print(f"Created synthetic data:")
    print(f"  Expression shape: {expression_data.shape}")
    print(f"  N experiments: {len(np.unique(experiment_ids))}")
    print(f"  N perturbations: {len(np.unique(perturbation_ids))}")
    print(f"  N cells: {len(expression_data)}")
    
    return expression_data, experiment_ids, perturbation_ids


def train_vae_example():
    """Train VAE on synthetic data."""
    # Create synthetic data
    expression_data, experiment_ids, perturbation_ids = create_synthetic_data()
    
    # Create dataset
    dataset = VAEPairedDataset(
        expression_data=expression_data,
        experiment_ids=experiment_ids,
        perturbation_ids=perturbation_ids,
        min_cells_per_pair=10
    )
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True
    )
    
    # Create model config
    config = VAEConfig(
        input_dim=expression_data.shape[1],  # Number of genes
        latent_dim=50,  # Smaller for demo
        hidden_dims=[256, 128],
        n_experiments=len(np.unique(experiment_ids)),
        n_perturbations=len(np.unique(perturbation_ids)),
        learning_rate=1e-3,
        batch_size=64
    )
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConditionalVAE(config).to(device)
    
    print(f"Model initialized on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(20):  # Short training for demo
        total_loss = 0
        for batch in train_loader:
            expression = batch['expression'].to(device)
            experiment_ids = batch['experiment_id'].to(device)
            perturbation_ids = batch['perturbation_id'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(expression, experiment_ids, perturbation_ids)
            
            # Simple loss (reconstruction only for demo)
            recon_loss = torch.nn.functional.mse_loss(outputs['reconstructed'], expression)
            kld_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
            kld_loss = kld_loss / expression.size(0)  # Normalize by batch size
            
            loss = recon_loss + 0.1 * kld_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
    
    return model, dataset, config


def demonstrate_perturbation_prediction(model, dataset, config):
    """Demonstrate how to predict perturbation effects."""
    model.eval()
    device = next(model.parameters()).device
    
    # Get a paired example
    exp_id = 0
    pert_id = 1  # Non-control perturbation
    
    try:
        paired_data = dataset.get_paired_examples(exp_id, pert_id, n_samples=5)
    except ValueError as e:
        print(f"Could not get paired examples: {e}")
        return
    
    control_expression = paired_data['control_expression'].to(device)
    control_exp_ids = paired_data['control_experiment_ids'].to(device)
    control_pert_ids = paired_data['control_perturbation_ids'].to(device)
    
    perturbed_expression = paired_data['perturbed_expression'].to(device)
    perturbed_exp_ids = paired_data['perturbed_experiment_ids'].to(device)
    perturbed_pert_ids = paired_data['perturbed_perturbation_ids'].to(device)
    
    with torch.no_grad():
        # Encode control cells (experiment conditioning only)
        control_mu, control_logvar = model.encode(control_expression, control_exp_ids)
        
        # Use the latent representation to predict perturbation effect
        # Decode with perturbation conditioning
        predicted_perturbed = model.decode(control_mu, perturbed_pert_ids, perturbed_exp_ids)
        
        # Also get actual reconstructions
        control_recon = model.decode(control_mu, control_pert_ids, control_exp_ids)
        
        # Encode actual perturbed cells for comparison
        perturbed_mu, _ = model.encode(perturbed_expression, perturbed_exp_ids)
        actual_perturbed_recon = model.decode(perturbed_mu, perturbed_pert_ids, perturbed_exp_ids)
    
    # Calculate perturbation effect
    predicted_effect = predicted_perturbed - control_recon
    actual_effect = perturbed_expression - control_expression
    
    # Compute correlation
    pred_effect_flat = predicted_effect.cpu().flatten()
    actual_effect_flat = actual_effect.cpu().flatten()
    correlation = torch.corrcoef(torch.stack([pred_effect_flat, actual_effect_flat]))[0, 1]
    
    print(f"\nPerturbation prediction results:")
    print(f"  Experiment ID: {exp_id}")
    print(f"  Perturbation ID: {pert_id}")
    print(f"  Correlation between predicted and actual effect: {correlation:.3f}")
    print(f"  Mean absolute predicted effect: {predicted_effect.abs().mean():.3f}")
    print(f"  Mean absolute actual effect: {actual_effect.abs().mean():.3f}")
    
    return {
        'predicted_effect': predicted_effect.cpu().numpy(),
        'actual_effect': actual_effect.cpu().numpy(),
        'correlation': correlation.item()
    }


def analyze_latent_space(model, dataset, config):
    """Analyze the latent space representation."""
    model.eval()
    device = next(model.parameters()).device
    
    # Sample some data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    batch = next(iter(data_loader))
    
    expression = batch['expression'].to(device)
    experiment_ids = batch['experiment_id'].to(device)
    perturbation_ids = batch['perturbation_id'].to(device)
    
    with torch.no_grad():
        # Encode to latent space
        mu, logvar = model.encode(expression, experiment_ids)
        z = model.reparameterize(mu, logvar)
    
    # Convert to numpy for analysis
    z_np = z.cpu().numpy()
    exp_ids_np = experiment_ids.cpu().numpy()
    pert_ids_np = perturbation_ids.cpu().numpy()
    
    # PCA for visualization
    if z_np.shape[1] > 2:
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_np)
        print(f"\nLatent space analysis:")
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        z_pca = z_np
    
    return {
        'latent_representations': z_np,
        'latent_pca': z_pca,
        'experiment_ids': exp_ids_np,
        'perturbation_ids': pert_ids_np
    }


def main():
    """Run the complete example."""
    print("VAE Perturbation Prediction Example")
    print("=" * 40)
    
    # Train model
    print("\n1. Training VAE on synthetic data...")
    model, dataset, config = train_vae_example()
    
    # Demonstrate perturbation prediction
    print("\n2. Demonstrating perturbation prediction...")
    pred_results = demonstrate_perturbation_prediction(model, dataset, config)
    
    # Analyze latent space
    print("\n3. Analyzing latent space...")
    latent_results = analyze_latent_space(model, dataset, config)
    
    print("\n4. Key insights:")
    print("   - The encoder only sees gene expression + experiment ID (no perturbation)")
    print("   - Perturbation information is injected only at the decoder")
    print("   - The model learns to predict perturbation effects in latent space")
    print("   - Cell types emerge naturally without explicit conditioning")
    print(f"   - Perturbation prediction correlation: {pred_results['correlation']:.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
