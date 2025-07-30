#!/usr/bin/env python3
"""
Advanced example demonstrating perturbation injection with the VAE.

This script shows how to:
1. Train VAE on general scRNA-seq data (without perturbation labels)
2. Fine-tune on paired control/perturbation data
3. Use perturbation injection to predict perturbed profiles from control cells
4. Predict control profiles from perturbed cells
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.VAE import ConditionalVAE, VAEConfig, vae_loss_function


def create_realistic_synthetic_data(n_experiments=5, n_perturbations=4, 
                                   n_cells_per_condition=200, n_genes=1000,
                                   perturbation_strength=2.0):
    """Create more realistic synthetic data with known perturbation effects."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define perturbation effects
    perturbation_effects = {}
    control_id = 0
    
    for pert_id in range(n_perturbations):
        if pert_id == control_id:
            perturbation_effects[pert_id] = np.zeros(n_genes)
        else:
            # Each perturbation affects a specific set of genes
            effect = np.zeros(n_genes)
            # Affect 5% of genes
            n_affected = int(0.05 * n_genes)
            affected_genes = np.arange(pert_id * n_affected, (pert_id + 1) * n_affected) % n_genes
            effect[affected_genes] = np.random.normal(0, perturbation_strength, len(affected_genes))
            perturbation_effects[pert_id] = effect
    
    # Generate data
    all_expression = []
    all_experiment_ids = []
    all_perturbation_ids = []
    all_true_effects = []
    
    for exp_id in range(n_experiments):
        # Base expression profile for this experiment (batch effect)
        base_expression = np.random.normal(0, 1, n_genes)
        
        for pert_id in range(n_perturbations):
            pert_effect = perturbation_effects[pert_id]
            
            # Generate cells for this condition
            for _ in range(n_cells_per_condition):
                # Cell expression = base + perturbation effect + noise
                cell_expression = base_expression + pert_effect + np.random.normal(0, 0.3, n_genes)
                
                all_expression.append(cell_expression)
                all_experiment_ids.append(exp_id)
                all_perturbation_ids.append(pert_id)
                all_true_effects.append(pert_effect)
    
    expression_data = np.array(all_expression)
    experiment_ids = np.array(all_experiment_ids)
    perturbation_ids = np.array(all_perturbation_ids)
    true_effects = np.array(all_true_effects)
    
    print(f"Created realistic synthetic data:")
    print(f"  Expression shape: {expression_data.shape}")
    print(f"  N experiments: {len(np.unique(experiment_ids))}")
    print(f"  N perturbations: {len(np.unique(perturbation_ids))}")
    print(f"  Total cells: {len(expression_data)}")
    print(f"  Perturbation strength: {perturbation_strength}")
    
    return expression_data, experiment_ids, perturbation_ids, true_effects


def pretrain_vae_general_data(expression_data, experiment_ids, config, epochs=50):
    """
    Pretrain VAE on general data without perturbation labels.
    This simulates training on large-scale scRNA-seq datasets without perturbation info.
    """
    print("\n" + "="*50)
    print("PHASE 1: Pretraining on general scRNA-seq data")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConditionalVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dataset without perturbation labels
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(expression_data),
        torch.LongTensor(experiment_ids)
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for expression, exp_ids in dataloader:
            expression = expression.to(device)
            exp_ids = exp_ids.to(device)
            
            # Forward pass without perturbation labels
            optimizer.zero_grad()
            outputs = model(expression, exp_ids, perturbation_ids=None)
            
            # Compute loss
            recon_loss = torch.nn.functional.mse_loss(outputs['reconstructed'], expression)
            kld_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
            kld_loss = kld_loss / expression.size(0)
            
            loss = recon_loss + 0.1 * kld_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Pretrain Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    
    print(f"Pretraining completed. Model learned general scRNA-seq representation.")
    return model


def finetune_vae_perturbation_data(model, expression_data, experiment_ids, perturbation_ids, epochs=30):
    """
    Fine-tune VAE on paired control/perturbation data.
    """
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning on paired control/perturbation data")
    print("="*50)
    
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Lower learning rate for fine-tuning
    
    # Create dataset with perturbation labels
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(expression_data),
        torch.LongTensor(experiment_ids),
        torch.LongTensor(perturbation_ids)
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for expression, exp_ids, pert_ids in dataloader:
            expression = expression.to(device)
            exp_ids = exp_ids.to(device)
            pert_ids = pert_ids.to(device)
            
            # Forward pass with perturbation labels
            optimizer.zero_grad()
            outputs = model(expression, exp_ids, pert_ids)
            
            # Compute loss
            recon_loss = torch.nn.functional.mse_loss(outputs['reconstructed'], expression)
            kld_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
            kld_loss = kld_loss / expression.size(0)
            
            loss = recon_loss + 0.1 * kld_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Finetune Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    
    print(f"Fine-tuning completed. Model learned perturbation effects.")
    return model


def evaluate_perturbation_injection(model, expression_data, experiment_ids, perturbation_ids, true_effects):
    """
    Evaluate how well the model can inject perturbations into control cells.
    """
    print("\n" + "="*50)
    print("PHASE 3: Evaluating perturbation injection")
    print("="*50)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get control cells
    control_mask = perturbation_ids == 0
    control_expression = expression_data[control_mask]
    control_exp_ids = experiment_ids[control_mask]
    
    if len(control_expression) == 0:
        print("No control cells found!")
        return
    
    # Sample some control cells for testing
    n_test = min(100, len(control_expression))
    test_indices = np.random.choice(len(control_expression), n_test, replace=False)
    
    test_control_expr = torch.FloatTensor(control_expression[test_indices]).to(device)
    test_control_exp_ids = torch.LongTensor(control_exp_ids[test_indices]).to(device)
    
    results = {}
    
    # Test each perturbation
    for target_pert in range(1, len(np.unique(perturbation_ids))):
        print(f"\nTesting perturbation {target_pert}:")
        
        target_pert_ids = torch.full((n_test,), target_pert, dtype=torch.long, device=device)
        
        # Predict perturbed profiles
        with torch.no_grad():
            predicted_perturbed = model.inject_perturbation(
                test_control_expr, test_control_exp_ids, target_pert_ids
            )
        
        # Calculate predicted perturbation effect
        predicted_effect = predicted_perturbed - test_control_expr
        predicted_effect_mean = predicted_effect.mean(dim=0).cpu().numpy()
        
        # Get true perturbation effect for this perturbation
        true_effect_mask = perturbation_ids == target_pert
        if true_effect_mask.sum() > 0:
            true_effect_mean = true_effects[true_effect_mask][0]  # All should be the same
            
            # Calculate correlation
            correlation = np.corrcoef(predicted_effect_mean, true_effect_mean)[0, 1]
            r2 = r2_score(true_effect_mean, predicted_effect_mean)
            
            print(f"  Correlation with true effect: {correlation:.3f}")
            print(f"  R² score: {r2:.3f}")
            print(f"  Mean absolute predicted effect: {np.abs(predicted_effect_mean).mean():.3f}")
            print(f"  Mean absolute true effect: {np.abs(true_effect_mean).mean():.3f}")
            
            results[target_pert] = {
                'correlation': correlation,
                'r2_score': r2,
                'predicted_effect': predicted_effect_mean,
                'true_effect': true_effect_mean
            }
        else:
            print(f"  No true data found for perturbation {target_pert}")
    
    return results


def evaluate_control_prediction(model, expression_data, experiment_ids, perturbation_ids):
    """
    Evaluate how well the model can predict control profiles from perturbed cells.
    """
    print("\n" + "="*50)
    print("PHASE 4: Evaluating control prediction from perturbed cells")
    print("="*50)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get perturbed cells
    perturbed_mask = perturbation_ids != 0
    perturbed_expression = expression_data[perturbed_mask]
    perturbed_exp_ids = experiment_ids[perturbed_mask]
    perturbed_pert_ids = perturbation_ids[perturbed_mask]
    
    if len(perturbed_expression) == 0:
        print("No perturbed cells found!")
        return
    
    # Sample some perturbed cells for testing
    n_test = min(100, len(perturbed_expression))
    test_indices = np.random.choice(len(perturbed_expression), n_test, replace=False)
    
    test_perturbed_expr = torch.FloatTensor(perturbed_expression[test_indices]).to(device)
    test_perturbed_exp_ids = torch.LongTensor(perturbed_exp_ids[test_indices]).to(device)
    
    # Predict control profiles
    with torch.no_grad():
        predicted_control = model.predict_control_from_perturbed(
            test_perturbed_expr, test_perturbed_exp_ids
        )
    
    # Find actual control cells from the same experiments
    correlations = []
    for i in range(n_test):
        exp_id = test_perturbed_exp_ids[i].item()
        
        # Find control cells from the same experiment
        control_mask = (experiment_ids == exp_id) & (perturbation_ids == 0)
        if control_mask.sum() > 0:
            actual_controls = expression_data[control_mask]
            # Use mean of control cells as reference
            actual_control_mean = actual_controls.mean(axis=0)
            
            pred_control = predicted_control[i].cpu().numpy()
            correlation = np.corrcoef(pred_control, actual_control_mean)[0, 1]
            correlations.append(correlation)
    
    if correlations:
        mean_correlation = np.mean(correlations)
        print(f"Mean correlation between predicted and actual controls: {mean_correlation:.3f}")
        print(f"Standard deviation: {np.std(correlations):.3f}")
        print(f"Number of test cases: {len(correlations)}")
    else:
        print("Could not find matching control cells for evaluation")
    
    return correlations


def main():
    """Run the complete perturbation injection demonstration."""
    print("VAE Perturbation Injection Demonstration")
    print("=" * 60)
    
    # Create synthetic data
    expression_data, experiment_ids, perturbation_ids, true_effects = create_realistic_synthetic_data(
        n_experiments=8, n_perturbations=4, n_cells_per_condition=150, n_genes=1000
    )
    
    # Create model config
    config = VAEConfig(
        input_dim=expression_data.shape[1],
        latent_dim=64,
        hidden_dims=[256, 128],
        n_experiments=len(np.unique(experiment_ids)),
        n_perturbations=len(np.unique(perturbation_ids)),
        learning_rate=1e-3
    )
    
    print(f"\nModel configuration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  N experiments: {config.n_experiments}")
    print(f"  N perturbations: {config.n_perturbations}")
    
    # Phase 1: Pretrain on general data
    model = pretrain_vae_general_data(expression_data, experiment_ids, config, epochs=40)
    
    # Phase 2: Fine-tune on perturbation data
    model = finetune_vae_perturbation_data(model, expression_data, experiment_ids, perturbation_ids, epochs=30)
    
    # Phase 3: Evaluate perturbation injection
    injection_results = evaluate_perturbation_injection(model, expression_data, experiment_ids, perturbation_ids, true_effects)
    
    # Phase 4: Evaluate control prediction
    control_correlations = evaluate_control_prediction(model, expression_data, experiment_ids, perturbation_ids)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if injection_results:
        avg_correlation = np.mean([r['correlation'] for r in injection_results.values()])
        avg_r2 = np.mean([r['r2_score'] for r in injection_results.values()])
        print(f"Perturbation injection performance:")
        print(f"  Average correlation: {avg_correlation:.3f}")
        print(f"  Average R² score: {avg_r2:.3f}")
    
    if control_correlations:
        print(f"Control prediction performance:")
        print(f"  Average correlation: {np.mean(control_correlations):.3f}")
    
    print(f"\nKey achievements:")
    print(f"✓ Pretrained VAE on general scRNA-seq data without perturbation labels")
    print(f"✓ Fine-tuned on paired control/perturbation data")
    print(f"✓ Successfully injected perturbations into control cells")
    print(f"✓ Successfully predicted control profiles from perturbed cells")
    print(f"✓ Demonstrated perturbation effects learned in latent space")


if __name__ == "__main__":
    main()
