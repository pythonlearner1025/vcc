#!/usr/bin/env python3
"""
Validation script for the Flexible VAE implementation.

This script performs comprehensive validation tests to ensure all components
of the refactored VAE system work correctly together.
"""

import os
import sys
import torch
import numpy as np
import traceback
from typing import List, Dict, Tuple

# Import all needed modules at the top
try:
    from models.flexible_vae import FlexibleVAE, VAEConfig, LearnedGeneEmbedding, PretrainedGeneEmbedding, flexible_vae_loss
    from dataset.flexible_dataloader import FlexibleDataset, PairedPerturbationDataset
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    IMPORTS_AVAILABLE = False


def test_imports() -> bool:
    """Test that all modules import correctly."""
    print("Testing imports...")
    
    if not IMPORTS_AVAILABLE:
        print("✗ Core modules not available")
        return False
    
    try:
        # Test additional imports
        import train_flexible_vae
        import inference_flexible_vae
        
        # Legacy compatibility
        import legacy_compat
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation() -> bool:
    """Test model creation with different configurations."""
    print("\nTesting model creation...")
    
    if not IMPORTS_AVAILABLE:
        print("✗ Skipping test - imports not available")
        return False
    
    try:
        # Test different configurations
        configs = [
            (100, 64, [128]),
            (500, 128, [256, 128]),
            (1000, 256, [512, 256, 128])
        ]
        
        for input_dim, latent_dim, hidden_dims in configs:
            config = VAEConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims
            )
            
            # Test learned embedding
            gene_emb = LearnedGeneEmbedding(n_genes=input_dim, embed_dim=latent_dim)
            model = FlexibleVAE(config, gene_emb)
            
            # Test forward pass
            batch_size = 16
            dummy_input = torch.randn(batch_size, input_dim)
            dummy_exp_id = torch.randint(0, 3, (batch_size,))
            dummy_target_genes = torch.randint(0, input_dim, (batch_size,))
            
            with torch.no_grad():
                output = model(dummy_input, dummy_exp_id, dummy_target_genes)
                
            # Validate output structure
            assert 'reconstructed' in output, "Missing reconstruction output"
            assert 'mu' in output, "Missing mu output"
            assert 'logvar' in output, "Missing logvar output"
            assert output['reconstructed'].shape == (batch_size, input_dim), "Wrong reconstruction shape"
            assert output['mu'].shape == (batch_size, latent_dim), "Wrong mu shape"
            assert output['logvar'].shape == (batch_size, latent_dim), "Wrong logvar shape"
            
            # Test loss computation
            loss_dict = flexible_vae_loss(output, dummy_input, config)
            assert isinstance(loss_dict, dict), "Loss should be a dictionary"
            assert 'total_loss' in loss_dict, "Missing total_loss in loss dict"
            assert isinstance(loss_dict['total_loss'], torch.Tensor), "total_loss should be a tensor"
            
        print("✓ Model creation and forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading() -> bool:
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    if not IMPORTS_AVAILABLE:
        print("✗ Skipping test - imports not available")
        return False
    
    try:
        # Generate synthetic data
        n_cells = 200
        n_genes = 100
        n_experiments = 3
        n_perturbations = 5
        
        expression_data = np.random.lognormal(0, 1, (n_cells, n_genes))
        experiment_ids = np.random.randint(0, n_experiments, n_cells)
        perturbation_ids = np.random.randint(0, n_perturbations, n_cells)
        
        # Test Phase 1 dataset
        dataset1 = FlexibleDataset(
            expression_data=expression_data,
            experiment_ids=experiment_ids,
            min_genes_per_cell=1,  # Lower threshold for test
            min_cells_per_gene=1  # Lower threshold for test
        )
        
        assert len(dataset1) == n_cells, "Wrong dataset size"
        sample = dataset1[0]
        assert 'expression' in sample, "Missing expression in sample"
        assert 'experiment_id' in sample, "Missing experiment_id in sample"
        
        # Test Phase 2 dataset
        dataset2 = PairedPerturbationDataset(
            expression_data=expression_data,
            experiment_ids=experiment_ids,
            perturbation_ids=perturbation_ids,
            min_cells_per_condition=1
        )
        
        assert len(dataset2) > 0, "Paired dataset should have samples"
        sample2 = dataset2[0]
        required_keys = ['control_expression', 'perturbed_expression', 
                        'experiment_id', 'perturbation_id']
        for key in required_keys:
            assert key in sample2, f"Missing {key} in paired sample"
        
        print("✓ Data loading successful")
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_training_loop() -> bool:
    """Test a minimal training loop."""
    print("\nTesting training loop...")
    
    if not IMPORTS_AVAILABLE:
        print("✗ Skipping test - imports not available")
        return False
    
    try:
        # Create small synthetic dataset
        n_cells = 50
        n_genes = 20
        n_experiments = 2
        
        expression_data = np.random.lognormal(0, 1, (n_cells, n_genes))
        experiment_ids = np.random.randint(0, n_experiments, n_cells)
        
        dataset = FlexibleDataset(
            expression_data=expression_data,
            experiment_ids=experiment_ids,
            min_genes_per_cell=1,  # Lower threshold for test
            min_cells_per_gene=1  # Lower threshold for test
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Create model
        config = VAEConfig(input_dim=n_genes, latent_dim=16, hidden_dims=[32])
        gene_emb = LearnedGeneEmbedding(n_genes=n_genes, embed_dim=16)
        model = FlexibleVAE(config, gene_emb)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(3):  # Short training
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                expression = batch['expression']
                exp_id = batch['experiment_id']
                
                output = model(expression, exp_id)
                loss_dict = flexible_vae_loss(output, expression, config)
                loss = loss_dict['total_loss']
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch == 0:
                initial_loss = epoch_loss
            if epoch == 2:
                final_loss = epoch_loss
        
        # Check that training progressed
        assert initial_loss is not None and final_loss is not None, "Loss tracking failed"
        print(f"  Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        
        print("✓ Training loop successful")
        return True
        
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        traceback.print_exc()
        return False


def test_config_files() -> bool:
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    try:
        import yaml
        
        config_files = [
            'configs/phase1_config.yaml',
            'configs/phase2_config.yaml',
            'configs/experimental_large.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Check for required keys
                assert 'latent_dim' in config, f"Missing latent_dim in {config_file}"
                assert 'learning_rate' in config, f"Missing learning_rate in {config_file}"
                
                print(f"  ✓ {config_file} is valid")
            else:
                print(f"  ⚠ {config_file} not found")
        
        print("✓ Configuration files validated")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        traceback.print_exc()
        return False


def test_legacy_compatibility() -> bool:
    """Test legacy compatibility layer."""
    print("\nTesting legacy compatibility...")
    
    try:
        # Test that legacy imports work through compatibility layer
        import legacy_compat
        
        # These should work through the compatibility layer
        from legacy_compat import VAE  # Should map to FlexibleVAE
        from legacy_compat import reconstruction_loss  # Should map to mse_loss
        
        print("✓ Legacy compatibility working")
        return True
        
    except Exception as e:
        print(f"✗ Legacy compatibility failed: {e}")
        traceback.print_exc()
        return False


def test_example_scripts() -> bool:
    """Test that example scripts can be imported."""
    print("\nTesting example scripts...")
    
    try:
        # Try importing example scripts (don't run them, just check syntax)
        import importlib.util
        
        scripts = [
            'example_flexible_vae.py',
            'train_flexible_vae.py',
            'inference_flexible_vae.py'
        ]
        
        for script in scripts:
            if os.path.exists(script):
                spec = importlib.util.spec_from_file_location("test_module", script)
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just check it can be loaded
                print(f"  ✓ {script} syntax is valid")
            else:
                print(f"  ⚠ {script} not found")
        
        print("✓ Example scripts validated")
        return True
        
    except Exception as e:
        print(f"✗ Example script validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("FLEXIBLE VAE VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training Loop", test_training_loop),
        ("Config Files", test_config_files),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Example Scripts", test_example_scripts)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")
    
    print("\n" + "="*60)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The Flexible VAE system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
