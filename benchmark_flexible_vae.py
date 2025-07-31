#!/usr/bin/env python3
"""
Benchmarking utility for the Flexible VAE architecture.

This script helps users understand computational requirements and performance
characteristics of different model configurations.
"""

import torch
import time
import numpy as np
import argparse
import psutil
import os
from typing import Dict, List, Tuple
from models.flexible_vae import FlexibleVAE, VAEConfig, LearnedGeneEmbedding
from dataset.flexible_dataloader import FlexibleDataset
from torch.utils.data import DataLoader


def benchmark_model_sizes():
    """Benchmark different model configurations."""
    print("="*60)
    print("FLEXIBLE VAE ARCHITECTURE BENCHMARKS")
    print("="*60)
    
    # Define different configurations to test
    configs = [
        ("Small (laptop-friendly)", 1000, 128, [512, 256]),
        ("Medium (workstation)", 2000, 512, [1024, 512, 256]),
        ("Large (server/cluster)", 5000, 1024, [2048, 1024, 512, 256]),
        ("Extra Large (HPC)", 10000, 2048, [4096, 2048, 1024, 512])
    ]
    
    results = []
    
    for name, input_dim, latent_dim, hidden_dims in configs:
        print(f"\n{name}:")
        print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        
        try:
            # Create model
            config = VAEConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims
            )
            gene_emb = LearnedGeneEmbedding(n_genes=input_dim, embed_dim=latent_dim)
            model = FlexibleVAE(config, gene_emb)
            
            # Calculate parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough approximation)
            param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32, convert to MB
            
            print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
            print(f"  Model memory: ~{param_memory:.1f} MB")
            
            # Benchmark forward pass
            batch_size = 32
            dummy_input = torch.randn(batch_size, input_dim)
            dummy_exp_id = torch.randint(0, 3, (batch_size,))
            dummy_target_genes = torch.randint(0, input_dim, (batch_size,))
            
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                for _ in range(10):  # Average over 10 runs
                    output = model(dummy_input, dummy_exp_id, dummy_target_genes)
                end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = batch_size / avg_time
            
            print(f"  Forward pass: {avg_time*1000:.2f} ms/batch")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
            results.append({
                'name': name,
                'total_params': total_params,
                'memory_mb': param_memory,
                'forward_time_ms': avg_time * 1000,
                'throughput': throughput
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return results


def benchmark_training_speed():
    """Benchmark training speed with synthetic data."""
    print("\n" + "="*60)
    print("TRAINING SPEED BENCHMARKS")
    print("="*60)
    
    # Generate synthetic data
    n_cells = 1000
    n_genes = 1000
    n_experiments = 3
    
    print(f"Generating synthetic data: {n_cells} cells, {n_genes} genes...")
    
    # Create synthetic data
    expression_data = torch.randn(n_cells, n_genes).abs()  # Positive expression
    experiment_ids = torch.randint(0, n_experiments, (n_cells,))
    
    # Create dataset and dataloader
    dataset = FlexibleDataset(
        expression_data=expression_data.numpy(),
        experiment_ids=experiment_ids.numpy()
    )
    
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        if batch_size > n_cells // 4:  # Skip if batch size too large
            continue
            
        print(f"\nBatch size: {batch_size}")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        config = VAEConfig(input_dim=n_genes, latent_dim=256)
        gene_emb = LearnedGeneEmbedding(n_genes=n_genes, embed_dim=256)
        model = FlexibleVAE(config, gene_emb)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Benchmark training
        model.train()
        start_time = time.time()
        
        total_batches = 0
        for batch in dataloader:
            if total_batches >= 10:  # Limit to 10 batches for timing
                break
                
            expression = batch['expression']
            exp_id = batch['experiment_id']
            
            optimizer.zero_grad()
            output = model(expression, exp_id)
            loss = model.compute_loss(output, expression)
            loss.backward()
            optimizer.step()
            
            total_batches += 1
        
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_batch = total_time / total_batches
        samples_per_sec = (total_batches * batch_size) / total_time
        
        print(f"  Time per batch: {time_per_batch*1000:.2f} ms")
        print(f"  Samples/sec: {samples_per_sec:.1f}")


def benchmark_memory_usage():
    """Benchmark memory usage during training."""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARKS")
    print("="*60)
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**2)  # MB
    
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test different data sizes
    data_sizes = [
        ("Small dataset", 500, 500),
        ("Medium dataset", 2000, 1000),
        ("Large dataset", 5000, 2000)
    ]
    
    for name, n_cells, n_genes in data_sizes:
        print(f"\n{name}: {n_cells} cells, {n_genes} genes")
        
        try:
            # Create data
            expression_data = torch.randn(n_cells, n_genes).abs()
            experiment_ids = torch.randint(0, 3, (n_cells,))
            
            memory_after_data = process.memory_info().rss / (1024**2)
            print(f"  Memory after data creation: {memory_after_data:.1f} MB (+{memory_after_data-initial_memory:.1f} MB)")
            
            # Create model
            config = VAEConfig(input_dim=n_genes, latent_dim=256)
            gene_emb = LearnedGeneEmbedding(n_genes=n_genes, embed_dim=256)
            model = FlexibleVAE(config, gene_emb)
            
            memory_after_model = process.memory_info().rss / (1024**2)
            print(f"  Memory after model creation: {memory_after_model:.1f} MB (+{memory_after_model-memory_after_data:.1f} MB)")
            
            # Single forward pass
            model.eval()
            with torch.no_grad():
                batch = expression_data[:32]  # Use small batch
                exp_id = experiment_ids[:32]
                output = model(batch, exp_id)
            
            memory_after_forward = process.memory_info().rss / (1024**2)
            print(f"  Memory after forward pass: {memory_after_forward:.1f} MB (+{memory_after_forward-memory_after_model:.1f} MB)")
            
            # Clean up
            del expression_data, experiment_ids, model, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ERROR: {e}")


def generate_recommendation():
    """Generate hardware recommendations."""
    print("\n" + "="*60)
    print("HARDWARE RECOMMENDATIONS")
    print("="*60)
    
    recommendations = {
        "Laptop/Desktop (8-16GB RAM)": {
            "max_genes": 2000,
            "max_latent_dim": 512,
            "batch_size": 32,
            "notes": "Good for initial experiments and development"
        },
        "Workstation (32-64GB RAM)": {
            "max_genes": 5000,
            "max_latent_dim": 1024,
            "batch_size": 128,
            "notes": "Suitable for medium-scale experiments"
        },
        "Server/HPC (64GB+ RAM)": {
            "max_genes": 10000,
            "max_latent_dim": 2048,
            "batch_size": 256,
            "notes": "For large-scale experiments and production use"
        }
    }
    
    for hardware, specs in recommendations.items():
        print(f"\n{hardware}:")
        print(f"  Max genes: {specs['max_genes']:,}")
        print(f"  Max latent dim: {specs['max_latent_dim']}")
        print(f"  Recommended batch size: {specs['batch_size']}")
        print(f"  Notes: {specs['notes']}")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark Flexible VAE performance")
    parser.add_argument("--benchmark", 
                       choices=["model", "training", "memory", "all"],
                       default="all",
                       help="Type of benchmark to run")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks only")
    
    args = parser.parse_args()
    
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if args.benchmark in ["model", "all"]:
        benchmark_model_sizes()
    
    if args.benchmark in ["training", "all"] and not args.quick:
        benchmark_training_speed()
    
    if args.benchmark in ["memory", "all"] and not args.quick:
        benchmark_memory_usage()
    
    generate_recommendation()
    
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
