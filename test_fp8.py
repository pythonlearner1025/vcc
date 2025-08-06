#!/usr/bin/env python3
"""
Test script to diagnose FP8 performance issues.
"""

import torch
import time
import os
import numpy as np
from models.diffusion import (
    ConditionalModelConfig,
    ConditionalDiffusionTransformer,
    PartialMaskingDiffusion,
    HAS_TE
)

# Enable TF32 for fair comparison
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def benchmark_model(use_fp8=False, batch_size=128, n_iterations=10):
    """Benchmark model with and without FP8."""
    
    # Set environment variable for FP8
    os.environ['TE'] = '1' if use_fp8 else '0'
    
    # Reload the module to pick up the environment variable change
    import importlib
    import models.diffusion as diffusion_module
    importlib.reload(diffusion_module)
    
    # Create config - matching your actual model size
    config = ConditionalModelConfig(
        dim=1024,
        n_head=8,
        n_layer=8,
        ffn_mult=4,
        n_genes=6288,
        gene_embed_dim=128,
        n_technical_batches=48,
        batch_embed_dim=64,
        use_batch_conditioning=True,
        control_set_encoder_layers=2,
        control_set_dim_hidden=256,
        use_fp8=use_fp8,
        grad_ckpt=False,  # Disable checkpointing for true kernel speed benchmarking
        esm_matrix_path=None,  # Disable ESM for testing
    )
    
    # Create model
    model = diffusion_module.ConditionalDiffusionTransformer(config).cuda()
    diffusion = diffusion_module.PartialMaskingDiffusion(config)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Testing with FP8={'ENABLED' if use_fp8 else 'DISABLED'}")
    print(f"Model parameters: {n_params:,}")
    print(f"Batch size: {batch_size}")
    print(f"HAS_TE: {diffusion_module.HAS_TE}")
    print(f"{'='*60}")
    
    # Create dummy data
    tokens = torch.randint(0, 64, (batch_size, config.n_genes)).cuda()
    control_set = torch.randn(batch_size, 64, config.n_genes).cuda()
    target_gene_idx = torch.randint(0, config.n_total_genes, (batch_size,)).cuda()
    batch_idx = torch.randint(0, config.n_technical_batches, (batch_size,)).cuda()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if not use_fp8 else diffusion_module.fp8_autocast(enabled=True):
            loss = diffusion.compute_loss(
                model, tokens, 
                control_set=control_set,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx
            )
        loss.backward()
        torch.cuda.synchronize()
    
    # Clear gradients
    model.zero_grad()
    
    # Benchmark forward pass
    print(f"\nBenchmarking {n_iterations} iterations...")
    forward_times = []
    backward_times = []
    
    for i in range(n_iterations):
        torch.cuda.synchronize()
        
        # Forward pass
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if not use_fp8 else diffusion_module.fp8_autocast(enabled=True):
            loss = diffusion.compute_loss(
                model, tokens,
                control_set=control_set,
                target_gene_idx=target_gene_idx,
                batch_idx=batch_idx
            )
        torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)
        
        # Backward pass
        t2 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        backward_times.append((t3 - t2) * 1000)
        
        # Clear gradients for next iteration
        model.zero_grad()
        
        if i == 0:
            print(f"  First iteration - Forward: {forward_times[0]:.1f}ms, Backward: {backward_times[0]:.1f}ms")
    
    # Report results
    print(f"\nResults (excluding first iteration):")
    print(f"  Forward:  {np.mean(forward_times[1:]):.1f} ± {np.std(forward_times[1:]):.1f} ms")
    print(f"  Backward: {np.mean(backward_times[1:]):.1f} ± {np.std(backward_times[1:]):.1f} ms")
    print(f"  Total:    {np.mean(forward_times[1:]) + np.mean(backward_times[1:]):.1f} ms")
    
    # Memory usage
    print(f"\nMemory usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return {
        'forward_mean': np.mean(forward_times[1:]),
        'backward_mean': np.mean(backward_times[1:]),
        'memory_gb': torch.cuda.memory_allocated() / 1024**3
    }

def main():
    """Run benchmarks."""
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name())
    
    # Test different configurations
    configs = [
        (False, 32),   # BF16, batch_size=64
    ]
    
    # Only test FP8 if TE is available
    if HAS_TE:
        configs.extend([
            (True, 32),    # FP8, batch_size=64
        ])
    else:
        print("\n⚠️  Transformer Engine not available, skipping FP8 tests")
    
    results = []
    for use_fp8, batch_size in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        result = benchmark_model(use_fp8=use_fp8, batch_size=batch_size)
        result['fp8'] = use_fp8
        result['batch_size'] = batch_size
        results.append(result)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'Memory (GB)':<15}")
    print("-"*80)
    
    for r in results:
        config_str = f"{'FP8' if r['fp8'] else 'BF16'} BS={r['batch_size']}"
        total = r['forward_mean'] + r['backward_mean']
        print(f"{config_str:<20} {r['forward_mean']:<15.1f} {r['backward_mean']:<15.1f} {total:<15.1f} {r['memory_gb']:<15.2f}")
    
    # Calculate speedup if FP8 results exist
    if len(results) >= 4:
        bf16_64 = results[0]['forward_mean'] + results[0]['backward_mean']
        bf16_128 = results[1]['forward_mean'] + results[1]['backward_mean']
        fp8_64 = results[2]['forward_mean'] + results[2]['backward_mean']
        fp8_128 = results[3]['forward_mean'] + results[3]['backward_mean']
        
        print("\n" + "="*80)
        print("SPEEDUP ANALYSIS")
        print("="*80)
        print(f"BS=64:  FP8 is {bf16_64/fp8_64:.2f}x {'faster' if fp8_64 < bf16_64 else 'slower'}")
        print(f"BS=128: FP8 is {bf16_128/fp8_128:.2f}x {'faster' if fp8_128 < bf16_128 else 'slower'}")

if __name__ == "__main__":
    main()