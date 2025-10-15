#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt

# Simulate soft targets (Gaussian-like distributions)
torch.manual_seed(42)
N, K = 4, 63  # 4 genes, 63 bins
bins = torch.linspace(-5, 5, K)

# Create sample soft targets
soft_targets = torch.zeros(N, K)
for i in range(N):
    mean = torch.randn(1) * 2
    std = 0.5 + torch.rand(1) * 1.5
    soft_targets[i] = torch.exp(-0.5 * ((bins - mean) / std) ** 2)
    soft_targets[i] /= soft_targets[i].sum()

# Apply sharpening
target_gamma = 10.0
st = soft_targets.clamp_min(1e-12).pow(target_gamma)
st = st / st.sum(dim=-1, keepdim=True)

# Compute effective bins
def effective_bins(p):
    p = p.clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=-1)
    return torch.exp(H)

eff_orig = effective_bins(soft_targets)
eff_sharp = effective_bins(st)

# Plot
fig, axes = plt.subplots(2, N, figsize=(16, 6))
for i in range(N):
    # Original
    axes[0, i].bar(range(K), soft_targets[i].numpy(), color='blue', alpha=0.6)
    axes[0, i].set_title(f'Gene {i+1} Original\nEff bins: {eff_orig[i]:.1f}')
    axes[0, i].set_ylim(0, max(soft_targets.max(), st.max()) * 1.1)
    
    # Sharpened
    axes[1, i].bar(range(K), st[i].numpy(), color='red', alpha=0.6)
    axes[1, i].set_title(f'Sharpened (γ={target_gamma})\nEff bins: {eff_sharp[i]:.1f}')
    axes[1, i].set_ylim(0, max(soft_targets.max(), st.max()) * 1.1)
    
    if i == 0:
        axes[0, i].set_ylabel('Original Prob')
        axes[1, i].set_ylabel('Sharpened Prob')
    axes[1, i].set_xlabel('Bin')

plt.suptitle(f'Distribution Sharpening Effect (γ={target_gamma})', fontsize=14)
plt.tight_layout()
plt.savefig('sharpening_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean effective bins - Original: {eff_orig.mean():.2f}, Sharpened: {eff_sharp.mean():.2f}")
print(f"Reduction factor: {eff_orig.mean() / eff_sharp.mean():.2f}x")