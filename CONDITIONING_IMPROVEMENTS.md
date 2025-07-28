# Expressive Conditioning Implementation Summary

## Overview
Implemented o3's suggestion to replace the discrete 3-class perturbation type embedding with a more expressive continuous conditioning scheme that captures perturbation magnitude, direction, and optional context.

## Key Changes

### 1. Model Architecture Updates

#### Configuration Changes (`ConditionalModelConfig`)
- **Removed**: Single `perturb_embed_dim` for discrete types
- **Added**:
  - `perturb_sign_dim`: Embedding for perturbation direction (-1, 0, +1)
  - `perturb_magnitude_dim`: Hidden dimension for continuous magnitude processing
  - `perturb_context_dim`: Optional context features (time, concentration, etc.)
  - `magnitude_clip`: Clipping range for log2 fold changes (default: ±5)

#### Model Components (`ConditionalDiffusionTransformer`)
- **Replaced** discrete `perturb_type_embed` with:
  - `perturb_sign_embed`: Learnable embeddings for direction
  - `perturb_magnitude`: MLP for processing continuous log2 fold changes
  - `perturb_context`: Optional MLP for additional context
  - `null_perturb`: Null embedding for classifier-free guidance

### 2. Conditioning Interface

#### Forward Method Signature
```python
def forward(
    tokens, timesteps,
    target_gene_idx,        # Single or multiple genes
    perturb_magnitude,      # Continuous log2 fold change
    perturb_sign,          # Direction: -1, 0, +1
    perturb_context,       # Optional context features
    use_null_conditioning  # For classifier-free guidance
)
```

### 3. Key Features

#### Continuous Magnitude Representation
- Log2 fold changes normalized to [-1, 1] range
- Processed through dedicated MLP
- Captures effect size variability (e.g., 30% vs 90% knockdown)

#### Multi-Perturbation Support
- Handles single or multiple simultaneous perturbations
- Set-based aggregation (mean pooling) for combinations
- Order-invariant representation

#### Classifier-Free Guidance
- Random conditioning dropout during training
- Guidance scale during inference for stronger effects
- Null perturbation embedding for unconditional generation

### 4. Training Updates

#### Loss Computation
- Added `classifier_free_prob` parameter (default: 0.1)
- Randomly drops conditioning to enable guidance

#### Realistic Perturbation Sampling
- 70% knockdowns (magnitude: -0.5 to -3.0)
- 20% activations (magnitude: +0.5 to +3.0)
- 10% controls (magnitude: 0)

### 5. Inference Capabilities

#### Variable Effect Sizes
```python
# Generate cells with different knockdown strengths
prepare_perturbation_conditioning(
    "TP53", gene_to_idx, 
    perturbation_type="knockdown",
    magnitude=-2.0  # Specific 4-fold knockdown
)
```

#### Combinatorial Perturbations
```python
# Double knockdown
prepare_perturbation_conditioning(
    ["TP53", "MYC"], gene_to_idx,
    perturbation_type="knockdown"
)
```

#### Mixed Perturbations
```python
# Knockdown + activation
gene_indices = torch.tensor([idx_TP53, idx_STAT3])
magnitudes = torch.tensor([[-2.0, 2.0]])  # KD, Act
signs = torch.tensor([[-1, 1]])
```

## Benefits

1. **Generalization**: Model trained on 50% knockdowns can extrapolate to 80%
2. **Flexibility**: Handles drugs, CRISPRa, CRISPRi with same framework
3. **Precision**: Captures continuous variation in perturbation effects
4. **Composability**: Natural support for combinatorial perturbations
5. **Future-proof**: Extensible with context features (time, dose, etc.)

## Usage Example

```python
# Prepare conditioning for 2-fold knockdown of TP53
gene_idx, magnitude, sign = prepare_perturbation_conditioning(
    "TP53", 
    gene_to_idx,
    perturbation_type="knockdown",
    magnitude=-1.0,  # log2(0.5) = -1
    batch_size=100
)

# Generate with classifier-free guidance
samples = diffusion.p_sample_loop(
    model,
    shape=(100, n_genes),
    target_gene_idx=gene_idx,
    perturb_magnitude=magnitude,
    perturb_sign=sign,
    guidance_scale=2.0  # Stronger perturbation effect
)
```

## Backward Compatibility
The old discrete `perturb_type` interface is removed. Models need retraining with the new conditioning scheme to benefit from the expressive representation. 