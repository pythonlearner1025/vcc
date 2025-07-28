# ST-Style Conditional Diffusion for Single-Cell Perturbation Prediction

A PyTorch implementation of ST-style conditional discrete diffusion transformers for single-cell RNA-seq perturbation prediction. This codebase implements paired control-perturbed cell training with cross-attention conditioning.

## Overview

This repository implements:
- **ST-style architecture**: Conditional diffusion with control set cross-attention
- **Paired training**: Control and perturbed cell sets processed together
- **Adaptive masking**: Condition-aware noise scheduling
- **Combined HVGs**: Union of highly variable genes and perturbed genes for optimal feature selection

### Key Components

```
vcc/
├── models/
│   └── diffusion.py              # Core diffusion model with control set conditioning
├── dataset/
│   ├── vcc_dataloader.py        # Basic VCC data loading
│   └── vcc_paired_dataloader.py # Paired control-perturbed sets (critical for fine-tuning)
├── scripts/
│   └── compute_combined_hvgs.py  # Combine expression HVGs with perturbed genes
├── train_st_conditional_diffusion.py  # Main training script
└── eval.py                       # Evaluation metrics
```

## Installation

```bash
# Install required packages
pip install torch numpy pandas scanpy h5py gdown tqdm wandb

# Or create conda environment
conda create -n vcc python=3.9
conda activate vcc
pip install -r requirements.txt
```

## Dataset Setup

### 1. Download Required Datasets

You need two datasets:
1. **scBaseCount**: Large-scale single-cell expression data for pretraining
2. **VCC data**: Perturbation data with control/perturbed pairs

```bash
# Download VCC data (required)
pip install gdown
gdown https://drive.google.com/uc?export=download&id=1QUhQ6nS_etBOcI0oiaWGOHHWQMWSvy36
unzip vcc_data.zip -d data/

# Download scBaseCount (optional, for pretraining)
# Follow instructions from the original download.py script
python download.py --cell-count 1000000 --output-dir data/scRNA
```

### 2. Generate HVG Indices

Compute highly variable genes for both datasets:

```bash
# For scBaseCount data (if using pretraining)
python preprocess_hvgs.py --data-dir data/scRNA/processed --n-hvgs 2000

# For VCC data (uses built-in HVG computation)
# HVGs are computed automatically when loading VCC data
```

### 3. Combine HVGs (Strongly Recommended)

**This step is critical for optimal performance.** It combines:
- Top N highly variable genes from expression data
- All perturbed genes from VCC experiments

```bash
python scripts/compute_combined_hvgs.py \
    --vcc-data-path data/vcc_data/adata_Training.h5ad \
    --n-hvgs 2000 \
    --output-dir data/vcc_data/
```

This creates `hvg_info_combined.json` containing the union of expression HVGs and perturbed genes, ensuring the model can learn perturbation effects while maintaining general expression patterns.

## Training

### Quick Start

```bash
# Train ST-style conditional diffusion model
python train_st_conditional_diffusion.py
```

### Training Phases

The training follows a two-phase approach:

1. **Pretraining** (optional): Train on large-scale scRNA data to learn general expression patterns
2. **Fine-tuning**: Train on VCC paired data with control set conditioning

```python
# Key configuration in train_st_conditional_diffusion.py
config = ConditionalModelConfig(
    # Model size
    dim=128,
    n_layer=4,
    n_head=4,
    
    # Conditioning
    control_set_encoder_layers=2,
    control_set_dim_out=128,  # Must match model dim
    
    # Training phases
    pretrain_epochs=50,   # Set to 0 to skip pretraining
    finetune_epochs=100,  # Fine-tuning with paired data
)
```

## Understanding the Dataloaders

### VCCPairedDataloader (Critical for Fine-tuning)

The `vcc_paired_dataloader.py` is the most important component for training conditional models. It returns paired control and perturbed cell sets:

```python
# What the dataloader returns
batch = {
    'control_expr': torch.Tensor,      # (batch_size, set_size, n_genes)
    'perturbed_expr': torch.Tensor,    # (batch_size, set_size, n_genes)
    'target_gene': str,                # Name of perturbed gene
    'target_gene_idx': int,            # Index in gene list
    'log2fc': torch.Tensor,            # Log2 fold change vector
    'perturbation_magnitude': float,   # L2 norm of perturbation
}
```

Key features:
- **Set-based sampling**: Returns sets of cells (e.g., 16 cells) not individual cells
- **Matched controls**: Control cells from same experimental batch when possible
- **Perturbation metadata**: Includes gene target and effect magnitude

### Why Paired Loading Matters

Traditional approaches process cells independently. ST-style training requires:
1. **Simultaneous access** to control and perturbed sets
2. **Batch matching** to reduce technical variation
3. **Set-level statistics** for robust perturbation estimation

The paired dataloader handles all of this automatically:

```python
from dataset import create_vcc_paired_dataloader

dataset, dataloader = create_vcc_paired_dataloader(
    set_size=16,          # Cells per set
    batch_size=4,         # Number of perturbations per batch
    match_by_batch=True,  # Match controls from same batch
    use_hvgs=True,        # Use combined HVGs
)

# Each iteration provides matched control-perturbed pairs
for batch in dataloader:
    # Model uses control sets for cross-attention conditioning
    loss = diffusion.compute_loss(
        model,
        batch['perturbed_expr'],
        control_set=batch['control_expr'],
        target_gene_idx=batch['target_gene_idx'],
        perturb_magnitude=batch['log2fc']
    )
```

## Model Architecture

The ST-style model implements:

1. **Control Set Encoder**: Processes control cells into conditioning vectors
2. **Cross-Attention**: Perturbed cells attend to control set representations
3. **Perturbation Embeddings**: Gene identity and magnitude conditioning
4. **Adaptive Masking**: Higher mask rates for perturbed gene regions

```python
# Simplified forward pass
def forward(self, x, control_set, target_gene_idx, perturb_magnitude):
    # Encode control set
    control_features = self.control_encoder(control_set)
    
    # Add perturbation conditioning
    gene_embed = self.gene_embeddings(target_gene_idx)
    magnitude_embed = self.magnitude_encoder(perturb_magnitude)
    
    # Transform with cross-attention to controls
    h = self.transformer(
        x, 
        context=control_features,
        gene_cond=gene_embed,
        magnitude_cond=magnitude_embed
    )
    return h
```

## Evaluation

The model is evaluated on:
1. **Zero-shot genes**: Predicting perturbations for held-out genes
2. **Expression correlation**: Pearson correlation of predicted vs actual
3. **Differential expression**: Accuracy of top DE gene prediction

```bash
# Evaluation happens automatically during training
# Check wandb logs for metrics
```

## Tips for Best Results

1. **Always use combined HVGs**: This ensures all perturbed genes are included
2. **Start with pretraining**: Even 10-20 epochs helps with general expression patterns
3. **Use mixed training**: Alternate between conditioned and unconditioned batches
4. **Monitor zero-shot performance**: Key metric for generalization
5. **Adjust set size**: Larger sets (16-32) are more stable but need more memory

## Common Issues

### Out of Memory
- Reduce `set_size` (e.g., from 16 to 8)
- Reduce `batch_size`
- Use gradient checkpointing

### Poor Zero-shot Performance
- Ensure combined HVGs are used
- Increase model capacity (`dim`, `n_layer`)
- Train longer with mixed conditioning

### Slow Training
- Enable multiple workers in dataloader
- Use larger batches if memory allows
- Consider reducing `n_hvgs` if using many genes

## Citation

If you use this codebase, please cite the relevant papers for ST, diffusion models, and the VCC dataset. 