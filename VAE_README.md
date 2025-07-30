# VAE for Single-Cell RNA-seq Data

This directory contains a Conditional Variational Autoencoder (VAE) implementation for single-cell RNA-seq data analysis. The VAE can predict perturbation effects using the batch file format produced by `download.py`.

## Overview

The Conditional VAE consists of:

1. **ConditionalVAE** (`models/VAE.py`): The main model that can be conditioned on:
   - Perturbation type (control, treatment, knockout, etc.)
   - Experiment source (SRX accession) to account for technical confounding effects

2. **Training Script** (`train_vae.py`): Complete training pipeline with data loading, model training, and evaluation

3. **Inference Script** (`inference_vae.py`): Analysis tools for trained models including perturbation effect analysis

**Important Note**: Cell types are NOT explicitly modeled as conditioning variables. Instead, the model learns cell type representations implicitly from transcriptomic profiles in the latent space. This allows the model to discover cell types naturally without requiring pre-defined labels.

## Key Features

- **Conditional Generation**: Generate synthetic cells with specific perturbation conditions and experiment contexts
- **Perturbation Effect Analysis**: Compare perturbed vs control conditions in a controlled manner
- **Latent Space Analysis**: Analyze learned representations where cell types emerge naturally
- **Experiment Conditioning**: Account for technical batch effects through SRX accession embedding
- **Scalable Data Loading**: Efficient handling of large datasets using HDF5 batch files

## Installation

Install dependencies:
```bash
pip install -r requirements_vae.txt
```

## Data Format

The VAE works with the batch file format produced by `download.py`:

```
data/scRNA/processed/
├── batch_0000.h5
├── batch_0001.h5
├── ...
└── config.json
```

Each batch file contains:
- `X`: Gene expression matrix [n_cells, n_genes]
- `genes`: Gene names
- `cells`: Cell barcodes
- `obs/gene_count`: Gene counts per cell
- `obs/umi_count`: UMI counts per cell
- `obs/SRX_accession`: Experiment accession IDs

## Usage

### 1. Training a VAE Model

Basic training:
```bash
python train_vae.py \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_training \
    --n-epochs 100 \
    --batch-size 256
```

Training with HVG filtering:
```bash
python train_vae.py \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_training \
    --hvg-file data/hvg_genes.txt \
    --input-dim 2000 \
    --n-epochs 100
```

Advanced configuration:
```bash
python train_vae.py \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_training \
    --hvg-file data/hvg_genes.txt \
    --input-dim 2000 \
    --latent-dim 128 \
    --hidden-dims 512 256 \
    --learning-rate 1e-3 \
    --n-epochs 100 \
    --batch-size 256 \
    --val-split 0.1 \
    --device cuda
```

### 2. Model Inference and Analysis

Load a trained model and run analysis:

```bash
python inference_vae.py \
    --model-path output/vae_training/best_model.pt \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_analysis \
    --predict-cell-types \
    --analyze-perturbations \
    --analyze-latent
```

Generate synthetic cells:
```bash
python inference_vae.py \
    --model-path output/vae_training/best_model.pt \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_analysis \
    --generate-cells \
    --n-generate 1000 \
    --generate-cell-type T_cell \
    --generate-perturbation treatment_A
```

### 3. Programmatic Usage

```python
import torch
from models.VAE import ConditionalVAE, VAEConfig
from inference_vae import load_trained_model

# Load trained model
model, config, vocabularies = load_trained_model('output/vae_training/best_model.pt')

# Prepare input data
x = torch.randn(32, 2000)  # Gene expression [batch_size, n_genes]
cell_type_ids = torch.randint(0, 10, (32,))  # Cell type IDs
perturbation_ids = torch.randint(0, 5, (32,))  # Perturbation IDs
experiment_ids = torch.randint(0, 100, (32,))  # Experiment IDs

# Forward pass
with torch.no_grad():
    outputs = model(x, cell_type_ids, perturbation_ids, experiment_ids)
    reconstructed = outputs['reconstructed']
    latent_representation = outputs['z']

# Generate new samples
generated = model.generate(
    cell_type_ids=torch.tensor([0]),  # T_cell
    perturbation_ids=torch.tensor([1]),  # treatment_A
    experiment_ids=torch.tensor([0]),  # experiment_1
    n_samples=10
)
```

## Model Architecture

The ConditionalVAE consists of:

### Encoder
- Input: Gene expression + conditioning embeddings
- Architecture: Fully connected layers with batch normalization and dropout
- Output: Latent mean (μ) and log variance (log σ²)

### Decoder
- Input: Sampled latent representation + conditioning embeddings
- Architecture: Fully connected layers (reverse of encoder)
- Output: Reconstructed gene expression

### Conditioning
- **Perturbation Embedding**: Maps perturbation IDs to dense vectors
- **Experiment Embedding**: Maps SRX accession IDs to dense vectors (for batch effect correction)

### Loss Function
- **Reconstruction Loss**: MSE between input and reconstructed expression
- **KL Divergence**: Regularizes latent space to follow standard normal distribution
- **Total Loss**: Weighted combination of reconstruction and KL losses

## Configuration

The `VAEConfig` class controls model hyperparameters:

```python
@dataclass
class VAEConfig:
    # Architecture
    input_dim: int = 1808  # Number of genes in VCC dataset
    latent_dim: int = 128  # Latent space dimension
    hidden_dims: List[int] = [512, 256]  # Encoder/decoder hidden layers
    
    # Conditioning vocabularies (automatically determined from data)
    n_perturbations: int = 100  # Number of unique perturbations
    n_experiments: int = 500    # Number of unique SRX accessions
    
    # Embedding dimensions
    perturbation_embed_dim: int = 32
    experiment_embed_dim: int = 16
    
    # Loss weighting
    kld_weight: float = 1.0
    reconstruction_weight: float = 1.0
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
```

## Output Files

### Training Outputs
- `best_model.pt`: Best model checkpoint
- `checkpoint_epoch_X.pt`: Periodic checkpoints
- `config.json`: Model configuration
- `vocabularies.json`: Metadata vocabularies
- `tensorboard/`: Training logs for visualization

### Inference Outputs
- `predicted_cell_types.npy`: Cell type predictions
- `true_cell_types.npy`: True cell type labels
- `perturbation_effects.json`: Perturbation effect analysis
- `generated_cells.npy`: Synthetic cell data
- `latent_representations.npy`: Latent space embeddings
- `latent_space_pca.png`: PCA visualization
- `latent_space_tsne.png`: t-SNE visualization

## Applications

### 1. Perturbation Effect Prediction
Compare gene expression under different perturbation conditions:

```python
control_expression = trainer.predict_perturbation_effect(
    x=perturbed_expression,
    perturbation_ids=current_perturbation_ids,
    experiment_ids=experiment_ids,
    control_perturbation_id=0
)

# Compute fold change
fold_change = perturbed_expression / (control_expression + 1e-8)
```

### 2. Data Augmentation
Generate synthetic cells to augment training datasets:

```python
synthetic_cells = model.generate(
    perturbation_ids=desired_perturbations,
    experiment_ids=experiment_contexts,
    n_samples=1000
)
```

### 3. Latent Space Analysis
Explore cell relationships in the learned latent space where cell types emerge naturally:

```python
mu, logvar = model.encode(x, perturbation_ids, experiment_ids)
# Analyze mu using PCA, t-SNE, clustering, etc.
# Cell types will cluster naturally without explicit supervision
```

### 4. Batch Effect Correction
Use experiment conditioning to account for technical confounding:

```python
# Model learns to separate biological signal from technical batch effects
# through experiment embeddings
```

## Best Practices

1. **HVG Selection**: Use highly variable genes (≤2000) for better performance
2. **Data Normalization**: Log-transform and scale gene expression data
3. **Batch Effects**: Include experiment IDs as conditioning to handle batch effects
4. **Hyperparameter Tuning**: Adjust latent dimension and loss weights based on data
5. **Validation**: Use held-out data to evaluate cell type prediction accuracy
6. **Memory Management**: Use `max_cells_per_batch` for large datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use `max_cells_per_batch`
2. **Poor Reconstruction**: Increase model capacity or reduce regularization
3. **Collapsed Latent Space**: Reduce KL weight or increase latent dimension
4. **Slow Training**: Use fewer genes or reduce dataset size for prototyping

### Performance Optimization

- Use GPU acceleration with CUDA
- Enable mixed precision training for faster training
- Use multiple data loader workers
- Pin memory for faster data transfer

## Integration with Existing Pipeline

The VAE integrates seamlessly with the existing pipeline:

1. **Data Download**: Use `download.py` to create batch files
2. **HVG Selection**: Use `preprocess_hvgs.py` or `scripts/seurat_hvg.py`
3. **VAE Training**: Train conditional VAE with `train_vae.py`
4. **Analysis**: Run inference and analysis with `inference_vae.py`

This provides a complete end-to-end pipeline from raw data download to trained generative models for single-cell analysis.
