# VCC - Variational Cell Conditioning

A modular, extensible framework for single-cell RNA-seq perturbation prediction using Variational Autoencoders (VAEs).

> **🎉 Refactoring Complete!** The VCC repository has been successfully refactored with a modern, flexible VAE architecture. See [`REFACTORING_COMPLETION.md`](REFACTORING_COMPLETION.md) for details and [`validate_flexible_vae.py`](validate_flexible_vae.py) for validation results.

## Overview

This project implements a **Flexible VAE architecture** designed for two-phase training:

1. **Phase 1**: Self-supervised pretraining on large scRNA-seq datasets (learn cellular representations)
2. **Phase 2**: Perturbation fine-tuning with paired control/perturbation data (learn perturbation effects)

### Key Features

- **Modular Architecture**: Easy to adjust latent space size and swap gene embeddings
- **Flexible Gene Embeddings**: Support for ESM2, Gene2Vec, or learned embeddings  
- **Robust Data Handling**: Phase-aware loading with validation of control/perturbation pairs
- **Clean Separation**: Encoder sees only cellular state, decoder applies perturbations
- **Extensible Design**: Easy to add new embedding types or architectures

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/vcc.git
cd vcc

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements_dev.txt
```

### Basic Usage

1. **Generate synthetic data and test the pipeline**:
```bash
python example_flexible_vae.py --mode synthetic --save_model
```

2. **Phase 1 training (pretraining)**:
```bash
python train_flexible_vae.py \
    --phase 1 \
    --data_path data/pretraining_data.npz \
    --config configs/phase1_config.yaml \
    --num_epochs 100
```

3. **Phase 2 training (fine-tuning)**:
```bash
python train_flexible_vae.py \
    --phase 2 \
    --data_path data/paired_data.npz \
    --config configs/phase2_config.yaml \
    --pretrained_model logs/phase1/checkpoints/best_model.pt \
    --num_epochs 50
```

4. **Perturbation injection**:
```bash
python inference_flexible_vae.py inject \
    --model logs/phase2/checkpoints/best_model.pt \
    --data data/control_cells.npz \
    --perturbation_id 5 \
    --output results/predictions.npz
```

## Architecture Overview

### Flexible VAE Design

- **Encoder**: `gene_expression + experiment_id → latent_representation`
- **Decoder**: `latent_representation + experiment_id + target_gene_embedding → perturbed_expression`

### Gene Embedding System

The modular gene embedding system supports:

- **Learned Embeddings**: Trainable embeddings similar to word embeddings
- **ESM2 Embeddings**: Pretrained protein language model embeddings  
- **Gene2Vec Embeddings**: Biological pathway-aware embeddings
- **Custom Embeddings**: Easy to add new embedding types

## Training Phases

### Phase 1: Self-Supervised Pretraining

Train on large scRNA-seq datasets without perturbation labels to learn general cellular representations:

```python
# No perturbation information provided
outputs = model(expression, experiment_ids)
```

**Benefits**:
- Learn robust cellular representations
- Capture cell type information in latent space
- Handle batch effects through experiment embeddings

### Phase 2: Perturbation Fine-Tuning

Fine-tune on paired control/perturbation data to learn perturbation effects:

```python
# With target gene information for perturbation
outputs = model(control_expression, experiment_ids, target_gene_ids=target_genes)
```

**Benefits**:
- Learn targeted perturbation effects
- Maintain cellular representations from Phase 1
- Enable zero-shot perturbation prediction

## Configuration

Use YAML configuration files for reproducible experiments:

```yaml
# configs/phase1_config.yaml
input_dim: 1808
latent_dim: 512
hidden_dims: [1024, 512, 256]
learning_rate: 0.001
batch_size: 256
kld_weight: 0.5
```

Available configurations:
- `configs/phase1_config.yaml`: Phase 1 pretraining
- `configs/phase2_config.yaml`: Phase 2 fine-tuning  
- `configs/experimental_large.yaml`: Large model experiments

## Data Format

### Phase 1 Data (Pretraining)

```python
# Required fields in .npz file
{
    'expression': np.ndarray,     # [n_cells, n_genes]
    'experiment_ids': np.ndarray, # [n_cells] - batch/experiment identifiers
    'gene_names': List[str]       # Optional: gene identifiers
}
```

### Phase 2 Data (Fine-tuning)

```python
# Required fields in .npz file  
{
    'expression': np.ndarray,      # [n_cells, n_genes]
    'experiment_ids': np.ndarray,  # [n_cells] - batch/experiment identifiers
    'perturbation_ids': np.ndarray,# [n_cells] - perturbation identifiers (0=control)
    'target_gene_ids': np.ndarray, # [n_cells] - target gene for each perturbation
    'gene_names': List[str]        # Optional: gene identifiers
}
```

## Advanced Usage

### Custom Gene Embeddings

```python
# Load pretrained embeddings
gene_embeddings = torch.load('embeddings/esm2_gene_embeddings.pt')
gene_emb = PretrainedGeneEmbedding(gene_embeddings, freeze=True)

# Create model with custom embeddings
model = FlexibleVAE(config, gene_emb)
```

### Latent Space Analysis

```python
# Analyze learned representations
python inference_flexible_vae.py analyze \
    --model checkpoints/best_model.pt \
    --data data/cells.npz \
    --output analysis/latent_space.png \
    --n_clusters 15
```

### Batch Processing

```python
# Process large datasets efficiently
python inference_flexible_vae.py predict \
    --model checkpoints/best_model.pt \
    --data data/large_dataset.npz \
    --batch_size 1000 \
    --output results/predictions/ \
    --evaluate
```

## Model Evaluation

The framework includes comprehensive evaluation metrics:

- **Reconstruction Quality**: R², Pearson correlation, MSE
- **Perturbation Accuracy**: Effect size, target specificity
- **Latent Space Quality**: Clustering, separability
- **Cross-experiment Transfer**: Batch effect removal

## Legacy Code

**Note**: This repository contains legacy scripts that are being phased out:

- `train_vae.py`, `train_vae_clean.py` → Use `train_flexible_vae.py`
- `example_vae.py`, `example_vae_clean.py` → Use `example_flexible_vae.py`  
- `models/VAE.py` → Use `models/flexible_vae.py`

The new flexible architecture is more modular, better documented, and easier to extend.

## TODO

- [ ] More efficient HVG calculation over large datasets
- [ ] Technical batch encoding improvements
- [x] Modular VAE architecture with flexible embeddings
- [x] Phase-aware training pipeline
- [x] Comprehensive documentation and examples
- [ ] Integration with Scanpy/AnnData ecosystem
- [ ] Distributed training support
- [ ] Model zoo with pretrained embeddings

## Experiment Queue

- [ ] Different masking strategies for pretraining
- [ ] ESM2 gene embeddings integration
- [ ] Auxiliary losses for count distribution modeling
- [ ] Transformer-based architectures
- [ ] Multi-modal conditioning (cell type + perturbation)

## Installation

```bash
# Setup script (if available)
./setup.sh

# Manual installation
pip install torch torchvision matplotlib seaborn scikit-learn pandas numpy scipy pyyaml tqdm tensorboard
```

## Data Download

- Download pretraining data with `dataset/download.py`
- Download VCC data from [Google Drive link] 

## HVG Build

Building HVGs requires high RAM (150GB+) for large datasets. Consider more efficient algorithms:

### Method: SCRAN
```bash
python scripts/scran_hvg.py
```

### Method: Seurat V3  
```bash
python scripts/seurat_hvg.py
```

## Citation

If you use this code, please cite:

```bibtex
@software{vcc_flexible_vae,
  title={VCC: Flexible VAE for Single-Cell Perturbation Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/vcc}
}
```

## License

[Specify your license here]
perturbed = model.inject_perturbation(control_expr, exp_ids, pert_ids)

# Predict control from perturbed cells  
control = model.predict_control_from_perturbed(perturbed_expr, exp_ids)
```

## Files

- `models/VAE.py`: Core VAE implementation with perturbation injection
- `example_perturbation_injection.py`: Comprehensive demonstration
- `train_vae_clean.py`: Training script for paired data
- `dataset/vae_paired_dataloader.py`: Data loader ensuring control/perturbation pairs
- `VAE_PERTURBATION_INJECTION.md`: Detailed technical documentation
- `train_vae.py`: Training pipeline
- `inference_vae.py`: Analysis and inference tools
- `example_vae.py`: Example usage with synthetic data
- `VAE_README.md`: Detailed documentation
- `requirements_vae.txt`: Python dependencies

See `VAE_README.md` for comprehensive documentation and usage examples.