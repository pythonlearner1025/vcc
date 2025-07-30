# TODO
- [ ] More efficient HVG calculation over large datasets
- [ ] Technical batch encoding

# Experiment Queue
- [ ] Different mixes of masking for pretraining
- [ ] Different mixes of masking for perturbation (since zero-shot prediction begins from full masking, some full masking would help)
- [ ] Use ESM embeddings for gene 
- [ ] Auxillary loss for heteroscedastic HVG counts (Maximum Mean Discrepancy)
- [ ] Different architecture

# Installation

setup.sh 

# Data Download

- Download pretraining data scBaseCount with dataset/download.py
- Download VCC data with gdown utility from google drive link [ADD LINK]

# HVG Build

Building the HVG takes very long and requires high RAM (150GB+) for a dataset with ~300,000 cells. Algorithms for more efficient HVG calculation or moving away from it entirely should be considered.  

### Method SCRAN

Computes HVGs over only the set of intersecting genes in scRNA and VCC 

- scripts/scran_hvg.py

### Method Seurat V3

Computes HVGs over only the set of intersecting genes in scRNA and VCC 

- scripts/seurat_hvg.py

# VAE Implementation

A Conditional Variational Autoencoder (VAE) has been implemented for single-cell RNA-seq analysis with **perturbation injection capabilities**. The VAE can predict perturbation effects, work with unlabeled data, and transfer learned perturbations across experiments.

## Key Features

- **Perturbation Injection**: Predict how perturbations would affect control cells
- **Control Prediction**: Remove perturbation effects to predict control states  
- **General Data Training**: Pretrain on scRNA-seq data without perturbation labels
- **Cross-experiment Transfer**: Apply learned perturbations to new batches
- **Latent Space Analysis**: Cell types emerge naturally without explicit conditioning

## Architecture Overview

- **Encoder**: Takes gene expression + experiment ID (no perturbation info)
- **Decoder**: Takes latent + experiment ID + optional perturbation ID
- **Training**: Supports both general data pretraining and perturbation fine-tuning

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements_vae.txt
```

2. Basic perturbation injection example:
```bash
python example_perturbation_injection.py
```

3. Train VAE with paired data:
```bash
python train_vae_clean.py \
    --data_path data/paired_data.npz \
    --latent_dim 128 \
    --num_epochs 100 \
    --save_dir checkpoints/
```

4. Test VAE functionality:
```bash
python models/VAE.py
```

## Training Paradigms

### Phase 1: Pretraining (Optional)
Train on general scRNA-seq data without perturbation labels:
```python
# No perturbation information provided
outputs = model(expression, experiment_ids, perturbation_ids=None)
```

### Phase 2: Fine-tuning 
Train on paired control/perturbation data:
```python
# With perturbation information
outputs = model(expression, experiment_ids, perturbation_ids)
```

### Phase 3: Inference
Inject perturbations or predict controls:
```python
# Inject perturbation into control cells
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