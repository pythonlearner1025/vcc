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

A Conditional Variational Autoencoder (VAE) has been implemented for single-cell RNA-seq analysis. The VAE can predict cell types and perturbation effects using the batch file format from download.py.

## Key Features

- **Conditional Generation**: Generate cells with specific cell types and perturbation conditions
- **Cell Type Prediction**: Classify cells based on gene expression patterns  
- **Perturbation Analysis**: Compare treated vs control conditions
- **Latent Space Analysis**: Explore learned representations

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements_vae.txt
```

2. Train a VAE model:
```bash
python train_vae.py \
    --data-dir data/scRNA/processed \
    --output-dir output/vae_training \
    --hvg-file data/hvg_genes.txt \
    --n-epochs 100
```

3. Run inference and analysis:
```bash
python inference_vae.py \
    --model-path output/vae_training/best_model.pt \
    --data-dir data/scRNA/processed \
    --predict-cell-types \
    --analyze-perturbations \
    --analyze-latent
```

4. Test with synthetic data:
```bash
python example_vae.py
```

## Files

- `models/VAE.py`: Core VAE implementation
- `train_vae.py`: Training pipeline
- `inference_vae.py`: Analysis and inference tools
- `example_vae.py`: Example usage with synthetic data
- `VAE_README.md`: Detailed documentation
- `requirements_vae.txt`: Python dependencies

See `VAE_README.md` for comprehensive documentation and usage examples.