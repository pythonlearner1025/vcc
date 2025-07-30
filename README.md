# VCC: Perturbation Prediction with Diffusion Models

Implementation of perturbation prediction models using discrete diffusion transformers, inspired by the Visual Compressive Clustering (VCC) architecture.

## Overview

This repository contains:
- ST-style conditional diffusion transformer for perturbation prediction
- Discrete tokenization of gene expression values
- Control set cross-attention mechanism
- Zero-shot perturbation prediction capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vcc
cd vcc

# Create conda environment
conda create -n vcc python=3.9
conda activate vcc

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Download scRNA-seq Data

Download a subset of the scBaseCount dataset:

```bash
python dataset/download.py --cell-count 1e5 --output-dir data/scRNA_1e5
```

This downloads ~100k cells from human scRNA-seq experiments.

### 2. Prepare VCC Data

Place the VCC dataset files in `data/vcc_data/`:
- `adata_Training.h5ad` - Training data
- `adata_Validation.h5ad` - Validation data  
- `pert_counts_Validation.csv` - Validation perturbation counts

### 3. Compute Cross-Dataset HVGs (CRITICAL)

**Important**: You must compute cross-dataset HVGs to ensure consistent gene indices between pretraining and fine-tuning:

```bash
python scripts/compute_cross_dataset_hvgs.py \
    --scrna-dir data/scRNA_1e5/processed \
    --vcc-path data/vcc_data/adata_Training.h5ad \
    --output-dir data/vcc_data
```

This creates mappings that align gene indices across both datasets. Without this step, transfer learning will not work properly. See [docs/cross_dataset_hvgs.md](docs/cross_dataset_hvgs.md) for details.

## Model Architecture

The ST-style conditional diffusion transformer includes:

## Training

### Basic Training

Train the ST-style conditional diffusion model:

```bash
python train_st_conditional_diffusion.py
```

### Configuration

Key hyperparameters in `train_st_conditional_diffusion.py`:

```python
config = ConditionalModelConfig(
    # Model
    dim=256,
    n_head=8, 
    n_layer=8,
    
    # Diffusion
    n_timesteps=10,
    mask_ratio=0.30,
    
    # Training
    pretrain_epochs=10,
    finetune_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
)
```

### Training Phases

1. **Pretraining**: Train on single cells from scRNA-seq data (using cross-dataset HVGs)
2. **Fine-tuning**: Train with control-perturbed paired sets from VCC data

### Mixed Training

During fine-tuning, the model randomly alternates between:
- Conditioned generation (using control sets)
- Unconditioned generation (standard diffusion)

This improves robustness and generation quality.

## Evaluation

### Zero-shot Perturbation Prediction

The model is evaluated on held-out perturbations not seen during training:

```python
python eval_zero_shot.py --checkpoint checkpoint_st_final.pt
```

### Metrics

- **Gene-wise correlation**: Pearson correlation of predicted vs actual log2FC
- **Top-20 DE gene recovery**: Overlap of top differentially expressed genes
- **Direction accuracy**: Accuracy of up/down regulation prediction

## Model Checkpoints

Checkpoints are saved during training:
- `checkpoint_st_pretrained.pt` - After pretraining phase
- `checkpoint_st_epoch_X_step_Y.pt` - During training
- `checkpoint_st_final.pt` - Final model

## Troubleshooting

### Common Issues

1. **Transfer learning not working**: Make sure you've run `compute_cross_dataset_hvgs.py`
2. **Out of memory**: Reduce batch size or set size
3. **Slow training**: Enable gradient checkpointing or use mixed precision

### Verifying Gene Alignment

Check that gene indices are properly aligned:

```python
# Load cross-dataset HVG info
import json
with open('data/vcc_data/cross_dataset_hvg_info.json', 'r') as f:
    hvg_info = json.load(f)

print(f"Total HVGs: {len(hvg_info['hvg_names'])}")
print(f"Genes mapped from scRNA: {len(hvg_info['scrna_to_hvg'])}")
print(f"Genes mapped from VCC: {len(hvg_info['vcc_to_hvg'])}")
```

## Citation

If you use this code, please cite:

```bibtex
@article{vcc2024,
  title={Visual Compressive Clustering: Perturbation Prediction with Diffusion Models},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details. 