# Cross-Dataset HVG Alignment

## The Problem

A critical issue was discovered in the original training pipeline where pretraining and fine-tuning phases were using completely different gene indices. This breaks transfer learning because:

1. **Pretraining (scRNA data)**: Used the first 2000 genes from the downloaded dataset in arbitrary order
2. **Fine-tuning (VCC data)**: Used 2000 HVGs computed only from VCC data

This means gene at index 0 during pretraining could be "ACTB" while gene at index 0 during fine-tuning could be "GAPDH". The model's learned gene embeddings become meaningless when switching phases.

## The Solution

We've implemented cross-dataset HVG selection that ensures consistent gene indices across both phases:

### 1. Compute Cross-Dataset HVGs

Run the new script to compute HVGs across both datasets:

```bash
python scripts/compute_cross_dataset_hvgs.py \
    --scrna-dir data/scRNA_1e5/processed \
    --vcc-path data/vcc_data/adata_Training.h5ad \
    --output-dir data/vcc_data \
    --n-hvgs 2000
```

This script:
- Loads gene statistics from both scRNA and VCC datasets
- Finds the gene intersection
- Includes all perturbed genes from VCC (critical for fine-tuning)
- Computes weighted dispersion across both datasets
- Selects top 2000 HVGs that are informative in both contexts
- Creates mapping files for both datasets

### 2. Updated Data Pipeline

The training script now uses `ScRNADatasetWithHVGs` for pretraining, which:
- Loads the cross-dataset HVG mappings
- Remaps scRNA gene indices to the common HVG space
- Ensures both phases use identical gene ordering

### 3. File Outputs

The script creates several files:

- `cross_dataset_hvg_info.json`: Main file with all mappings
  - `hvg_names`: List of selected HVG gene names
  - `scrna_to_hvg`: Mapping from scRNA indices to HVG indices
  - `vcc_to_hvg`: Mapping from VCC indices to HVG indices
  
- `hvg_info.json`: VCC-compatible format (updated to use cross-dataset HVGs)

## Impact

This fix is **critical** for transfer learning to work properly:

- **Before**: Pretrained gene embeddings were useless during fine-tuning
- **After**: Gene embeddings learned during pretraining directly transfer to fine-tuning

## Verification

You can verify the alignment is working by checking the training logs:

```
Creating scRNA pretrain dataloader with cross-dataset HVGs...
Initialized ScRNA dataset with cross-dataset HVGs:
  Total cells: 100,000
  Original genes: 33,538
  HVG genes: 2000
  Genes mapped to HVGs: 1,847

...

Phase 2: Fine-tuning with Control Sets
Using same HVG gene indices as pretraining - transfer learning enabled!
```

The key indicator is that both phases report using the same number of HVG genes (2000).

## Technical Details

### Gene Selection Strategy

1. **Perturbed genes have priority**: All VCC perturbed genes that exist in both datasets are included
2. **Weighted dispersion**: Genes are ranked by dispersion weighted across both datasets
3. **Minimum expression filter**: Genes must be expressed in at least 0.01% of cells

### Memory Optimization

The `ScRNADatasetWithHVGs` class includes:
- Batch-level caching to avoid repeated gene remapping
- Option to disable caching for very large datasets
- Efficient sparse-to-dense conversion

### Backwards Compatibility

The VCC dataloaders automatically use the updated `hvg_info.json` file, so existing VCC code continues to work without modification. 