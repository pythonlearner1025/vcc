# Single-Cell RNA-seq Data Downloader for ML Training

A production-ready script to download and preprocess a random subset of human single-cell RNA-seq data from the Arc Institute's scBaseCount dataset, optimized for discrete diffusion transformer training with PyTorch.

## Features

- **Random Sampling**: Intelligently samples experiments to reach target cell count
- **Memory Efficient**: Processes data in batches to handle large datasets
- **ML-Optimized Format**: Saves data in HDF5 format with int16 precision for fast loading
- **PyTorch Ready**: Includes a custom Dataset class for seamless integration
- **Metadata Preservation**: Keeps essential cell and experiment metadata
- **Progress Tracking**: Real-time download progress with detailed logging
- **Resumable**: Can reuse previously downloaded files
- **HVG Selection**: Automatic selection of top 2000 highly variable genes to reduce memory usage

## Installation

```bash
# Install required packages
pip install numpy pandas scanpy gcsfs pyarrow h5py tqdm torch matplotlib seaborn

# Or use the provided conda environment
conda env create -f conda_env.yml
conda activate scrna-ml
```

## Usage

### 1. Download Data

```bash
# Download 1 million cells (default)
python download.py

# Download specific number of cells
python download.py --cell-count 5000000 --output-dir data/scRNA

# Resume interrupted download
python download.py --resume

# Download with specific parameters
python download.py \
    --cell-count 2000000 \
    --output-dir data/scRNA \
    --batch-size 10000 \
    --max-genes 5000 \
    --n-threads 4
```

### 2. Preprocess Highly Variable Genes (HVGs)

After downloading, compute the top 2000 highly variable genes to reduce memory usage:

```bash
# Compute HVGs (creates hvg_info.json and visualizations)
python preprocess_hvgs.py --data-dir data/scRNA/processed

# Custom parameters
python preprocess_hvgs.py \
    --data-dir data/scRNA/processed \
    --n-hvgs 3000 \
    --min-cells-pct 0.05 \
    --output-dir results/hvg

# Force recomputation
python preprocess_hvgs.py --force
```

This will:
- Compute gene statistics across all downloaded cells
- Select top N genes by dispersion (variance/mean ratio)
- Create visualizations of HVG distributions
- Save mappings for O(1) access during training

### 3. Use in PyTorch Training

```python
# Standard usage with all genes
from dataset import ScRNADataset, create_dataloader

dataset = ScRNADataset("data/scRNA/processed")
dataloader = create_dataloader("data/scRNA/processed", batch_size=32)

# Memory-efficient usage with HVGs only
from example_usage import HVGDataset, create_hvg_dataloader

# Automatically loads precomputed HVGs
dataset = HVGDataset("data/scRNA/processed", n_hvgs=2000)
dataloader = create_hvg_dataloader("data/scRNA/processed", batch_size=32)

# Each batch returns (expression, metadata)
for batch_idx, (expression, metadata) in enumerate(dataloader):
    # expression: [batch_size, 2000] - only HVG features
    # metadata: dict with cell_id, experiment_id, etc.
    pass
```

## Output Structure

```
data/scRNA/
├── config.json              # Dataset configuration
├── hvg_info.json           # HVG selection results (after preprocessing)
├── hvg_analysis.png        # HVG statistics visualization
├── processed/
│   ├── batch_0000.h5       # Expression data in batches
│   ├── batch_0001.h5
│   └── ...
└── metadata/
    ├── experiments.json     # Experiment metadata
    └── cells.json          # Cell metadata
```

## HVG Selection Strategy

The HVG selection follows standard single-cell practices:

1. **Compute dispersion** (variance/mean ratio) for each gene across all cells
2. **Filter genes** expressed in at least 0.1% of cells
3. **Select top 2000 genes** by dispersion
4. **Smart fallback**: For cells with few HVGs expressed, fill remaining slots with top expressed non-HVG genes

This reduces memory from ~30,000 to 2,000 features while preserving the most informative genes.

## Data Format

### HDF5 Batch Files
Each batch file contains:
- `expression`: int16 array of shape [n_cells, n_genes]
- `cell_ids`: Cell barcodes
- `experiment_ids`: Source experiment IDs

### HVG Info File
The `hvg_info.json` contains:
- `hvg_indices`: List of selected gene indices
- `hvg_names`: Gene names/symbols
- `gene_to_hvg_idx`: Mapping for O(1) lookup
- `statistics`: Dispersion and expression stats

## Memory Estimates

- **Full genes (30K)**: ~60KB per cell
- **HVG only (2K)**: ~4KB per cell (93% reduction)
- **1M cells**: ~4GB with HVGs vs ~60GB full

## Performance Tips

1. **Preprocessing**: Always run `preprocess_hvgs.py` before training
2. **Batch Size**: Use larger batches (128-512) for better GPU utilization
3. **Data Loading**: Use multiple workers in DataLoader
4. **Caching**: HVG info is cached for fast repeated access

## Examples

See `example_usage.py` for a complete training example with:
- HVG preprocessing
- Custom tokenization strategies
- Simple transformer model
- Training loop

## Citation

If you use this data, please cite the Arc Institute's scBaseCount dataset. 