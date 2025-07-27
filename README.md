# Single-Cell RNA-seq Data Downloader for ML Training

A production-ready pipeline to download and preprocess human single-cell RNA-seq data from the Arc Institute's scBaseCount dataset, optimized for discrete diffusion transformer training with PyTorch.

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

### 3. Train Your Model

```python
# Standard usage with all genes
from dataset import ScRNADataset, create_dataloader

dataset = ScRNADataset("data/scRNA/processed")
dataloader = create_dataloader("data/scRNA/processed", batch_size=32)

# Memory-efficient usage with HVGs only
from train import HVGDataset, create_hvg_dataloader

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

## Data Format

### How download.py Works

The download script performs the following steps:

1. **Connects to GCS**: Uses `gcsfs` to access the Arc Institute's public Google Cloud Storage bucket
2. **Loads metadata**: Reads experiment metadata to identify human samples
3. **Random sampling**: Randomly selects experiments until reaching the target cell count
4. **Batch processing**: Downloads and processes experiments in batches to manage memory
5. **Data conversion**: Converts sparse matrices to dense int16 format for efficient ML training
6. **File organization**: Saves processed data in numbered batch files

### Batch Files (batch_XXXX.h5)

Each batch file is an HDF5 file containing cells from multiple experiments, structured for efficient loading:

```
batch_0000.h5
├── X                    # Expression matrix [n_cells, n_genes], dtype=int16
├── genes               # Gene identifiers (ENSG IDs), shape [n_genes]
├── cells               # Cell barcodes, shape [n_cells]  
└── obs/                # Cell-level metadata group
    ├── gene_count      # Number of genes detected per cell
    ├── umi_count       # Total UMI counts per cell
    └── SRX_accession   # Source experiment ID for each cell
```

**Key fields:**
- `X`: The gene expression count matrix (not 'expression'). Each row is a cell, each column is a gene
- `genes`: Ensembl gene IDs in the same order across all batches
- `cells`: Unique cell barcodes/identifiers
- `obs/gene_count`: Quality metric - cells with very low gene counts may be low quality
- `obs/umi_count`: Total molecular counts per cell
- `obs/SRX_accession`: Links each cell back to its source experiment

### config.json

The configuration file contains global dataset information:

```json
{
  "total_cells_downloaded": 1000000,
  "n_batches": 100,
  "batch_size": 10000,
  "gene_list": ["ENSG00000243485", "ENSG00000237613", ...],  # All 36,601 genes
  "max_expression_value": 32500,  # Maximum count observed
  "sparsity": 0.94,  # Fraction of zeros in the data
  "download_timestamp": "2024-01-15T10:30:00",
  "source_samples": [
    {
      "srx_accession": "SRX12345",
      "n_cells": 5000,
      "tissue": "blood",
      "organism": "Homo sapiens"
    },
    ...
  ]
}
```

### HVG Info File (hvg_info.json)

The highly variable gene selection file contains:

```json
{
  "hvg_indices": [4521, 4456, 4523, ...],  # Indices into gene_list
  "hvg_names": ["ENSG00000211675", "ENSG00000211592", ...],  # Gene IDs
  "gene_to_hvg_idx": {
    "4521": 0,  # Maps gene index to position in HVG list
    "4456": 1,
    ...
  },
  "statistics": {
    "total_cells_analyzed": 1000000,
    "mean_dispersion": 156.7,  # Average variance/mean ratio
    "min_dispersion": 9.73,
    "max_dispersion": 24818.79,
    "top_10_hvgs": [...],  # Most variable genes
    "top_10_dispersions": [...],  # Their dispersion values
    "top_10_pct_cells": [...]  # Percentage of cells expressing
  }
}
```

### HVG Computation Details

HVGs are selected using the dispersion metric (variance/mean ratio):

1. **Gene statistics**: For each gene, compute mean expression and variance across all cells
2. **Dispersion**: Calculate variance/mean ratio - high dispersion indicates biological variability beyond technical noise
3. **Filtering**: Exclude genes expressed in < 0.1% of cells (likely noise)
4. **Selection**: Take top 2000 genes by dispersion

**Why dispersion?** 
- Genes with high mean expression naturally have high variance
- Dispersion normalizes for expression level
- High dispersion genes are often cell-type markers or involved in biological processes
- Standard practice in single-cell analysis (used by Seurat, Scanpy, etc.)

**Biological interpretation:**
- Top HVGs often include immunoglobulin genes, cell surface markers, and transcription factors
- These genes distinguish cell types and states
- Focusing on HVGs reduces noise while preserving biological signal

## Memory Estimates

- **Full genes (36,601)**: ~73KB per cell
- **HVG only (2,000)**: ~4KB per cell (94.5% reduction)
- **1M cells**: ~4GB with HVGs vs ~73GB full

## Performance Tips

1. **Preprocessing**: Always run `preprocess_hvgs.py` before training
2. **Batch Size**: Use larger batches (128-512) for better GPU utilization
3. **Data Loading**: Use multiple workers in DataLoader
4. **Caching**: HVG info is cached for fast repeated access

## Examples

See `train.py` for a complete training example with:
- HVG preprocessing integration
- Custom tokenization strategies (direct, binned, log-scale)
- Simple transformer architecture
- Training loop with loss tracking

## Citation

If you use this data, please cite the Arc Institute's scBaseCount dataset. 