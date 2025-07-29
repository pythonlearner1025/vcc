# Single-Cell RNA-seq Data Downloader for ML Training

A production-ready pipeline to download and preprocess human single-cell RNA-seq data from the Arc Institute's scBaseCount dataset, optimized for discrete diffusion transformer training with PyTorch.


**Current Implementation Issues:**
1. **No normalization**: Data is stored as raw UMI counts without sequencing depth normalization
2. **No batch effect correction**: HVGs might be biased toward genes with high technical variation between experiments
3. **No log transformation**: Uses raw counts rather than log-transformed values (common in scRNA-seq analysis)

**Critical Limitation - Sequencing Depth:**
The biggest issue is that **no normalization is applied to account for sequencing depth differences**:
- Some cells may have 1,000 total UMIs, others 15,000+ UMIs
- Raw count comparison across experiments is invalid
- HVG selection may be biased toward high-depth experiments
- Training models on raw counts can lead to batch effects

**Recommended Improvements:**
- Add sequencing depth normalization (CPM, TPM, or DESeq2-style)
- Consider log(counts + 1) transformation before computing dispersion
- Add batch effect assessment in HVG visualization
- Implement alternative methods (e.g., highly variable features from Seurat v4)b seaborn

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
# Download 100,000 cells (default)
python dataset/download.py

# Download specific number of cells
python dataset/download.py --cell-count 1000000 --output-dir data/scRNA

# Download with custom batch size (number of samples processed together)
python dataset/download.py \
    --cell-count 2000000 \
    --output-dir data/scRNA \
    --batch-size 20
```

**Available Parameters:**
- `--cell-count`: Target number of cells to download (default: 100,000)
- `--output-dir`: Output directory (default: data/scRNA)
- `--batch-size`: Number of samples to process in each batch (default: 10)

**Note:** The script downloads complete experiments until the target cell count is reached, so the actual cell count may exceed the target.

### 2. Preprocess Highly Variable Genes (HVGs)

After downloading, compute the top 2000 highly variable genes to reduce memory usage and focus on biologically meaningful features:

```bash
# Compute HVGs from downloaded batch files (creates hvg_info.json and visualizations)
python preprocess_hvgs.py --data-dir data/scRNA/processed

# Custom parameters
python preprocess_hvgs.py \
    --data-dir data/scRNA/processed \
    --n-hvgs 3000 \
    --min-cells-pct 0.05 \
    --output-dir results/hvg

# Force recomputation (if files already exist)
python preprocess_hvgs.py --force

# Process VCC data specifically
python preprocess_hvgs.py --process-vcc --vcc-data-path data/vcc_data/adata_Training.h5ad
```

**Available Parameters:**
- `--data-dir`: Directory containing batch_*.h5 files (default: data/scRNA/processed)
- `--n-hvgs`: Number of HVGs to select (default: 2000)  
- `--min-cells-pct`: Minimum % of cells expressing a gene to be considered (default: 0.1%)
- `--output-dir`: Output directory for results (default: data/scRNA)
- `--force`: Force recomputation even if hvg_info.json exists
- `--process-vcc`: Process VCC-specific data format
- `--vcc-data-path`: Path to VCC .h5ad file

### How HVG Preprocessing Works

The script performs **cross-dataset HVG selection**, computing statistics across ALL downloaded experiments:

#### Step 1: Cross-Dataset Statistics Computation
- Iterates through all `batch_*.h5` files in the processed directory
- Uses online/incremental computation to avoid loading entire dataset into memory
- Computes for each gene across the full dataset:
  - **Mean expression**: Average count across all cells
  - **Variance**: Expression variance across all cells  
  - **Cell count**: Number of cells expressing the gene (count > 0)

#### Step 2: Dispersion-Based Selection
- Calculates **dispersion** = variance/mean for each gene
- Filters out rarely expressed genes (< 0.1% of cells by default)
- Selects top N genes by dispersion (default: 2000)

#### Step 3: Results and Visualization
- Saves HVG indices and mappings to `hvg_info.json`
- Creates comprehensive visualizations in `hvg_analysis.png`:
  - Dispersion vs expression scatter plot (with HVGs highlighted)
  - HVG expression distribution
  - Cell coverage histogram  
  - Top 20 HVGs ranked by dispersion

**Important Notes:**
- HVG selection is performed **across the entire downloaded dataset**, not within individual experiments
- This ensures HVGs capture the most variable genes across different cell types, tissues, and conditions
- Memory usage scales with number of genes (~36k), not number of cells
- The script handles datasets with millions of cells efficiently
- **Data is stored as raw UMI counts** - normalization should be applied during training

### 3. Train Your Model

```python
# Standard usage with all genes (memory-intensive)
from dataset import ScRNADataset, create_dataloader

dataset = ScRNADataset("data/scRNA/processed")
dataloader = create_dataloader("data/scRNA/processed", batch_size=32)

# Memory-efficient usage with HVGs only (RECOMMENDED)
from train import HVGDataset, create_hvg_dataloader

# Automatically loads precomputed HVGs from hvg_info.json
dataset = HVGDataset("data/scRNA/processed", n_hvgs=2000)
dataloader = create_hvg_dataloader("data/scRNA/processed", batch_size=64)

# Each batch returns (expression, metadata)
for batch_idx, (expression, metadata) in enumerate(dataloader):
    # expression: [batch_size, 2000] - only HVG features (vs 36,601 full genes)
    # metadata: dict with cell_id, experiment_id, etc.
    # expression values are RAW UMI COUNTS - normalization needed!
    
    # IMPORTANT: Apply normalization before training
    # Option 1: Log normalization
    expression_log = torch.log1p(expression.float())
    
    # Option 2: CPM (Counts Per Million) normalization
    total_counts = expression.sum(dim=1, keepdim=True)
    expression_cpm = (expression.float() / total_counts) * 1e6
    expression_log_cpm = torch.log1p(expression_cpm)
    
    # Use normalized data for training
    pass

# Load HVG info for custom processing
import json
with open("data/scRNA/hvg_info.json", 'r') as f:
    hvg_info = json.load(f)
    hvg_indices = hvg_info['hvg_indices']  # Indices into full gene set
    hvg_names = hvg_info['hvg_names']      # Ensembl gene IDs
```

## Output Structure

```
data/scRNA/
├── config.json              # Dataset configuration and metadata
├── hvg_info.json           # HVG selection results (after preprocessing)
├── hvg_analysis.png        # HVG statistics visualization
├── raw/                    # Downloaded .h5ad files (optional, can be deleted)
│   ├── SRX12345.h5ad      # Individual experiment files
│   ├── SRX12346.h5ad
│   └── ...
└── processed/
    ├── batch_0000.h5      # Expression data in batches
    ├── batch_0001.h5
    └── ...
```

## Data Format

### How download.py Works

The download script performs a sophisticated multi-step pipeline to efficiently download and process large-scale single-cell RNA-seq data:

#### Step 1: Cloud Storage Connection
- Connects to the Arc Institute's public Google Cloud Storage bucket using `gcsfs`
- Accesses the scBaseCount dataset at path: `gs://arc-scbasecount/2025-02-25/`
- Uses the "GeneFull_Ex50pAS" feature type (full gene set with 50% antisense filtering)

#### Step 2: Metadata Loading
- Downloads sample metadata from: `metadata/GeneFull_Ex50pAS/Homo_sapiens/sample_metadata.parquet`
- Metadata includes for each experiment:
  - `srx_accession`: Unique experiment identifier (e.g., SRX12345)
  - `file_path`: GCS path to the .h5ad file
  - `obs_count`: Number of cells in the experiment
  - `tissue`, `tech_10x`, `cell_prep`, `disease`, `perturbation`: Biological metadata

#### Step 3: Random Sample Selection
- Randomly shuffles all available experiments (using random seed 42 for reproducibility)
- Greedily selects experiments until the target cell count is reached
- **Important**: Downloads complete experiments, so final cell count often exceeds target
- Example: If targeting 1M cells and the last experiment has 50k cells, you'll get 1.05M total

#### Step 4: Batch Processing Pipeline
The script processes experiments in batches to manage memory efficiently:

**4a. Download Phase:**
- Downloads individual .h5ad files to `raw/` directory
- Skips download if file already exists (natural resume functionality)
- Each .h5ad file contains one experiment with sparse expression matrix

**4b. Gene Union Computation:**
- Tracks all unique genes across downloaded experiments
- Builds a master gene list ensuring consistent indexing across batches
- Typical result: ~36,601 unique Ensembl gene IDs

**4c. Batch Concatenation:**
- Groups experiments into batches (default: 10 experiments per batch)
- Concatenates expression matrices using scanpy's `sc.concat()` with outer join
- Fills missing genes with zeros to maintain consistent gene ordering
- Converts sparse matrices to dense int16 format for ML training

**4d. HDF5 Export:**
- Saves each batch as `processed/batch_XXXX.h5` with optimized structure:
  - Expression matrix compressed with gzip level 4
  - Consistent gene ordering across all batches
  - Cell metadata preserved for traceability

#### Step 5: Configuration and Cleanup
- Saves comprehensive dataset configuration to `config.json`
- Calculates dataset-wide statistics (sparsity, max values, etc.)
- Optionally deletes raw .h5ad files to save disk space
- Logs detailed summary including timing and file sizes

#### Memory Management Features
- **Batch processing**: Prevents loading entire dataset into memory
- **Incremental gene union**: Builds gene list progressively
- **Dense conversion**: Optimizes for PyTorch training (sparse tensors are slow)
- **Int16 storage**: Reduces memory by 50% vs int32 (max gene count ~32k)

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
- `X`: The gene expression count matrix (raw UMI counts, not normalized). Each row is a cell, each column is a gene
- `genes`: Ensembl gene IDs in the same order across all batches
- `cells`: Unique cell barcodes/identifiers
- `obs/gene_count`: Quality metric - cells with very low gene counts may be low quality
- `obs/umi_count`: Total molecular counts per cell (important for normalization)
- `obs/SRX_accession`: Links each cell back to its source experiment

### config.json

The configuration file contains complete dataset metadata and statistics:

```json
{
  "total_cells_requested": 1000000,     # Target cell count specified
  "total_cells_downloaded": 1045231,    # Actual cells downloaded
  "total_samples": 87,                  # Number of experiments included
  "download_timestamp": "2024-01-15T10:30:00",
  "gene_list": ["ENSG00000243485", "ENSG00000237613", ...],  # All 36,601 genes
  "max_expression_value": 32500,        # Highest count observed across dataset
  "sparsity": 0.94,                    # Fraction of zero values
  "samples": [                         # Complete experiment metadata
    {
      "srx_accession": "SRX12345",
      "file_path": "gs://arc-scbasecount/.../SRX12345.h5ad",
      "obs_count": 5000,
      "tissue": "blood",
      "tech_10x": "10x 3' v3",
      "cell_prep": "droplet",
      "disease": "healthy",
      "perturbation": "none"
    },
    ...
  ]
}
```

### HVG Info File (hvg_info.json)

The highly variable gene selection file contains comprehensive information:

```json
{
  "hvg_indices": [4521, 4456, 4523, ...],  # Indices into gene_list from config.json
  "hvg_names": ["ENSG00000211675", "ENSG00000211592", ...],  # Ensembl gene IDs
  "gene_to_hvg_idx": {
    "4521": 0,  # Maps original gene index to HVG position
    "4456": 1,
    ...
  },
  "gene_name_to_hvg_idx": {  # For VCC data only
    "ENSG00000211675": 0,
    "ENSG00000211592": 1,
    ...
  },
  "total_genes": 36601,  # Total genes in original dataset
  "statistics": {
    "total_cells_analyzed": 1000000,
    "expressed_genes": 18234,  # Genes expressed in >=0.1% of cells
    "n_hvgs": 2000,
    "mean_dispersion": 156.7,  # Average dispersion of selected HVGs
    "median_dispersion": 89.4,
    "min_dispersion": 9.73,
    "max_dispersion": 24818.79,
    "mean_expression": 2.34,  # Average expression level of HVGs
    "median_pct_cells": 15.6,  # Median % of cells expressing HVGs
    "top_10_hvgs": [...],  # Most variable genes
    "top_10_dispersions": [...],  # Their dispersion values
    "top_10_pct_cells": [...]  # Percentage of cells expressing each
  }
}
```

### HVG Computation Details

The script implements a sophisticated cross-dataset HVG selection algorithm:

#### Mathematical Approach
HVGs are selected using the **dispersion metric** (variance/mean ratio):

1. **Cross-dataset statistics**: For each gene, compute mean expression and variance across ALL cells in the dataset
2. **Dispersion calculation**: dispersion = variance/mean - normalizes for expression level differences
3. **Expression filtering**: Exclude genes expressed in < 0.1% of cells (likely technical noise)
4. **Top-k selection**: Select top 2000 genes by dispersion

#### Why Cross-Dataset Selection?
- **Captures global variability**: HVGs represent genes that vary across different experiments, tissues, and cell types
- **Avoids batch effects**: Single-experiment HVGs might capture technical rather than biological variation
- **Maximizes information**: Genes variable across diverse contexts are most informative for ML models
- **Consistent with standards**: Matches approaches used in meta-analyses and large-scale studies

#### Biological Interpretation
**High dispersion genes typically include:**
- **Cell type markers**: Genes defining specific cell populations (CD markers, transcription factors)
- **Stress response genes**: Genes responding to experimental conditions
- **Immunoglobulin genes**: Highly variable in immune cell populations
- **Tissue-specific genes**: Genes with high expression in specific tissues

**Filtering rationale:**
- Dispersion normalizes for expression level (high-expression genes naturally have high variance)
- Rarely expressed genes (< 0.1% cells) are often noise or cell-type-specific artifacts
- Top HVGs balance biological signal with broad relevance across cell types

#### Performance Characteristics
- **Memory efficient**: Uses online statistics - memory scales with genes (~36k), not cells
- **Numerically stable**: Uses parallel Welford's algorithm for precise variance computation
- **Computationally fast**: Single pass through data, ~1-2 minutes for 1M cells
- **Deterministic**: Same input always produces same HVG selection
- **Scalable**: Handles datasets with 10M+ cells without memory issues

## Data Normalization (Critical!)

### The Normalization Problem

**The downloaded data contains raw UMI counts without normalization**, which creates several issues:

1. **Sequencing depth variation**: Cells have vastly different total UMI counts (1k-20k+)
2. **Invalid comparisons**: A gene with 10 counts in a 1,000-UMI cell (1%) vs 10 counts in a 10,000-UMI cell (0.1%) 
3. **Batch effects**: Experiments with different sequencing protocols will have systematic differences
4. **Model bias**: ML models may learn to distinguish experiments rather than biological features

### Recommended Normalization Approaches

**1. Log-CPM (Most Common)**
```python
import torch

def normalize_log_cpm(counts):
    """Normalize to log(CPM + 1)"""
    # Calculate total counts per cell
    total_counts = counts.sum(dim=1, keepdim=True)
    
    # Convert to CPM (Counts Per Million)
    cpm = (counts.float() / total_counts) * 1e6
    
    # Log transform
    return torch.log1p(cpm)

# Apply during training
expression_normalized = normalize_log_cpm(expression)
```

**2. DESeq2-style (Size Factor)**
```python
import numpy as np

def deseq2_normalize(counts):
    """DESeq2-style normalization"""
    # Calculate geometric mean for each gene
    gene_means = np.exp(np.log(counts + 1).mean(axis=0)) - 1
    
    # Calculate size factors for each cell
    size_factors = np.median(counts / gene_means[None, :], axis=1)
    
    # Normalize
    return counts / size_factors[:, None]
```

**3. Scanpy Standard Pipeline**
```python
import scanpy as sc

def scanpy_normalize(adata):
    """Standard scanpy normalization pipeline"""
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Normalize to 10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    sc.pp.log1p(adata)
    
    return adata
```

### Implementation in Training Loop

```python
class NormalizedDataLoader:
    def __init__(self, dataloader, norm_method='log_cpm'):
        self.dataloader = dataloader
        self.norm_method = norm_method
    
    def __iter__(self):
        for batch_data, metadata in self.dataloader:
            if self.norm_method == 'log_cpm':
                # Log-CPM normalization
                total_counts = batch_data.sum(dim=1, keepdim=True)
                cpm = (batch_data.float() / total_counts) * 1e6
                normalized_data = torch.log1p(cpm)
            elif self.norm_method == 'log':
                # Simple log normalization
                normalized_data = torch.log1p(batch_data.float())
            
            yield normalized_data, metadata

# Usage
raw_dataloader = create_hvg_dataloader("data/scRNA/processed", batch_size=64)
normalized_dataloader = NormalizedDataLoader(raw_dataloader, norm_method='log_cpm')
```

## Data Quality and Validation

### Quality Metrics Tracked
The download script automatically calculates key quality metrics:

- **Sparsity**: Typical values 90-95% (most genes not expressed in most cells)
- **Max expression**: Usually 20k-40k UMI counts (higher suggests doublets)
- **Cells per gene**: Genes expressed in <0.1% of cells are often noise
- **UMI counts per cell**: Stored in `obs/umi_count` for quality filtering

### Data Validation
After download, validate your dataset:

```python
import h5py
import numpy as np

# Check batch consistency
with h5py.File('data/scRNA/processed/batch_0000.h5', 'r') as f:
    print(f"Genes per batch: {f['genes'].shape[0]}")
    print(f"Cells in batch: {f['X'].shape[0]}")
    print(f"Sparsity: {(f['X'][:] == 0).mean():.2%}")
    print(f"Max expression: {f['X'][:].max()}")
```

### Common Issues
- **Gene count variation**: Different experiments may have different gene sets
- **Batch effects**: Experiments from different labs/protocols may cluster separately  
- **Doublets**: Cells with unusually high UMI counts (>50k) may be doublets
- **Empty droplets**: Cells with very low gene counts (<200 genes) may be empty

### HVG Processing Limitations

**Current Implementation Issues:**
1. **No batch effect correction**: HVGs might be biased toward genes with high technical variation between experiments
2. **No log transformation**: Uses raw counts rather than log-transformed values (common in scRNA-seq analysis)

**Recommended Improvements:**
- Consider log(counts + 1) transformation before computing dispersion
- Add batch effect assessment in HVG visualization
- Implement alternative methods (e.g., highly variable features from Seurat v4)

## Technical Considerations

### Download Performance
- **Network speed**: Downloads ~50-100 MB/s from Google Cloud Storage
- **Storage requirements**: ~70-80 KB per cell for full gene set
- **Memory usage**: Peak ~8GB RAM during batch processing (10 experiments)
- **Disk usage**: Raw files 2x larger than processed (can be deleted)

### Batch Size Selection
- **Small batches (5-10)**: Lower memory, more files, slower processing
- **Large batches (20-50)**: Higher memory, fewer files, faster processing  
- **Recommended**: 10-20 experiments per batch for balanced performance

### Gene Set Consistency
The script ensures all batches have identical gene ordering:
- Downloads each experiment with its native gene set
- Computes union of all genes across experiments
- Reindexes each batch to the full gene union with zero-filling
- Results in consistent [n_cells, 36601] matrices

### Storage Optimization
- **Int16 format**: Reduces storage by 50% vs int32 (max count ~32k)
- **HDF5 compression**: Gzip level 4 provides 60-80% size reduction
- **Dense arrays**: Optimized for PyTorch (sparse operations are slow)
- **Batch splitting**: Enables incremental loading and parallel processing

### Memory Estimates

- **Full genes (36,601)**: ~73KB per cell
- **HVG only (2,000)**: ~4KB per cell (94.5% reduction)
- **1M cells**: ~4GB with HVGs vs ~73GB full

## Performance Tips

1. **Pre-download planning**: Check available disk space (estimate 80KB per cell)
2. **Batch size tuning**: Use larger batches (15-25) for faster processing if you have RAM
3. **HVG preprocessing**: Always run `preprocess_hvgs.py` before training to reduce memory 18x
4. **Parallel loading**: Use multiple DataLoader workers for training
5. **SSD storage**: Store processed batches on SSD for faster I/O during training
6. **Raw file cleanup**: Delete raw/ directory after processing to save 50% disk space

## Troubleshooting

### Common Download Issues

**"No module named 'gcsfs'"**
```bash
pip install gcsfs pyarrow
```

**"Permission denied" or "Access denied"**
- The scBaseCount dataset is publicly accessible, no authentication needed
- Check internet connection and firewall settings

**"Out of memory" during processing**
- Reduce `--batch-size` parameter (try 5 instead of 10)
- Close other memory-intensive applications
- Consider downloading in smaller chunks

**Interrupted downloads**
- The script automatically resumes - just run the same command again
- Previously downloaded files in raw/ directory will be skipped
- Partially processed batches will be regenerated

**Inconsistent gene counts between batches**
- This is normal - the script handles gene set differences automatically
- All final batches will have identical gene ordering
- Check the `gene_list` in config.json for the unified gene set

**HVG processing issues**
- "FileNotFoundError: config.json" - Run download.py first to create the dataset
- "No batch files found" - Check that --data-dir points to processed/ directory with batch_*.h5 files
- Very low HVG dispersions - May indicate log transformation is needed for your data
- Memory errors during HVG computation - Reduce batch size in download.py and re-download

**Training/Model issues**
- Model only learns to distinguish experiments - Apply normalization! Raw counts cause batch effects
- Poor convergence or weird loss patterns - Check if you're normalizing the data properly
- Extremely high or low expression values - Verify you're using log-transformed data
- "RuntimeError: Expected floating point type" - Convert int16 raw counts to float before normalization

### Validation Commands

Check download status:
```bash
# Count total cells downloaded
python -c "
import json
config = json.load(open('data/scRNA/config.json'))
print(f'Downloaded: {config[\"total_cells_downloaded\"]:,} cells')
print(f'Samples: {config[\"total_samples\"]} experiments')
"

# Check batch files
ls -lh data/scRNA/processed/batch_*.h5 | wc -l
```

Validate HVG processing:
```bash
# Check HVG selection results
python -c "
import json
import numpy as np
hvg_info = json.load(open('data/scRNA/hvg_info.json'))
stats = hvg_info['statistics']
print(f'HVGs selected: {len(hvg_info[\"hvg_indices\"])}')
print(f'From {stats[\"expressed_genes\"]:,} expressed genes')
print(f'Dispersion range: {stats[\"min_dispersion\"]:.1f} - {stats[\"max_dispersion\"]:.1f}')
print(f'Top 5 HVGs: {hvg_info[\"hvg_names\"][:5]}')
"

# Verify HVG data loading
python -c "
import h5py
import json
import numpy as np
hvg_info = json.load(open('data/scRNA/hvg_info.json'))
with h5py.File('data/scRNA/processed/batch_0000.h5', 'r') as f:
    hvg_data = f['X'][:, hvg_info['hvg_indices']]
    total_counts = f['obs/umi_count'][:]
    print(f'HVG matrix shape: {hvg_data.shape}')
    print(f'Sparsity: {(hvg_data == 0).mean():.1%}')
    print(f'Raw count range: {hvg_data.min()} - {hvg_data.max()}')
    print(f'Total UMI range: {total_counts.min()} - {total_counts.max()}')
    print(f'WARNING: Data is raw counts - normalization required!')
"
```

## Examples

### Download Examples

```bash
# Quick test download (100k cells, ~8GB)
python dataset/download.py --cell-count 100000

# Medium dataset for development (1M cells, ~80GB) 
python dataset/download.py --cell-count 1000000 --batch-size 15

# Large dataset for production (10M cells, ~800GB)
python dataset/download.py --cell-count 10000000 --batch-size 25 --output-dir /scratch/scRNA
```

### Basic Training Example
See `train.py` for a complete training example with:
- HVG preprocessing integration
- Custom tokenization strategies (direct, binned, log-scale)
- Simple transformer architecture
- Training loop with loss tracking

### Mini Discrete Diffusion Model
`train_mini_diffusion.py` provides a scaled-down discrete diffusion transformer:
- ~50M parameters (vs 17B full model)
- 512 hidden dim, 8 attention heads, 6 transformer layers
- Discrete diffusion with masking noise schedule
- Full training pipeline with evaluation and checkpointing
- Designed for sanity checking before large-scale training

```bash
# Train mini diffusion model
python train_mini_diffusion.py
```

### 17B Model Architecture
`model_17b.py` contains the full 17B parameter discrete diffusion transformer:
- 24 dense transformer layers + 12 MoE layers
- Top-2 routing with 16 experts per MoE layer
- 3072 hidden dim, 24 attention heads
- Perturbation conditioning with LoRA adapters
- Optimized for distributed training with DeepSpeed

## Citation

If you use this data, please cite the Arc Institute's scBaseCount dataset. 