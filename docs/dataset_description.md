# VCC and scRNA Dataset Description

This document provides detailed information about the two primary datasets used in this project: the VCC (CRISPR perturbation) dataset and the scRNA (single-cell RNA sequencing) dataset.

## Overview

Both datasets contain single-cell gene expression data but serve different purposes:
- **scRNA dataset**: Reference single-cell RNA-seq data from human samples
- **VCC dataset**: CRISPR perturbation data with known gene knockdowns

## Dataset Details

### scRNA Dataset

**Location**: `/data/scRNA/raw/`

**Format**: Multiple `.h5ad` files (AnnData format)

**Structure**:
- **Samples**: 11 `.h5ad` files from different experiments
- **Total genes**: 36,601 genes
- **Gene identifiers**: Ensembl IDs (e.g., `ENSG00000243485`)
- **Expression matrix**: Sparse CSC matrix format
- **Data type**: `float32` values (raw counts)

**Gene Information** (`adata.var`):
- Index: Ensembl gene IDs
- Columns:
  - `gene_symbols`: Human-readable gene names
  - `feature_types`: Type of feature (e.g., gene, protein)

**Sample Files**:
```
ERX11467034.h5ad (44MB)
SRX14182559.h5ad (76MB)
SRX10268256.h5ad (9.6MB)
SRX13549281.h5ad (17MB)
SRX14989041.h5ad (6.6MB)
SRX15182500.h5ad (9.2MB)
SRX17689200.h5ad (25MB)
SRX19704153.h5ad (17MB)
```

### VCC Dataset

**Location**: `/data/vcc_data/adata_Training.h5ad`

**Format**: Single `.h5ad` file (AnnData format)

**Structure**:
- **Total genes**: 18,080 genes
- **Gene identifiers**: 
  - Index: Gene symbols (e.g., `SAMD11`, `NOC2L`)
  - `gene_id` column: Ensembl IDs (e.g., `ENSG00000187634`)
- **Perturbation information**: Available in `adata.obs['target_gene']`

**Gene Information** (`adata.var`):
- Index: Gene symbols (human-readable names)
- Columns:
  - `gene_id`: Ensembl gene IDs matching those in scRNA data

**Key Features**:
- Contains CRISPR perturbation experiments
- Each cell has a known target gene that was perturbed
- Includes control cells with 'non-targeting' guides

## Gene Identifier Alignment

The two datasets use different gene identifier systems by default:

### Default State:
- **scRNA**: Uses Ensembl IDs in the index
- **VCC**: Uses gene symbols in the index (but provides Ensembl IDs in `gene_id` column)

### Alignment Process:
To find common genes between datasets, the VCC index must be converted to Ensembl IDs:
```python
vcc_adata.var.index = vcc_adata.var['gene_id']
```

### Common Genes:
- **Total common genes**: 18,080 (all VCC genes exist in scRNA)
- **Genes only in scRNA**: 18,521
- **Genes only in VCC**: 0

This indicates that the VCC dataset contains a curated subset of genes that are all present in the broader scRNA dataset.

## Data Processing Considerations

### Expression Values:
- **scRNA**: Raw count data stored as float32
- **VCC**: Processed expression data
- **No negative values** in the raw data (important for count-based models)

### Memory Optimization:
When processing these datasets together:
1. Use sparse matrix representations when possible
2. Consider subsampling for testing (both datasets support efficient slicing)
3. Align gene sets before concatenation to avoid memory overhead

### Gene Filtering:
Common preprocessing steps:
- `sc.pp.filter_cells(adata, min_counts=1)`: Remove empty cells
- `sc.pp.filter_genes(adata, min_cells=3)`: Remove rarely expressed genes

## Known Issues

### Integer Overflow in Batch Processing

If you encounter batch files (e.g., `batch_0000.h5`, `batch_0001.h5`) with negative values, this is likely due to integer overflow from an older processing pipeline that used `int16` data type.

**Issue Details**:
- `int16` range: -32,768 to 32,767
- Expression values > 32,767 wrap around to negative values
- Example: Gene 4377 commonly shows this issue with very high expression

**Solution**:
Use the raw `.h5ad` files instead, which store data as `float32` and don't have this limitation.

## Usage Example

```python
import scanpy as sc
import anndata

# Load scRNA data
scrna_files = glob.glob("data/scRNA/raw/*.h5ad")
scrna_list = [sc.read_h5ad(f) for f in scrna_files]
scrna_adata = anndata.concat(scrna_list, join='outer')

# Load VCC data
vcc_adata = sc.read_h5ad("data/vcc_data/adata_Training.h5ad")

# Align gene identifiers
vcc_adata.var.index = vcc_adata.var['gene_id']

# Find common genes
common_genes = scrna_adata.var.index.intersection(vcc_adata.var.index)

# Subset to common genes
scrna_adata = scrna_adata[:, common_genes]
vcc_adata = vcc_adata[:, common_genes]
```

## Important Notes

1. **Gene Order**: After subsetting to common genes, ensure both datasets have the same gene ordering before model training
2. **Batch Effects**: The scRNA data comes from multiple experiments and may contain batch effects
3. **Perturbation Effects**: The VCC data contains cells with perturbed genes that should be handled appropriately in downstream analysis
4. **Sparse vs Dense**: Consider memory implications when converting between sparse and dense representations 