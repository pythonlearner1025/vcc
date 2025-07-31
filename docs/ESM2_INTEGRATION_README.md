# ESM2 Integration for Diffusion Model

This document describes how to generate ESM2 embeddings for genes and integrate them into the diffusion model.

## Step 1: Generate ESM2 Embeddings

Use the batch processing script to generate embeddings for all HVG genes:

```bash
# For A100 GPU with 16GB+ memory
python scripts/esm2_batch.py \
    --hvg_file hvg_seuratv3_2000.txt \
    --out esm_all.pt \
    --device cuda \
    --batch_size 16 \
    --parallel_workers 10 \
    --rate_limit 10 \
    --model_size 650M

# For larger GPU (A100 80GB)
python scripts/esm2_batch.py \
    --hvg_file hvg_seuratv3_2000.txt \
    --out esm_all.pt \
    --device cuda \
    --batch_size 32 \
    --parallel_workers 15 \
    --rate_limit 15 \
    --model_size 3B  # Use larger model

# If getting 429 (Too Many Requests) errors from Ensembl
python scripts/esm2_batch.py \
    --hvg_file hvg_seuratv3_2000.txt \
    --out esm_all.pt \
    --device cuda \
    --batch_size 16 \
    --parallel_workers 5 \
    --rate_limit 5 \
    --model_size 650M
```

### Parameters:
- `--hvg_file`: Path to file containing gene IDs (one per line)
- `--out`: Output file path for embeddings (default: esm_all.pt)
- `--device`: Device to use (cuda/cpu)
- `--batch_size`: Batch size for ESM2 inference (16-32 for A100)
- `--parallel_workers`: Number of parallel workers for Ensembl API calls (default: 10)
- `--rate_limit`: Max concurrent Ensembl requests (default: 10, reduce if getting 429 errors)
- `--model_size`: ESM2 model size (650M or 3B)

### Expected Runtime:
- Ensembl fetching: ~5-10 minutes with 20 parallel workers
- ESM2 embedding: ~10-20 minutes on A100 for 2000 genes

## Step 2: Integration with Diffusion Model

The diffusion model has been updated to use `AugmentedGeneEmbedding` which combines:
1. Trainable gene ID embeddings (128-dim)
2. Frozen ESM2 protein embeddings (1280-dim for 650M model, 2560-dim for 3B)

### Key Features:
- **Automatic fallback**: If `esm_all.pt` is not found, the model uses only ID embeddings
- **Frozen ESM2 weights**: ESM2 embeddings are not updated during training
- **Gated fusion**: A learnable gate controls the contribution of ESM2 features
- **Handles non-coding genes**: Genes without protein sequences get zero ESM2 vectors

### Configuration:
The model config now includes ESM2 parameters:
```python
@dataclass
class ConditionalModelConfig:
    # ... other parameters ...
    
    # ESM2 embeddings
    esm_matrix_path: str = "esm_all.pt"  # Path to precomputed ESM2 embeddings
    esm_proj_dim: int = 256  # Projection dimension for ESM2 embeddings
```

## Step 3: Testing

Run the test script to verify integration:

```bash
python test_esm2_integration.py
```

This will:
1. Test the AugmentedGeneEmbedding module
2. Verify model initialization with ESM2 embeddings
3. Check that ESM2 weights are frozen
4. Test fallback behavior when embeddings are missing

## Technical Details

### Storage Format
The `esm_all.pt` file contains:
```python
{
    "emb": torch.FloatTensor[n_total_isoforms, esm_dim],  # All isoform embeddings (half precision)
    "genes": List[str],  # Original gene IDs in order
    "protein_ids": List[str],  # Protein IDs matching emb rows
    "gene_to_isoform_indices": Dict[str, List[int]]  # Maps gene_id to indices in emb
}
```

### Multi-Isoform Support
The system now handles multiple protein isoforms per gene:
- All protein isoforms are fetched and embedded
- The canonical isoform (marked in Ensembl) is stored first
- The `gene_to_isoform_indices` mapping allows lookup of all isoforms for a gene
- By default, the model uses the canonical (first) isoform
- Future extensions could allow isoform-specific modeling

### Memory Requirements
- ESM2-650M embeddings: ~2.5MB per 1000 isoforms (float16, 1280-dim)
- ESM2-3B embeddings: ~5MB per 1000 isoforms (float16, 2560-dim)
- For 2000 genes with ~1.5x isoforms on average: ~7.5MB (650M) or ~15MB (3B)
- Additional GPU memory during inference: ~2-4GB for ESM2 model

### Architecture
The `AugmentedGeneEmbedding` module:
1. Looks up trainable ID embedding
2. Looks up frozen ESM2 embedding (if available)
3. Projects ESM2 embedding to smaller dimension
4. Applies learnable gate to ESM2 features
5. Concatenates and mixes both embeddings
6. Returns fused embedding matching original dimension

This design allows the model to leverage biological sequence information while maintaining the same interface and dimensionality as the original gene embeddings.