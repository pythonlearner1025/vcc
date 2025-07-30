import scanpy as sc

# Load the adata
adata = sc.read_h5ad('../data/vcc_data/adata_Training.h5ad')

# Create bidirectional mappings
#gene_to_ensembl = dict(zip(adata.var.index, adata.var['gene_id']))
#ensembl_to_gene = dict(zip(adata.var['gene_id'], adata.var.index))

# Example usage:
#print(gene_to_ensembl['SAMD11'])  # Returns: ENSG00000187634
#print(ensembl_to_gene['ENSG00000187634'])  # Returns: SAMD11

def get_perturbation_and_control_cells(adata, target_gene):
    """
    Get perturbation and control cell sets for a specific target gene.
    
    Returns:
        perturbed_cells: Index of cells where target_gene was perturbed
        control_cells: Index of control cells
    """
    # Get perturbed cells
    perturbed_mask = adata.obs['target_gene'] == target_gene
    perturbed_cells = adata.obs.index[perturbed_mask]
    target_batches = adata.obs.loc[perturbed_mask, 'batch'].unique()
    print(len(target_batches))
    
    # Get control cells
    control_mask = (adata.obs['target_gene'] == 'non-targeting') & \
                   (adata.obs['batch'].isin(target_batches))
    control_cells = adata.obs.index[control_mask]
    
    return perturbed_cells, control_cells

import numpy as np

def get_balanced_perturbation_control_pairs(adata, target_gene, n_cells_per_batch=16):
    """
    Get balanced pairs of perturbed and control cells from each batch.
    
    Args:
        adata: AnnData object
        target_gene: Target gene name
        n_cells_per_batch: Number of cells to sample per group per batch
    
    Returns:
        Dictionary with batch names as keys, each containing:
        - 'perturbed': array of perturbed cell indices
        - 'control': array of control cell indices
    """
    # Get all perturbed cells for this gene
    perturbed_mask = adata.obs['target_gene'] == target_gene
    target_batches = adata.obs.loc[perturbed_mask, 'batch'].unique()
    
    batch_pairs = {}
    
    for batch in target_batches:
        # Get perturbed cells in this batch
        batch_perturbed_mask = (adata.obs['target_gene'] == target_gene) & \
                               (adata.obs['batch'] == batch)
        batch_perturbed_cells = adata.obs.index[batch_perturbed_mask].values
        
        # Get control cells in this batch
        batch_control_mask = (adata.obs['target_gene'] == 'non-targeting') & \
                             (adata.obs['batch'] == batch)
        batch_control_cells = adata.obs.index[batch_control_mask].values
        
        # Sample n_cells_per_batch from each group
        # Use replace=False if enough cells, else sample with replacement
        n_perturbed = len(batch_perturbed_cells)
        n_control = len(batch_control_cells)
        
        if n_perturbed >= n_cells_per_batch:
            sampled_perturbed = np.random.choice(batch_perturbed_cells, 
                                                n_cells_per_batch, 
                                                replace=False)
        else:
            print(f"Warning: Batch {batch} has only {n_perturbed} perturbed cells, sampling with replacement")
            sampled_perturbed = np.random.choice(batch_perturbed_cells, 
                                                n_cells_per_batch, 
                                                replace=True)
        
        if n_control >= n_cells_per_batch:
            sampled_control = np.random.choice(batch_control_cells, 
                                              n_cells_per_batch, 
                                              replace=False)
        else:
            print(f"Warning: Batch {batch} has only {n_control} control cells, sampling with replacement")
            sampled_control = np.random.choice(batch_control_cells, 
                                              n_cells_per_batch, 
                                              replace=True)
        
        batch_pairs[batch] = {
            'perturbed': sampled_perturbed,
            'control': sampled_control
        }
    
    return batch_pairs

# Example usage:
batch_pairs = get_balanced_perturbation_control_pairs(adata, 'EIF3H', n_cells_per_batch=16)

# Print summary
for batch, cells in batch_pairs.items():
    print(f"\nBatch {batch}:")
    print(f"  Perturbed cells: {len(cells['perturbed'])}")
    print(f"  Control cells: {len(cells['control'])}")

# Check validation genes
import pandas as pd

print("\n" + "="*80)
print("CHECKING VALIDATION GENES")
print("="*80)

# Load validation genes
val_df = pd.read_csv('../data/vcc_data/pert_counts_Validation.csv')
validation_genes = val_df['target_gene'].tolist()
print(f"\nNumber of validation genes: {len(validation_genes)}")

# Check which validation genes were perturbed in training
perturbed_genes_in_training = adata.obs['target_gene'].unique()
perturbed_genes_in_training = [g for g in perturbed_genes_in_training if g != 'non-targeting']

validation_perturbed_in_training = []
for gene in validation_genes:
    if gene in perturbed_genes_in_training:
        validation_perturbed_in_training.append(gene)
        n_cells = (adata.obs['target_gene'] == gene).sum()
        print(f"  {gene}: {n_cells} cells perturbed in training")

print(f"\nValidation genes perturbed in training: {len(validation_perturbed_in_training)}/{len(validation_genes)} ({100*len(validation_perturbed_in_training)/len(validation_genes):.1f}%)")

# Check which validation genes exist in the expression matrix
genes_in_expr_matrix = adata.var.index.tolist()
validation_in_expr_matrix = []
validation_not_in_expr_matrix = []

for gene in validation_genes:
    if gene in genes_in_expr_matrix:
        validation_in_expr_matrix.append(gene)
    else:
        validation_not_in_expr_matrix.append(gene)

print(f"\nValidation genes in expression matrix: {len(validation_in_expr_matrix)}/{len(validation_genes)} ({100*len(validation_in_expr_matrix)/len(validation_genes):.1f}%)")

if validation_not_in_expr_matrix:
    print(f"Genes NOT in expression matrix: {validation_not_in_expr_matrix}")

# Conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if validation_perturbed_in_training:
    print(f"\n✓ GENE COUNT DATA EXISTS for {len(validation_perturbed_in_training)} validation genes:")
    print(f"  These genes were perturbed in training and have expression data available.")
    print(f"  Genes: {', '.join(validation_perturbed_in_training)}")
else:
    print("\n✗ NO GENE COUNT DATA from perturbed cells for validation genes:")
    print("  None of the validation genes were perturbed in the training dataset.")
    print("  The model has never seen the perturbation effects of these genes.")

if validation_in_expr_matrix:
    print(f"\n✓ However, expression values ARE AVAILABLE for {len(validation_in_expr_matrix)}/{len(validation_genes)} validation genes:")
    print("  These genes exist in the expression matrix, so their baseline expression")
    print("  can be observed in control cells and cells perturbed with other genes.")

# Example: Check one validation gene
print("\n" + "="*80)
print("EXAMPLE: Checking 'SH3BP4' (first validation gene)")
print("="*80)

# Check if SH3BP4 was perturbed
example_gene = 'SH3BP4'
perturbed_count = (adata.obs['target_gene'] == example_gene).sum()
print(f"\nCells with {example_gene} perturbation: {perturbed_count}")

# Check if SH3BP4 exists in expression matrix
if example_gene in genes_in_expr_matrix:
    gene_idx = adata.var.index.get_loc(example_gene)
    print(f"{example_gene} found at index {gene_idx} in expression matrix")
    
    # Get expression in control cells
    control_mask = adata.obs['target_gene'] == 'non-targeting'
    control_expr = adata[control_mask, example_gene].X.flatten()
    print(f"\n{example_gene} expression in control cells:")
    print(f"  Mean: {control_expr.mean():.2f}")
    print(f"  Median: {np.median(control_expr):.2f}")
    print(f"  Max: {control_expr.max():.2f}")
